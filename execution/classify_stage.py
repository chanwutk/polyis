"""Classify thread: GPU tile classifier over one decoded video at a time."""

from __future__ import annotations

import json
import os
import queue

import cv2
import numpy as np
import torch

from polyis.io import cache, store
from polyis.train.select_model_optimization import select_model_optimization

from scripts.p020_exec_classify import classify_batch as _classify_batch
from scripts.p020_exec_classify import load_model

from execution.config import PipelineConfig
from execution.messages import ClassifiedVideo, DecodedVideo


def classify_stage(
    *,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
):
    """Main loop of the classify thread."""
    device = f'cuda:{config.classify_gpu}'
    torch.cuda.set_device(config.classify_gpu)

    model = _load_and_optimize_model(config, device)
    normalize_mean = torch.tensor(
        [0.485, 0.456, 0.406] * 2, device=device, dtype=torch.float16,
    ).view(1, 6, 1, 1)
    normalize_std = torch.tensor(
        [0.229, 0.224, 0.225] * 2, device=device, dtype=torch.float16,
    ).view(1, 6, 1, 1)

    always_relevant_path = cache.index(
        config.dataset, 'never-relevant', f'{config.tile_size}_all.npy',
    )
    assert os.path.exists(always_relevant_path), \
        f"Always relevant bitmap not found at {always_relevant_path}"
    always_relevant_mask = (
        torch.from_numpy(np.load(always_relevant_path).flatten())
        .to(device).to(torch.uint8)
    )

    with torch.no_grad():
        while True:
            msg = in_queue.get()
            if msg is None:
                out_queue.put(None)
                return
            assert isinstance(msg, DecodedVideo)
            out_queue.put(_classify_one_video(
                msg=msg,
                config=config,
                device=device,
                model=model,
                normalize_mean=normalize_mean,
                normalize_std=normalize_std,
                always_relevant_mask=always_relevant_mask,
            ))


def _classify_one_video(
    *,
    msg: DecodedVideo,
    config: PipelineConfig,
    device: str,
    model: torch.nn.Module,
    normalize_mean: torch.Tensor,
    normalize_std: torch.Tensor,
    always_relevant_mask: torch.Tensor,
) -> ClassifiedVideo:
    """Run classifier batches for one decoded video."""
    grid_width = msg.width // config.tile_size
    grid_height = msg.height // config.tile_size
    y_idx = torch.arange(grid_height, device=device, dtype=torch.uint8)
    x_idx = torch.arange(grid_width, device=device, dtype=torch.uint8)
    y_rep = y_idx.repeat_interleave(grid_width)
    x_rep = x_idx.repeat(grid_height)
    positions = torch.stack([y_rep, x_rep], dim=1).float()

    grids: list[np.ndarray] = []
    for batch_start in range(0, len(msg.sampled_indices), config.classify_batch_size):
        batch_end = min(batch_start + config.classify_batch_size, len(msg.sampled_indices))
        frame_positions = msg.batch_positions[batch_start:batch_end]
        prev_positions = msg.prev_positions[batch_start:batch_end]
        batch_frames = [msg.frames_rgb[p] for p in frame_positions]
        batch_prev_frames = [msg.frames_rgb[p] for p in prev_positions]

        probs, _runtime = _classify_batch(
            grid_width=grid_width,
            grid_height=grid_height,
            positions=positions,
            batch_frames=batch_frames,
            batch_prev_frames=batch_prev_frames,
            model=model,
            tile_size=config.tile_size,
            device=device,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            always_relevant_mask=always_relevant_mask,
        )
        grids.extend(np.asarray(probs.cpu().numpy(), dtype=np.uint8))

    classifications = np.stack(grids, axis=0)
    return ClassifiedVideo(
        video=msg.video,
        classifications=classifications,
        frames_rgb=msg.frames_rgb,
        width=msg.width,
        height=msg.height,
        frame_count=msg.frame_count,
        sampled_indices=msg.sampled_indices,
        buffer_frame_indices=msg.buffer_frame_indices,
    )


def _load_and_optimize_model(config: PipelineConfig, device: str) -> torch.nn.Module:
    """Load the classifier model and apply the benchmarked optimization."""
    model = load_model(config.dataset, config.tile_size, config.classifier, device)
    model = model.to(device)

    bench_path = cache.index(
        config.dataset, 'training', 'results',
        f'{config.classifier}_{config.tile_size}', 'model_compilation.jsonl',
    )
    with open(bench_path, 'r') as f:
        benchmark_results = [json.loads(line) for line in f]

    videoset_dir = store.dataset(config.dataset, config.videoset)
    first_video = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )[0]
    cap = cv2.VideoCapture(store.dataset(config.dataset, config.videoset, first_video))
    assert cap.isOpened()
    try:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    model, _method = select_model_optimization(
        model, benchmark_results, device, config.tile_size,
        (vid_w // config.tile_size) * (vid_h // config.tile_size),
    )
    return model
