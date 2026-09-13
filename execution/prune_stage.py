"""Prune worker process pool: polyomino grouping + ILP via Gurobi."""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from polyis.io import cache
from polyis.pack.adapters import group_tiles_all
from polyis.sample.ilp.c.gurobi import solve_ilp

from execution.config import PipelineConfig
from execution.messages import VideoClassifications


_ALL_ACCURACY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
_ACCURACY_THRESHOLD_TO_IDX = {t: i for i, t in enumerate(_ALL_ACCURACY_THRESHOLDS)}

TILEPADDING_MODE = 0
_DEFAULT_TIME_LIMIT_S = 0.1


def prune_worker(
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    config: PipelineConfig,
):
    """Entry point for one prune worker process."""
    assert config.tracking_accuracy_threshold is not None, \
        "Prune workers should not be spawned without a threshold"
    accuracy_idx = _ACCURACY_THRESHOLD_TO_IDX[config.tracking_accuracy_threshold]
    max_rate_table = _load_max_rate_table(config)

    while True:
        msg = in_queue.get()
        if msg is None:
            out_queue.put(None)
            return

        assert isinstance(msg, VideoClassifications)
        out_queue.put(_prune_one_video(
            msg=msg,
            config=config,
            accuracy_idx=accuracy_idx,
            max_rate_table=max_rate_table,
        ))


def _load_max_rate_table(config: PipelineConfig) -> np.ndarray:
    """Load the [H, W, num_thresholds] max-rate table for this tracker/tile."""
    max_rate_path = cache.index(
        config.dataset, 'track_rates',
        f'{config.tracker}_{config.tile_size}', 'max_rate_table.npy',
    )
    assert os.path.exists(max_rate_path), \
        f"Max rate table not found at {max_rate_path}"
    table = np.load(max_rate_path)
    assert table.ndim == 3, f"Expected 3D max_rate_table, got shape {table.shape}"
    return table


def _prune_one_video(
    msg: VideoClassifications,
    config: PipelineConfig,
    accuracy_idx: int,
    max_rate_table: np.ndarray,
) -> VideoClassifications:
    """Run group_tiles_all + ILP for one video."""
    cutoff = config.relevance_threshold * 255
    bitmaps = ((msg.classifications > cutoff).astype(np.uint8))
    num_frames, grid_height, grid_width = bitmaps.shape

    tile_to_polyomino_id, polyomino_lengths = group_tiles_all(
        bitmaps, TILEPADDING_MODE,
    )
    tile_to_polyomino_id = np.asarray(tile_to_polyomino_id)

    max_sampling_distance = max_rate_table[:, :, accuracy_idx]
    assert max_sampling_distance.shape == (grid_height, grid_width), \
        f"max_rate shape {max_sampling_distance.shape} != ({grid_height},{grid_width})"
    max_sampling_distance = max_sampling_distance // config.sample_rate
    max_sampling_distance = np.maximum(max_sampling_distance, 1)

    ilp_result = solve_ilp(
        tile_to_polyomino_id,
        polyomino_lengths,
        max_sampling_distance,
        grid_height,
        grid_width,
        time_limit_seconds=_DEFAULT_TIME_LIMIT_S,
    )
    selected = ilp_result.selected

    pruned = np.zeros((num_frames, grid_height, grid_width), dtype=np.uint8)
    for b in range(num_frames):
        selected_ids = {pid for (frame, pid) in selected if frame == b}
        tile_ids = tile_to_polyomino_id[b]
        mask = np.isin(tile_ids, list(selected_ids)) & (tile_ids >= 0)
        pruned[b] = mask.astype(np.uint8) * 255

    return VideoClassifications(
        video=msg.video,
        classifications=pruned,
        width=msg.width,
        height=msg.height,
        frame_count=msg.frame_count,
        sampled_indices=msg.sampled_indices,
    )
