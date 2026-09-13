"""Per-stage test for ``execution.prune_stage`` (worker process body).

Feeds a synthetic VideoClassifications and asserts that a pruned
VideoClassifications comes back.  Runs the worker in-thread (calling
``prune_worker`` directly with queue.Queue stand-ins for mp.Queue) so the
test does not need a full process spawn.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading

import numpy as np
import pytest

from execution.config import PipelineConfig
from execution.messages import VideoClassifications
from execution.prune_stage import prune_worker

from polyis.io import cache


def _build_config_with_prune() -> PipelineConfig:
    return PipelineConfig(
        dataset='caldot2-y05',
        videoset='valid',
        classifier='ShuffleNet05',
        tile_size=60,
        sample_rate=4,
        tilepadding='none',
        canvas_scale=1.0,
        tracker='sortcython',
        tracking_accuracy_threshold=0.8,
        relevance_threshold=0.5,
        classify_gpu=0,
        detect_gpu=0,
        prune_workers=1,
        compress_workers=1,
        max_videos_in_flight=1,
        classify_batch_size=16,
        detect_batch_size=4,
        no_interpolate=False,
        warmup=False,
    )


def _make_synthetic_classifications(num_frames: int, grid_h: int, grid_w: int) -> np.ndarray:
    """Build VideoClassifications entries with a few relevant tiles per frame."""
    classifications = np.zeros((num_frames, grid_h, grid_w), dtype=np.uint8)
    for idx in range(num_frames):
        classifications[idx, grid_h // 2, grid_w // 2] = 255
    return classifications


def test_prune_worker_round_trip():
    """prune_worker accepts VideoClassifications and emits the same shape back."""
    config = _build_config_with_prune()

    # The prune worker reads the max_rate_table for this dataset/tracker/tile.
    # Skip the test if that artifact isn't on the test machine.
    max_rate_path = cache.index(
        config.dataset, 'track_rates',
        f'{config.tracker}_{config.tile_size}', 'max_rate_table.npy',
    )
    if not os.path.exists(max_rate_path):
        pytest.skip(f"max_rate_table not available at {max_rate_path}")

    # Match the grid dimensions of the dataset's precomputed max_rate_table;
    # the prune stage asserts that they're equal.
    max_rate_table = np.load(max_rate_path)
    grid_h, grid_w = max_rate_table.shape[:2]
    num_frames = 5

    msg = VideoClassifications(
        video='va00.mp4',
        classifications=_make_synthetic_classifications(num_frames, grid_h, grid_w),
        width=grid_w * config.tile_size,
        height=grid_h * config.tile_size,
        frame_count=num_frames,
        sampled_indices=list(range(num_frames)),
    )

    in_q: mp.Queue = mp.Queue()
    out_q: mp.Queue = mp.Queue()

    t = threading.Thread(
        target=prune_worker,
        args=(in_q, out_q, config),
        daemon=True,
    )
    t.start()

    in_q.put(msg)
    in_q.put(None)

    result = out_q.get(timeout=60)
    assert isinstance(result, VideoClassifications)
    assert result.video == 'va00.mp4'
    # Pruning is a per-tile selection; the result should preserve frame count.
    assert result.classifications.shape[0] == num_frames
    # And the same metadata is passed through.
    assert result.width == grid_w * config.tile_size

    assert out_q.get(timeout=5) is None
    t.join(timeout=5)
