"""Tests for threaded compress stage behavior."""

from __future__ import annotations

import queue

import numpy as np

from execution.compress_stage import _compress_one_video, compress_worker
from execution.config import PipelineConfig
from execution.messages import ClassifiedVideo, CollageReady
from execution.pool import spawn_thread_pool


def _build_config() -> PipelineConfig:
    """Build a small CPU-only config for compress tests."""
    return PipelineConfig(
        dataset='caldot2-y05',
        videoset='valid',
        classifier='ShuffleNet05',
        tile_size=4,
        sample_rate=1,
        tilepadding='none',
        canvas_scale=1.0,
        tracker='sortcython',
        tracking_accuracy_threshold=None,
        relevance_threshold=0.5,
        classify_gpu=0,
        detect_gpu=0,
        prune_workers=1,
        compress_workers=2,
        max_videos_in_flight=2,
        classify_batch_size=16,
        detect_batch_size=4,
        no_interpolate=False,
        warmup=False,
    )


def _classified_video(video: str) -> ClassifiedVideo:
    """Create one tiny classified video with one relevant tile per frame."""
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    frames[0, :, :] = [255, 0, 0]
    frames[1, :, :] = [0, 255, 0]
    frames[2, :, :] = [0, 0, 255]
    classifications = np.zeros((3, 2, 2), dtype=np.uint8)
    classifications[0, 0, 0] = 255
    classifications[1, 0, 1] = 255
    classifications[2, 1, 0] = 255
    return ClassifiedVideo(
        video=video,
        classifications=classifications,
        frames_rgb=frames,
        width=8,
        height=8,
        frame_count=3,
        sampled_indices=[0, 1, 2],
        buffer_frame_indices=[0, 1, 2],
    )


def test_compress_one_video_emits_rgb_collage():
    """Compress renders at least one RGB canvas with valid mapping metadata."""
    out_q: queue.Queue = queue.Queue()
    _compress_one_video(_classified_video('v0.mp4'), _build_config(), out_q)

    msg = out_q.get(timeout=5)
    assert isinstance(msg, CollageReady)
    assert msg.video == 'v0.mp4'
    assert msg.canvas_rgb.shape == (8, 8, 3)
    assert msg.index_map.shape == (2, 2)
    assert len(msg.offset_lookup) >= 1


def test_compress_thread_pool_processes_multiple_videos():
    """Multiple compress threads can process independent videos concurrently."""
    upstream: queue.Queue = queue.Queue()
    downstream: queue.Queue = queue.Queue()
    config = _build_config()
    workers, _, _, relays = spawn_thread_pool(
        name='compress-test',
        worker_target=compress_worker,
        worker_args=(config,),
        num_workers=2,
        upstream_q=upstream,
        downstream_q=downstream,
    )

    expected_videos = {f'v{i}.mp4' for i in range(4)}
    for video in expected_videos:
        upstream.put(_classified_video(video))
    upstream.put(None)

    seen_videos: set[str] = set()
    while True:
        msg = downstream.get(timeout=20)
        if msg is None:
            break
        assert isinstance(msg, CollageReady)
        seen_videos.add(msg.video)

    assert seen_videos == expected_videos
    for t in workers + relays:
        t.join(timeout=5)
        assert not t.is_alive()
