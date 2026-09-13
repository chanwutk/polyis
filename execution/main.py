#!/usr/local/bin/python
"""One-command, pipeline-parallel runner for the polyis tracking pipeline."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

from polyis.io import store
from polyis.utilities import TILEPADDING_MODES

from execution.classify_stage import classify_stage
from execution.compress_stage import compress_worker
from execution.config import PipelineConfig
from execution.decode_stage import decode_stage
from execution.detect_stage import detect_stage
from execution.messages import ClassifiedVideo, TrackingResult, VideoClassifications
from execution.output import save_pipeline_runtime, save_tracking_result
from execution.pool import spawn_thread_pool
from execution.prune_stage import prune_worker
from execution.track_stage import track_stage


WARMUP_MAX_FRAMES = 64


def _parse_threshold(s: str) -> float | None:
    """Parse a tracking-accuracy threshold; accepts ``null``/``none`` for None."""
    if s.lower() in ('null', 'none'):
        return None
    return float(s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Pipeline-parallel runner for the polyis tracking pipeline'
    )

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--videoset', required=True, choices=['test', 'valid', 'train'])
    parser.add_argument('--classifier', required=True)
    parser.add_argument('--tile-size', dest='tile_size', type=int, required=True)
    parser.add_argument('--sample-rate', dest='sample_rate', type=int, required=True)
    parser.add_argument('--tilepadding', required=True, choices=list(TILEPADDING_MODES.keys()))
    parser.add_argument('--canvas-scale', dest='canvas_scale', type=float, required=True)
    parser.add_argument('--tracker', required=True)
    parser.add_argument(
        '--tracking-accuracy-threshold', dest='tracking_accuracy_threshold',
        type=_parse_threshold, required=True,
        help='Float in [0, 1] or "null"/"none" to disable the prune stage.',
    )
    parser.add_argument('--relevance-threshold', dest='relevance_threshold',
                        type=float, required=True)

    cpu_count = max(1, os.cpu_count() or 1)
    parser.add_argument('--classify-gpu', dest='classify_gpu', type=int, default=0)
    parser.add_argument('--detect-gpu', dest='detect_gpu', type=int, default=0)
    parser.add_argument('--prune-workers', dest='prune_workers', type=int,
                        default=max(1, cpu_count // 4))
    parser.add_argument('--compress-workers', dest='compress_workers', type=int,
                        default=max(2, cpu_count // 2))
    parser.add_argument('--max-videos-in-flight', dest='max_videos_in_flight',
                        type=int, default=2)
    parser.add_argument('--classify-batch-size', dest='classify_batch_size',
                        type=int, default=16)
    parser.add_argument('--detect-batch-size', dest='detect_batch_size',
                        type=int, default=4)

    parser.add_argument('--no-interpolate', dest='no_interpolate',
                        action='store_true', default=False)
    parser.add_argument('--no-warmup', dest='no_warmup',
                        action='store_true', default=False)
    parser.add_argument('--max-videos', dest='max_videos', type=int, default=None,
                        help='Debug: limit pipeline to the first N videos.')
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        dataset=args.dataset,
        videoset=args.videoset,
        classifier=args.classifier,
        tile_size=args.tile_size,
        sample_rate=args.sample_rate,
        tilepadding=args.tilepadding,
        canvas_scale=args.canvas_scale,
        tracker=args.tracker,
        tracking_accuracy_threshold=args.tracking_accuracy_threshold,
        relevance_threshold=args.relevance_threshold,
        classify_gpu=args.classify_gpu,
        detect_gpu=args.detect_gpu,
        prune_workers=args.prune_workers,
        compress_workers=args.compress_workers,
        max_videos_in_flight=args.max_videos_in_flight,
        classify_batch_size=args.classify_batch_size,
        detect_batch_size=args.detect_batch_size,
        no_interpolate=args.no_interpolate,
        warmup=not args.no_warmup,
    )


def _video_feeder(
    videos: list[str],
    video_q: queue.Queue,
    sem: threading.Semaphore,
):
    """Feed videos to the decoder, bounded by videos in flight."""
    for video in videos:
        sem.acquire()
        video_q.put(video)
    video_q.put(None)


def _to_prune_message(msg: ClassifiedVideo) -> VideoClassifications:
    """Drop decoded frames before crossing the prune process boundary."""
    return VideoClassifications(
        video=msg.video,
        classifications=msg.classifications,
        width=msg.width,
        height=msg.height,
        frame_count=msg.frame_count,
        sampled_indices=msg.sampled_indices,
    )


def _restore_pruned_context(
    pruned: VideoClassifications,
    original: ClassifiedVideo,
) -> ClassifiedVideo:
    """Attach pruned classifications to the original decoded frame context."""
    return ClassifiedVideo(
        video=pruned.video,
        classifications=pruned.classifications,
        frames_rgb=original.frames_rgb,
        width=original.width,
        height=original.height,
        frame_count=original.frame_count,
        sampled_indices=original.sampled_indices,
        buffer_frame_indices=original.buffer_frame_indices,
    )


def _prune_fan_out_relay(
    upstream_q: queue.Queue,
    prune_in_q: mp.Queue,
    context: dict[str, ClassifiedVideo],
    context_lock: threading.Lock,
    num_workers: int,
) -> None:
    """Store frame context locally and send classifications to prune workers."""
    while True:
        msg = upstream_q.get()
        if msg is None:
            for _ in range(num_workers):
                prune_in_q.put(None)
            return
        assert isinstance(msg, ClassifiedVideo)
        with context_lock:
            context[msg.video] = msg
        prune_in_q.put(_to_prune_message(msg))


def _prune_fan_in_relay(
    prune_out_q: mp.Queue,
    downstream_q: queue.Queue,
    context: dict[str, ClassifiedVideo],
    context_lock: threading.Lock,
    num_workers: int,
) -> None:
    """Reattach decoded frame context to pruned classifications."""
    nones_seen = 0
    while True:
        msg = prune_out_q.get()
        if msg is None:
            nones_seen += 1
            if nones_seen >= num_workers:
                downstream_q.put(None)
                return
            continue
        assert isinstance(msg, VideoClassifications)
        with context_lock:
            original = context.pop(msg.video)
        downstream_q.put(_restore_pruned_context(msg, original))


def _spawn_prune_pool(
    *,
    config: PipelineConfig,
    upstream_q: queue.Queue,
    downstream_q: queue.Queue,
) -> tuple[list[mp.Process], list[threading.Thread]]:
    """Spawn prune processes plus context-stripping relay threads."""
    prune_in_q: mp.Queue = mp.Queue()
    prune_out_q: mp.Queue = mp.Queue()
    workers: list[mp.Process] = []
    for i in range(config.prune_workers):
        p = mp.Process(
            target=prune_worker,
            args=(prune_in_q, prune_out_q, config),
            daemon=True,
            name=f'prune-{i}',
        )
        p.start()
        workers.append(p)

    context: dict[str, ClassifiedVideo] = {}
    context_lock = threading.Lock()
    fan_out_t = threading.Thread(
        target=_prune_fan_out_relay,
        args=(upstream_q, prune_in_q, context, context_lock, config.prune_workers),
        daemon=True,
        name='prune-fanout',
    )
    fan_in_t = threading.Thread(
        target=_prune_fan_in_relay,
        args=(prune_out_q, downstream_q, context, context_lock, config.prune_workers),
        daemon=True,
        name='prune-fanin',
    )
    fan_out_t.start()
    fan_in_t.start()
    return workers, [fan_out_t, fan_in_t]


def main() -> int:
    args = parse_args()
    config = build_config(args)

    mp.set_start_method('spawn', force=True)

    videoset_dir = store.dataset(config.dataset, config.videoset)
    assert os.path.exists(videoset_dir), f"Videoset directory {videoset_dir} does not exist"
    videos = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )
    assert len(videos) > 0, f"No videos found in {videoset_dir}"
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    print(f"Found {len(videos)} videos in {config.dataset}/{config.videoset}")

    video_q: queue.Queue = queue.Queue()
    decode_q: queue.Queue = queue.Queue()
    classify_out_q: queue.Queue = queue.Queue()
    compress_to_detect_q: queue.Queue = queue.Queue()
    detect_q: queue.Queue = queue.Queue()
    result_q: queue.Queue = queue.Queue()

    sem = threading.Semaphore(config.max_videos_in_flight)

    decode_t = threading.Thread(
        target=decode_stage,
        kwargs=dict(video_queue=video_q, out_queue=decode_q, config=config),
        daemon=True,
        name='decode',
    )
    classify_t = threading.Thread(
        target=classify_stage,
        kwargs=dict(in_queue=decode_q, out_queue=classify_out_q, config=config),
        daemon=True,
        name='classify',
    )

    prune_workers: list[mp.Process] = []
    prune_relays: list[threading.Thread] = []
    if config.use_prune:
        prune_to_compress_q: queue.Queue = queue.Queue()
        prune_workers, prune_relays = _spawn_prune_pool(
            config=config,
            upstream_q=classify_out_q,
            downstream_q=prune_to_compress_q,
        )
        compress_upstream = prune_to_compress_q
    else:
        compress_upstream = classify_out_q

    compress_workers, _, _, compress_relays = spawn_thread_pool(
        name='compress',
        worker_target=compress_worker,
        worker_args=(config,),
        num_workers=config.compress_workers,
        upstream_q=compress_upstream,
        downstream_q=compress_to_detect_q,
    )

    detect_t = threading.Thread(
        target=detect_stage,
        kwargs=dict(in_queue=compress_to_detect_q, out_queue=detect_q, config=config),
        daemon=True,
        name='detect',
    )
    track_t = threading.Thread(
        target=track_stage,
        kwargs=dict(in_queue=detect_q, out_queue=result_q, config=config),
        daemon=True,
        name='track',
    )

    decode_t.start()
    classify_t.start()
    detect_t.start()
    track_t.start()

    if config.warmup:
        print(f"Warming up with {videos[0]} ({WARMUP_MAX_FRAMES} frames)...")
        sem.acquire()
        video_q.put((videos[0], WARMUP_MAX_FRAMES))
        _ = result_q.get(timeout=300)
        sem.release()
        print("Warmup complete.")

    timer_start_ns = time.time_ns()
    feeder_t = threading.Thread(
        target=_video_feeder,
        args=(videos, video_q, sem),
        daemon=True,
        name='feeder',
    )
    feeder_t.start()

    per_video_complete_ts: dict[str, float] = {}
    results: list[TrackingResult] = []
    for i in range(len(videos)):
        result = result_q.get()
        assert isinstance(result, TrackingResult)
        elapsed_so_far_ms = (time.time_ns() - timer_start_ns) / 1e6
        per_video_complete_ts[result.video] = elapsed_so_far_ms
        results.append(result)
        save_tracking_result(result, config)
        sem.release()
        print(f"  [{i + 1}/{len(videos)}] {result.video} done at {elapsed_so_far_ms:.0f} ms")

    elapsed_ms = (time.time_ns() - timer_start_ns) / 1e6

    feeder_t.join(timeout=10)
    for t in [decode_t, classify_t, detect_t, track_t]:
        t.join(timeout=10)
    for p in prune_workers:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    for t in prune_relays + compress_relays + compress_workers:
        t.join(timeout=5)

    summary_path = save_pipeline_runtime(
        config=config,
        elapsed_ms=elapsed_ms,
        num_videos=len(videos),
        per_video_complete_ts=per_video_complete_ts,
    )

    print(f"\nPipeline complete: {len(results)} videos in {elapsed_ms:.0f} ms")
    print(f"Runtime summary: {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
