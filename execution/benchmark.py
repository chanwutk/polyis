"""External wall-clock benchmark wrapper for ``execution.main``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2

from polyis.io import store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark execution.main wall-clock throughput')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('runner_args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def _arg_value(args: list[str], flag: str) -> str | None:
    """Return the value after a CLI flag, if present."""
    if flag not in args:
        return None
    idx = args.index(flag)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _count_videos_and_frames(runner_args: list[str]) -> tuple[int | None, int | None]:
    """Count videos and frames when dataset arguments are available."""
    dataset = _arg_value(runner_args, '--dataset')
    videoset = _arg_value(runner_args, '--videoset')
    if dataset is None or videoset is None:
        return None, None

    videoset_dir = store.dataset(dataset, videoset)
    videos = sorted(
        f for f in os.listdir(videoset_dir)
        if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    )
    max_videos = _arg_value(runner_args, '--max-videos')
    if max_videos is not None:
        videos = videos[:int(max_videos)]

    frame_total = 0
    for video in videos:
        cap = cv2.VideoCapture(store.dataset(dataset, videoset, video))
        try:
            frame_total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
    return len(videos), frame_total


def _write_record(path: str | None, record: dict) -> None:
    """Append a benchmark record or print it when no output path is given."""
    line = json.dumps(record, default=str)
    if path is None:
        print(line)
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a') as f:
        f.write(line + '\n')


def main() -> int:
    args = parse_args()
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == '--':
        runner_args = runner_args[1:]

    if not runner_args:
        raise SystemExit('Pass execution.main arguments after --')

    video_count, frame_count = _count_videos_and_frames(runner_args)
    returncode = 0
    for repeat in range(args.repeats):
        cmd = [sys.executable, '-m', 'execution.main', *runner_args]
        start = time.perf_counter()
        completed = subprocess.run(cmd)
        elapsed_s = time.perf_counter() - start
        returncode = completed.returncode
        record = {
            'repeat': repeat,
            'command': cmd,
            'returncode': completed.returncode,
            'elapsed_s': elapsed_s,
            'video_count': video_count,
            'frame_count': frame_count,
            'videos_per_s': (video_count / elapsed_s) if video_count is not None else None,
            'frames_per_s': (frame_count / elapsed_s) if frame_count is not None else None,
        }
        _write_record(args.output, record)
        if completed.returncode != 0:
            break

    return returncode


if __name__ == '__main__':
    raise SystemExit(main())
