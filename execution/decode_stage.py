"""Decoder thread: reads one video and emits one whole-video message.

The decoder keeps the pipeline unit at video granularity.  For full-frame
sampling it uses OpenCV because that path is usually fastest.  For sparse
sampling it uses FFmpeg's ``select`` filter so Python only receives the
sampled frames and their previous frames.
"""

from __future__ import annotations

import queue
import subprocess

import cv2
import numpy as np

from polyis.io import store

from execution.config import PipelineConfig
from execution.messages import DecodedVideo


def decode_stage(
    *,
    video_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
):
    """Main loop of the decoder thread."""
    while True:
        item = video_queue.get()
        if item is None:
            out_queue.put(None)
            return

        # Warmup passes use a tuple to cap the number of source frames.
        if isinstance(item, tuple):
            video, max_frames = item
        else:
            video, max_frames = item, None

        out_queue.put(_decode_one_video(video, max_frames, config))


def _decode_one_video(
    video: str,
    max_frames: int | None,
    config: PipelineConfig,
) -> DecodedVideo:
    """Decode one video's needed RGB frames into a numpy array."""
    video_path = store.dataset(config.dataset, config.videoset, video)
    width, height, frame_count = _probe_with_opencv(video_path)
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)

    sampled_indices, buffer_frame_indices, batch_positions, prev_positions = \
        _compute_frame_plan(frame_count, config.sample_rate)

    if config.sample_rate == 1:
        frames = _decode_needed_with_opencv(
            video_path=video_path,
            shape=(len(buffer_frame_indices), height, width, 3),
            frame_count=frame_count,
            buffer_frame_indices=buffer_frame_indices,
        )
    else:
        frames = _decode_needed_with_ffmpeg(
            video_path=video_path,
            shape=(len(buffer_frame_indices), height, width, 3),
            frame_count=frame_count,
            sample_rate=config.sample_rate,
        )

    return DecodedVideo(
        video=video,
        frames_rgb=frames,
        width=width,
        height=height,
        frame_count=frame_count,
        sampled_indices=sampled_indices,
        buffer_frame_indices=buffer_frame_indices,
        batch_positions=batch_positions,
        prev_positions=prev_positions,
    )


def _probe_with_opencv(video_path: str) -> tuple[int, int, int]:
    """Read video width, height, and frame count using OpenCV metadata."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    assert frame_count > 0, f"Video {video_path} has no frames"
    assert width > 0 and height > 0, f"Video {video_path} has invalid resolution"
    return width, height, frame_count


def _compute_frame_plan(
    frame_count: int,
    sample_rate: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Compute sampled frames, retained buffer frames, and classify positions."""
    sampled_indices = [idx for idx in range(frame_count) if idx % sample_rate == 0]
    last_idx = frame_count - 1
    if sampled_indices[-1] != last_idx:
        sampled_indices.append(last_idx)

    needed_indices: set[int] = set()
    prev_map: dict[int, int] = {}
    for idx in sampled_indices:
        prev = idx - 1 if idx > 0 else min(1, frame_count - 1)
        needed_indices.add(idx)
        needed_indices.add(prev)
        prev_map[idx] = prev

    buffer_frame_indices = sorted(needed_indices)
    idx_to_buf_pos = {idx: pos for pos, idx in enumerate(buffer_frame_indices)}
    batch_positions = [idx_to_buf_pos[idx] for idx in sampled_indices]
    prev_positions = [idx_to_buf_pos[prev_map[idx]] for idx in sampled_indices]
    return sampled_indices, buffer_frame_indices, batch_positions, prev_positions


def _decode_needed_with_opencv(
    *,
    video_path: str,
    shape: tuple[int, int, int, int],
    frame_count: int,
    buffer_frame_indices: list[int],
) -> np.ndarray:
    """Decode needed frames with OpenCV and store them as RGB."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video {video_path}"
    frames = np.empty(shape, dtype=np.uint8)
    idx_to_buf_pos = {idx: pos for pos, idx in enumerate(buffer_frame_indices)}
    needed_set = set(buffer_frame_indices)

    try:
        for frame_idx in range(frame_count):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx in needed_set:
                frames[idx_to_buf_pos[frame_idx]] = frame_bgr[:, :, ::-1]
    finally:
        cap.release()

    return frames


def _decode_needed_with_ffmpeg(
    *,
    video_path: str,
    shape: tuple[int, int, int, int],
    frame_count: int,
    sample_rate: int,
) -> np.ndarray:
    """Decode sampled and previous frames with FFmpeg's select filter."""
    num_frames, height, width, channels = shape
    assert channels == 3
    frame_nbytes = height * width * channels
    frames = np.empty(shape, dtype=np.uint8)
    select_expr = _ffmpeg_select_expr(frame_count, sample_rate)
    cmd = [
        'ffmpeg',
        '-v', 'error',
        '-i', video_path,
        '-vf', f'select={select_expr}',
        '-vsync', '0',
        '-frames:v', str(num_frames),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        'pipe:1',
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    assert proc.stdout is not None

    try:
        for pos in range(num_frames):
            raw = proc.stdout.read(frame_nbytes)
            if len(raw) != frame_nbytes:
                stderr = proc.stderr.read().decode('utf-8', errors='replace') \
                    if proc.stderr is not None else ''
                raise RuntimeError(
                    f"FFmpeg decoded {pos}/{num_frames} selected frames from "
                    f"{video_path}: {stderr}"
                )
            frames[pos] = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    stderr = proc.stderr.read().decode('utf-8', errors='replace') \
        if proc.stderr is not None else ''
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {video_path}: {stderr}")
    return frames


def _ffmpeg_select_expr(frame_count: int, sample_rate: int) -> str:
    """Build an FFmpeg select expression for sampled and previous frames."""
    last_idx = frame_count - 1
    prev_last_idx = max(0, last_idx - 1)
    # Escaped commas are required because FFmpeg filter expressions use commas
    # as argument separators.
    sampled = f'eq(mod(n\\,{sample_rate})\\,0)'
    prev_sampled = f'eq(mod(n\\,{sample_rate})\\,{sample_rate - 1})'
    first_prev = 'eq(n\\,1)'
    last_terms = f'eq(n\\,{last_idx})+eq(n\\,{prev_last_idx})'
    in_prefix = f'lt(n\\,{frame_count})'
    return f"'{in_prefix}*({sampled}+{prev_sampled}+{first_prev}+{last_terms})'"
