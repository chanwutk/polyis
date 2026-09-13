"""Decoder tests for exact sampled-frame extraction."""

from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from execution.decode_stage import (
    _compute_frame_plan,
    _decode_needed_with_ffmpeg,
    _decode_needed_with_opencv,
)


def _write_lossless_video(path, frames_rgb: np.ndarray) -> None:
    """Write RGB frames to a lossless FFV1 video with ffmpeg."""
    if shutil.which('ffmpeg') is None:
        pytest.skip('ffmpeg is not available')
    height, width = frames_rgb.shape[1:3]
    cmd = [
        'ffmpeg',
        '-y',
        '-v', 'error',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', '10',
        '-i', 'pipe:0',
        '-c:v', 'ffv1',
        str(path),
    ]
    proc = subprocess.run(cmd, input=frames_rgb.tobytes(), capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode('utf-8', errors='replace')


def _synthetic_frames(num_frames: int, height: int = 12, width: int = 16) -> np.ndarray:
    """Build frames where each frame has a unique constant RGB color."""
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for idx in range(num_frames):
        frames[idx, :, :, 0] = idx * 13
        frames[idx, :, :, 1] = 255 - idx * 7
        frames[idx, :, :, 2] = idx * 3
    return frames


def test_compute_frame_plan_includes_sampled_previous_and_last():
    """Frame plan keeps sampled frames, previous frames, and the final frame."""
    sampled, buffer_indices, batch_positions, prev_positions = _compute_frame_plan(
        frame_count=10,
        sample_rate=4,
    )
    assert sampled == [0, 4, 8, 9]
    assert buffer_indices == [0, 1, 3, 4, 7, 8, 9]
    assert [buffer_indices[p] for p in batch_positions] == sampled
    assert [buffer_indices[p] for p in prev_positions] == [1, 3, 7, 8]


def test_ffmpeg_selected_decode_returns_exact_needed_rgb_frames(tmp_path):
    """FFmpeg sparse decode emits exactly the selected RGB frames in order."""
    frames_rgb = _synthetic_frames(10)
    video_path = tmp_path / 'synthetic.mkv'
    _write_lossless_video(video_path, frames_rgb)

    _sampled, buffer_indices, _batch_positions, _prev_positions = _compute_frame_plan(
        frame_count=10,
        sample_rate=4,
    )
    decoded = _decode_needed_with_ffmpeg(
        video_path=str(video_path),
        shape=(len(buffer_indices), frames_rgb.shape[1], frames_rgb.shape[2], 3),
        frame_count=10,
        sample_rate=4,
    )

    np.testing.assert_array_equal(decoded, frames_rgb[buffer_indices])


def test_opencv_full_decode_returns_exact_needed_rgb_frames(tmp_path):
    """OpenCV full decode returns exact RGB frames for the lossless fixture."""
    frames_rgb = _synthetic_frames(6)
    video_path = tmp_path / 'synthetic.mkv'
    _write_lossless_video(video_path, frames_rgb)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            pytest.skip('OpenCV cannot read the lossless fixture in this build')
    finally:
        cap.release()

    _sampled, buffer_indices, _batch_positions, _prev_positions = _compute_frame_plan(
        frame_count=6,
        sample_rate=1,
    )
    decoded = _decode_needed_with_opencv(
        video_path=str(video_path),
        shape=(len(buffer_indices), frames_rgb.shape[1], frames_rgb.shape[2], 3),
        frame_count=6,
        buffer_frame_indices=buffer_indices,
    )

    np.testing.assert_array_equal(decoded, frames_rgb[buffer_indices])
