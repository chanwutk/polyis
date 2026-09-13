"""NamedTuple message types for queues between pipeline stages.

The execution runner treats one video as the pipeline unit.  Large arrays stay
inside the main process unless they cross the prune process boundary; prune
messages intentionally carry classifications only, not decoded frames.
Sentinel ``None`` on any queue signals shutdown to that stage.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DecodedVideo(NamedTuple):
    """Decoder -> Classify, all frames needed for one video."""
    video: str
    frames_rgb: np.ndarray
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]
    buffer_frame_indices: list[int]
    batch_positions: list[int]
    prev_positions: list[int]


class ClassifiedVideo(NamedTuple):
    """Classify/Prune -> Compress, with decoded frames for in-process stages."""
    video: str
    classifications: np.ndarray
    frames_rgb: np.ndarray
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]
    buffer_frame_indices: list[int]


class VideoClassifications(NamedTuple):
    """Classify -> Prune and Prune -> main-process joiner."""
    video: str
    classifications: np.ndarray
    width: int
    height: int
    frame_count: int
    sampled_indices: list[int]


class CollageReady(NamedTuple):
    """Compress -> Detect, one packed collage ready for detection."""
    video: str
    collage_idx: int
    total_collages: int
    is_last: bool
    canvas_rgb: np.ndarray
    index_map: np.ndarray
    offset_lookup: list
    num_frames: int
    tile_size: int


class VideoDetections(NamedTuple):
    """Detect -> Track, accumulated per-frame detections for one video."""
    video: str
    frame_detections: dict[int, list[list[float]]]
    num_frames: int


class TrackingResult(NamedTuple):
    """Track -> Main, final tracker output for one video."""
    video: str
    frame_tracks: dict[int, list[list[float]]]
