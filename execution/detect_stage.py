"""Detect thread: batched object detection + coordinate uncompression."""

from __future__ import annotations

import queue

import numpy as np
import torch

import polyis.dtypes
import polyis.models.detector
from polyis.models.detector import DetectorMode
from scripts.p050_exec_uncompress import unpack_detections

from execution.config import PipelineConfig
from execution.messages import (
    CollageReady,
    VideoDetections,
)


def detect_stage(
    *,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
):
    """Main loop of the detect thread."""
    torch.cuda.set_device(config.detect_gpu)
    detector = polyis.models.detector.get_detector(
        config.dataset, config.detect_gpu, config.detect_batch_size, num_images=100,
    )

    video_detections: dict[str, dict[int, list[list[float]]]] = {}
    video_received: dict[str, int] = {}
    video_expected: dict[str, int] = {}
    video_num_frames: dict[str, int] = {}
    pending_msgs: list[CollageReady] = []
    pending_rgbs: list[np.ndarray] = []

    with torch.no_grad(), torch.inference_mode():
        while True:
            msg = in_queue.get()
            if msg is None:
                if pending_msgs:
                    _flush_batch(
                        pending_msgs, pending_rgbs, detector,
                        video_detections, video_received,
                        video_expected, video_num_frames, out_queue,
                    )
                out_queue.put(None)
                return

            assert isinstance(msg, CollageReady)
            if msg.video not in video_detections:
                video_detections[msg.video] = {i: [] for i in range(msg.num_frames)}
                video_received[msg.video] = 0
                video_expected[msg.video] = msg.total_collages
                video_num_frames[msg.video] = msg.num_frames

            if msg.total_collages == 0:
                out_queue.put(VideoDetections(
                    video=msg.video,
                    frame_detections=video_detections.pop(msg.video),
                    num_frames=video_num_frames.pop(msg.video),
                ))
                video_received.pop(msg.video, None)
                video_expected.pop(msg.video, None)
                continue

            assert polyis.dtypes.is_np_image(msg.canvas_rgb)
            pending_msgs.append(msg)
            pending_rgbs.append(msg.canvas_rgb)

            this_video_pending = sum(1 for m in pending_msgs if m.video == msg.video)
            will_complete_video = (
                video_received[msg.video] + this_video_pending
                >= video_expected[msg.video]
            )
            if len(pending_msgs) >= config.detect_batch_size or will_complete_video:
                _flush_batch(
                    pending_msgs, pending_rgbs, detector,
                    video_detections, video_received,
                    video_expected, video_num_frames, out_queue,
                )
                pending_msgs = []
                pending_rgbs = []


def _flush_batch(
    msgs: list[CollageReady],
    rgbs: list[np.ndarray],
    detector,
    video_detections: dict[str, dict[int, list[list[float]]]],
    video_received: dict[str, int],
    video_expected: dict[str, int],
    video_num_frames: dict[str, int],
    out_queue: queue.Queue,
):
    """Run detector on the pending batch and emit completed videos."""
    batch_output = polyis.models.detector.detect_batch(
        rgbs, detector, mode=DetectorMode.RGB,
    )

    for i, msg in enumerate(msgs):
        detections = batch_output[i].tolist()
        frame_dets, _not_in_tile, _center_not = unpack_detections(
            detections, msg.index_map, msg.offset_lookup, msg.tile_size,
        )
        for frame_idx, bboxes in frame_dets.items():
            video_detections[msg.video][frame_idx].extend(bboxes)
        video_received[msg.video] += 1

        if video_received[msg.video] >= video_expected[msg.video]:
            out_queue.put(VideoDetections(
                video=msg.video,
                frame_detections=video_detections.pop(msg.video),
                num_frames=video_num_frames.pop(msg.video),
            ))
            video_received.pop(msg.video, None)
            video_expected.pop(msg.video, None)
