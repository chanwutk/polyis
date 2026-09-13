"""Compress worker thread pool: group_tiles + pack + render RGB canvases."""

from __future__ import annotations

import queue

import numpy as np

from polyis import dtypes
from polyis.pack.group_tiles import group_tiles
from polyis.pack.pack import pack
from polyis.pack.render import (
    precompute_grid_boundaries,
    render_collage_cpu,
)
from polyis.utilities import TILEPADDING_MAPS

from execution.config import PipelineConfig
from execution.messages import ClassifiedVideo, CollageReady


_PACK_MODE_BEST_FIT = 2


def compress_worker(
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    config: PipelineConfig,
):
    """Entry point for one compress worker thread."""
    while True:
        msg = in_queue.get()
        if msg is None:
            out_queue.put(None)
            return
        assert isinstance(msg, ClassifiedVideo)
        _compress_one_video(msg=msg, config=config, out_queue=out_queue)


def _compress_one_video(
    msg: ClassifiedVideo,
    config: PipelineConfig,
    out_queue: queue.Queue,
) -> None:
    """Process one video: group tiles, pack, render collages, emit messages."""
    src_grid_height = msg.height // config.tile_size
    src_grid_width = msg.width // config.tile_size
    dst_grid_height = max(1, int(round(src_grid_height * config.canvas_scale)))
    dst_grid_width = max(1, int(round(src_grid_width * config.canvas_scale)))
    canvas_height = dst_grid_height * config.tile_size
    canvas_width = dst_grid_width * config.tile_size

    array_idx_to_frame_idx = {i: idx for i, idx in enumerate(msg.sampled_indices)}
    polyominoes_stacks = np.empty(len(msg.sampled_indices), dtype=np.uint64)
    cutoff = config.relevance_threshold * 255

    for array_idx, grid in enumerate(msg.classifications):
        bitmap = (grid > cutoff).astype(np.uint8)
        assert dtypes.is_bitmap(bitmap), bitmap.shape
        polyominoes_stacks[array_idx] = group_tiles(
            bitmap, TILEPADDING_MAPS[config.tilepadding],
        )

    raw_collages = pack(polyominoes_stacks, dst_grid_height, dst_grid_width, _PACK_MODE_BEST_FIT)
    collages: list[list[tuple]] = []
    for raw in raw_collages:
        collages.append([
            (pos.oy, pos.ox, pos.py, pos.px,
             array_idx_to_frame_idx[pos.frame], pos.shape)
            for pos in raw
        ])
    total_collages = len(collages)

    idx_to_buf_pos = {
        abs_idx: pos for pos, abs_idx in enumerate(msg.buffer_frame_indices)
    }

    if total_collages == 0:
        out_queue.put(CollageReady(
            video=msg.video,
            collage_idx=0,
            total_collages=0,
            is_last=True,
            canvas_rgb=np.empty((0, 0, 3), dtype=np.uint8),
            index_map=np.empty((0, 0), dtype=np.uint16),
            offset_lookup=[],
            num_frames=msg.frame_count,
            tile_size=config.tile_size,
        ))
        return

    src_y_starts, src_y_ends, src_x_starts, src_x_ends = \
        precompute_grid_boundaries(src_grid_height, src_grid_width, config.tile_size)
    dst_y_starts, dst_y_ends, dst_x_starts, dst_x_ends = \
        precompute_grid_boundaries(dst_grid_height, dst_grid_width, config.tile_size)

    def fetch_frame(abs_frame_idx: int) -> np.ndarray:
        return msg.frames_rgb[idx_to_buf_pos[abs_frame_idx]]

    canvas_shape = (canvas_height, canvas_width, 3)
    for collage_idx, collage in enumerate(collages):
        canvas_rgb = np.zeros(canvas_shape, dtype=np.uint8)
        index_map = np.zeros((dst_grid_height, dst_grid_width), dtype=np.uint16)
        offset_lookup: list = []

        render_collage_cpu(
            canvas=canvas_rgb,
            index_map=index_map,
            offset_lookup=offset_lookup,
            collage=collage,
            fetch_frame=fetch_frame,
            src_y_starts=src_y_starts, src_y_ends=src_y_ends,
            src_x_starts=src_x_starts, src_x_ends=src_x_ends,
            dst_y_starts=dst_y_starts, dst_y_ends=dst_y_ends,
            dst_x_starts=dst_x_starts, dst_x_ends=dst_x_ends,
        )

        out_queue.put(CollageReady(
            video=msg.video,
            collage_idx=collage_idx,
            total_collages=total_collages,
            is_last=(collage_idx == total_collages - 1),
            canvas_rgb=canvas_rgb,
            index_map=index_map,
            offset_lookup=offset_lookup,
            num_frames=msg.frame_count,
            tile_size=config.tile_size,
        ))
