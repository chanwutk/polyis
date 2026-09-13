"""Result persistence: tracking output + aggregated pipeline runtime."""

from __future__ import annotations

import json
import os

from polyis.io import cache
from polyis.utilities import build_param_str, save_tracking_results

from execution.config import PipelineConfig
from execution.messages import TrackingResult


def _output_param_str(config: PipelineConfig) -> str:
    """Build the param_str used by cache.exec to locate output paths.

    Mirrors the convention used by scripts/p060 so downstream evaluation
    scripts read both paths transparently.
    """
    return build_param_str(
        classifier=config.classifier,
        tilesize=config.tile_size,
        sample_rate=config.sample_rate,
        tilepadding=config.tilepadding,
        canvas_scale=config.canvas_scale,
        tracker=config.tracker,
        tracking_accuracy_threshold=config.tracking_accuracy_threshold,
        relevance_threshold=config.relevance_threshold,
    )


def save_tracking_result(result: TrackingResult, config: PipelineConfig) -> None:
    """Write one video's tracking.jsonl to the same path as scripts/p060."""
    param_str = _output_param_str(config)
    output_path = cache.exec(
        config.dataset, 'ucomp-tracks', result.video,
        param_str, 'tracking.jsonl',
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_tracking_results(result.frame_tracks, output_path)


def save_pipeline_runtime(
    *,
    config: PipelineConfig,
    elapsed_ms: float,
    num_videos: int,
    per_video_complete_ts: dict[str, float],
) -> str:
    """Write the aggregated pipeline runtime summary; return the file path."""
    param_str = _output_param_str(config)
    summary = {
        'config': {**{k: getattr(config, k) for k in config.__dataclass_fields__.keys()}},
        'param_str': param_str,
        'elapsed_ms': elapsed_ms,
        'num_videos': num_videos,
        'per_video_complete_ts': per_video_complete_ts,
    }

    summary_dir = cache.root(config.dataset, 'pipeline-runtime', param_str)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(str(summary_dir), 'runtime.jsonl')
    with open(summary_path, 'a') as f:
        f.write(json.dumps(summary, default=str) + '\n')
    return summary_path
