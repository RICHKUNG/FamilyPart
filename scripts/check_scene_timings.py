#!/usr/bin/env python3
"""Aggregate stage timings for experiment scenes and emit a markdown report with output statistics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any


# Update this list to change the default experiment directories processed when no arguments are provided.
DEFAULT_BASES = ["./outputs/experiments/v2_135_ssam2_filter2k_fill10k_prop30_iou06_ds03",
                "./outputs/experiments/v2_135_ssam2_filter2k_fill10k_prop50_iou06_ds03",
                "./outputs/experiments/v2_135_ssam2_filter500_fill3k_prop30_iou06_ds03",
                "./outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop10_iou06_ds03",
                "./outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop30_iou05_ds03",
                "./outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop30_iou06_ds03",
                "./outputs/experiments/v2_246_ssam2_filter2k_fill10k_prop50_iou06_ds03",
                "./outputs/experiments/v2_246_ssam2_filter500_fill3k_prop30_iou06_ds03",
                "./outputs/experiments/v2_246_ssam2_filter500_fill3k_prop30_iou07_ds03"]


@dataclass
class SceneStats:
    """Container for scene output statistics."""

    objects_per_level: Dict[str, int] = field(default_factory=dict)
    total_objects: int = 0
    total_families: int = 0  # Objects with children
    orphan_objects: int = 0  # Objects without parent or children
    max_depth: int = 0
    frames_per_level: Dict[str, int] = field(default_factory=dict)


@dataclass
class SceneTiming:
    """Container for per-scene timing and statistics information."""

    name: str
    summary_found: bool
    total: str
    stage_durations: Dict[str, str]
    stats: Optional[SceneStats] = None


def format_seconds(seconds: Optional[float]) -> str:
    """Return a compact H:MM:SS or M:SS string for a duration in seconds."""
    if seconds is None:
        return "N/A"
    try:
        total_seconds = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "N/A"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_scene_stats(scene_dir: Path) -> Optional[SceneStats]:
    """Load output statistics from relations.json and level indices."""
    relations_path = scene_dir / "relations.json"
    if not relations_path.is_file():
        return None

    try:
        with relations_path.open("r", encoding="utf-8") as fh:
            relations = json.load(fh)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to read {relations_path}: {exc}", file=sys.stderr)
        return None

    stats = SceneStats()
    hierarchy = relations.get("hierarchy", {})

    # Count objects per level
    for level_str, objects in hierarchy.items():
        level = f"L{level_str}"
        stats.objects_per_level[level] = len(objects)
        stats.total_objects += len(objects)

        # Count families and orphans
        for obj_id, obj_data in objects.items():
            has_children = len(obj_data.get("children", [])) > 0
            has_parent = obj_data.get("parent") is not None

            if has_children:
                stats.total_families += 1
            if not has_children and not has_parent:
                stats.orphan_objects += 1

    # Calculate max depth from paths
    paths = relations.get("paths", {})
    if paths:
        stats.max_depth = max((len(path) for path in paths.values()), default=0)

    # Load frame counts from level indices
    for level_str in hierarchy.keys():
        index_path = scene_dir / f"level_{level_str}" / "index.json"
        if index_path.is_file():
            try:
                with index_path.open("r", encoding="utf-8") as fh:
                    index_data = json.load(fh)
                    # Count unique frames from all objects
                    frames_set = set()
                    for obj_data in index_data.get("objects", {}).values():
                        frames_set.update(obj_data.get("frames", []))
                    stats.frames_per_level[f"L{level_str}"] = len(frames_set)
            except Exception:  # pylint: disable=broad-except
                pass

    return stats


def load_scene_timing(scene_dir: Path) -> SceneTiming:
    """Load timing information and output statistics from scene directory."""
    summary_path = scene_dir / "summary.json"
    if not summary_path.is_file():
        return SceneTiming(scene_dir.name, False, "N/A", {}, None)

    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to read {summary_path}: {exc}", file=sys.stderr)
        return SceneTiming(scene_dir.name, True, "N/A", {}, None)

    timings = summary.get("timings") or {}
    total = timings.get("total_duration_text") or format_seconds(
        timings.get("total_duration_sec")
    )

    stage_durations: Dict[str, str] = {}
    for stage in timings.get("stages") or []:
        stage_name = stage.get("stage") or "unknown"
        duration = stage.get("duration_text") or format_seconds(
            stage.get("duration_sec")
        )
        stage_durations[stage_name] = duration or "N/A"

    # Load output statistics
    stats = load_scene_stats(scene_dir)

    return SceneTiming(scene_dir.name, True, total or "N/A", stage_durations, stats)


def collect_stage_order(scenes: Iterable[SceneTiming]) -> List[str]:
    """Derive a stable stage order from the scene data."""
    ordered: List[str] = []
    for scene in scenes:
        for stage in scene.stage_durations:
            if stage not in ordered:
                ordered.append(stage)
    return ordered


def render_markdown_table(stage_order: List[str], scenes: Iterable[SceneTiming]) -> str:
    """Produce a markdown table summarising timings."""
    headers = ["Scene", "Total"] + stage_order
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for scene in scenes:
        row = [scene.name, scene.total if scene.summary_found else "N/A"]
        for stage in stage_order:
            value = scene.stage_durations.get(stage, "N/A")
            row.append(value if scene.summary_found else "N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def render_stats_table(scenes: Iterable[SceneTiming]) -> str:
    """Produce a markdown table summarising output statistics."""
    # Collect all levels across scenes
    all_levels = set()
    for scene in scenes:
        if scene.stats:
            all_levels.update(scene.stats.objects_per_level.keys())
    sorted_levels = sorted(all_levels)

    # Build headers
    obj_headers = [f"{lvl}_Objs" for lvl in sorted_levels]
    frame_headers = [f"{lvl}_Frames" for lvl in sorted_levels]
    headers = ["Scene", "Total_Objs", "Families", "Orphans", "Max_Depth"] + obj_headers + frame_headers

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for scene in scenes:
        if not scene.stats or scene.stats.total_objects == 0:
            row = [scene.name] + ["N/A"] * (len(headers) - 1)
        else:
            s = scene.stats
            row = [
                scene.name,
                str(s.total_objects),
                str(s.total_families),
                str(s.orphan_objects),
                str(s.max_depth),
            ]
            # Add object counts per level
            for level in sorted_levels:
                row.append(str(s.objects_per_level.get(level, 0)))
            # Add frame counts per level
            for level in sorted_levels:
                row.append(str(s.frames_per_level.get(level, 0)))

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def build_report(base_path: Path) -> str:
    """Generate the markdown report for every scene under base_path."""
    scene_dirs = sorted(
        (path for path in base_path.iterdir() if path.is_dir()),
        key=lambda p: p.name,
    )

    scenes = [load_scene_timing(scene_dir) for scene_dir in scene_dirs]
    stage_order = collect_stage_order(scenes)

    # Build report with both timing and statistics
    report_parts = [
        f"# Experiment Report: {base_path.name}\n",
        "## Stage Timings\n",
        render_markdown_table(stage_order, scenes),
        "\n## Output Statistics\n",
        render_stats_table(scenes),
    ]

    return "\n".join(report_parts)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown table summarising scene stage timings."
    )
    parser.add_argument(
        "base_dirs",
        nargs="*",
        default=list(DEFAULT_BASES),
        help=(
            "One or more experiment directories containing scene folders "
            f"(default: {', '.join(DEFAULT_BASES)})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    exit_code = 0

    for base_dir in args.base_dirs:
        base_path = Path(base_dir).resolve()
        if not base_path.is_dir():
            print(f"[ERROR] Base directory not found: {base_path}", file=sys.stderr)
            exit_code = 1
            continue

        report = build_report(base_path)
        output_path = base_path / "check.md"
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote {output_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
