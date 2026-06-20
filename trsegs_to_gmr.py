#!/usr/bin/env python3
"""Batch-retarget segmented TRC clips to GMR robot-motion pickle files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from trc_to_gmr import convert_trc_to_gmr


SEGMENT_RE = re.compile(r"(?:^|_)seg_(\d+)(?:_|\.|$)")


def segment_id(path: Path, fallback_index: int) -> str:
    match = SEGMENT_RE.search(path.stem)
    if match:
        return f"seg_{int(match.group(1)):03d}"
    return f"seg_{fallback_index:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment_dir", type=Path, help="Folder containing segmented .trc clips")
    parser.add_argument("--output-dir", type=Path, default=Path("retargeted"), help="Folder for generated .pkl files")
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--human-height", type=float, help="Measured subject height in meters")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate .pkl files that already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trc_paths = sorted(args.segment_dir.glob("*.trc"))
    if not trc_paths:
        raise ValueError(f"No .trc clips found in {args.segment_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "segment_dir": str(args.segment_dir),
        "output_dir": str(args.output_dir),
        "robot": args.robot,
        "target_fps": args.target_fps,
        "segments": [],
    }

    for index, trc_path in enumerate(trc_paths):
        seg_name = segment_id(trc_path, index)
        output_path = args.output_dir / f"{seg_name}.pkl"
        if output_path.exists() and not args.overwrite:
            print(f"Skip existing {output_path}")
            summary["segments"].append({
                "segment": seg_name,
                "trc": str(trc_path),
                "pkl": str(output_path),
                "status": "skipped_existing",
            })
            continue

        print(f"Converting {trc_path.name} -> {output_path.name}")
        _, height, source_frames, selected_frames = convert_trc_to_gmr(
            trc_path=trc_path,
            output_path=output_path,
            robot=args.robot,
            target_fps=args.target_fps,
            human_height=args.human_height,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
        )
        summary["segments"].append({
            "segment": seg_name,
            "trc": str(trc_path),
            "pkl": str(output_path),
            "status": "converted",
            "estimated_or_given_height_m": height,
            "source_frames": source_frames,
            "retargeted_frames": selected_frames,
        })

    summary_path = args.output_dir / "segments_gmr.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
