#!/usr/bin/env python3
"""Segment TT smash TRC files around racket-ball contact events."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from trc_to_gmr import load_trc


HERE = Path(__file__).resolve().parent
BALL_MARKERS = ("ball1", "ball2", "ball3", "ball4")
RACKET_MARKER_SETS = (
    ("paddle1", "paddle2", "paddle3", "paddle4"),
    ("paddle_body1", "paddle_body2", "paddle_body3", "paddle_body4"),
)


def marker_center(markers: dict[str, np.ndarray], marker_names: tuple[str, ...]) -> np.ndarray:
    missing = [marker_name for marker_name in marker_names if marker_name not in markers]
    if missing:
        raise ValueError(f"Missing markers for center calculation: {missing}")
    return np.mean(np.stack([markers[marker_name] for marker_name in marker_names]), axis=0)


def racket_marker_names(markers: dict[str, np.ndarray]) -> tuple[str, ...]:
    for marker_names in RACKET_MARKER_SETS:
        if all(marker_name in markers for marker_name in marker_names):
            return marker_names
    expected = " or ".join(str(marker_names) for marker_names in RACKET_MARKER_SETS)
    raise ValueError(f"Could not find racket markers. Expected one of: {expected}")


def contact_events(distance: np.ndarray, threshold: float) -> list[int]:
    contact_frames = np.flatnonzero(distance < threshold)
    if len(contact_frames) == 0:
        return []

    events = []
    window_start = 0
    for index in range(1, len(contact_frames)):
        if contact_frames[index] != contact_frames[index - 1] + 1:
            window = contact_frames[window_start:index]
            events.append(int(window[np.argmin(distance[window])]))
            window_start = index

    window = contact_frames[window_start:]
    events.append(int(window[np.argmin(distance[window])]))
    return events


def read_trc_rows(path: Path) -> tuple[list[list[str]], int, list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "Frame#"),
        None,
    )
    if header_index is None:
        raise ValueError(f"{path} does not contain a TRC Frame# header")

    data_rows = [row for row in rows[header_index + 2 :] if row and row[0].strip()]
    return rows, header_index, data_rows


def trc_marker_names(rows: list[list[str]], header_index: int) -> list[str]:
    marker_names = []
    for column in range(2, len(rows[header_index]), 3):
        raw_name = rows[header_index][column].strip()
        if raw_name:
            marker_names.append(raw_name.split(":")[-1])
    return marker_names


def update_metadata(rows: list[list[str]], header_index: int, frame_count: int) -> None:
    metadata_names = rows[header_index - 2]
    metadata_values = rows[header_index - 1]
    metadata = {name: index for index, name in enumerate(metadata_names)}
    for key in ("NumFrames", "OrigNumFrames"):
        if key in metadata:
            metadata_values[metadata[key]] = str(frame_count)
    if "OrigDataStartFrame" in metadata:
        metadata_values[metadata["OrigDataStartFrame"]] = "1"


def format_time(seconds: float) -> str:
    return f"{seconds:.8f}".rstrip("0").rstrip(".")


def format_position(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def marker_to_trc_xyz(position: np.ndarray) -> np.ndarray:
    return np.array([position[0], position[2], -position[1]]) * 1000.0


def write_segment(
    output_path: Path,
    rows: list[list[str]],
    header_index: int,
    markers: dict[str, np.ndarray],
    marker_names: list[str],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> None:
    segment_rows = [row.copy() for row in rows[: header_index + 2]]
    selected_rows = []

    for frame_index, source_frame in enumerate(range(start_frame, end_frame), start=1):
        row = [str(frame_index), format_time((frame_index - 1) / fps)]
        for marker_name in marker_names:
            row.extend(format_position(value) for value in marker_to_trc_xyz(markers[marker_name][source_frame]))
        selected_rows.append(row)

    update_metadata(segment_rows, header_index, len(selected_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(segment_rows)
        writer.writerows(selected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trc_path", type=Path)
    parser.add_argument("--output-root", type=Path, default=HERE / "segments_trc")
    parser.add_argument("--threshold", type=float, default=0.1, help="Contact distance threshold in meters")
    parser.add_argument("--pre-seconds", type=float, default=0.54)
    parser.add_argument("--post-seconds", type=float, default=0.54)
    parser.add_argument("--allow-partial", action="store_true", help="Save edge clips even if full context is unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fps, markers = load_trc(args.trc_path)
    racket_center = marker_center(markers, racket_marker_names(markers))
    ball_center = marker_center(markers, BALL_MARKERS)
    distance = np.linalg.norm(racket_center - ball_center, axis=1)
    events = contact_events(distance, args.threshold)

    rows, header_index, data_rows = read_trc_rows(args.trc_path)
    marker_names = trc_marker_names(rows, header_index)
    if len(data_rows) != len(distance):
        raise ValueError(f"TRC row count mismatch: {len(data_rows)} rows, {len(distance)} loaded frames")

    pre_frames = int(round(args.pre_seconds * fps))
    post_frames = int(round(args.post_seconds * fps))
    output_dir = args.output_root / args.trc_path.stem
    metadata = {
        "source_trc": str(args.trc_path),
        "fps": fps,
        "threshold_m": args.threshold,
        "pre_seconds": args.pre_seconds,
        "post_seconds": args.post_seconds,
        "segments": [],
    }

    saved_count = 0
    skipped_count = 0
    for event_index, hit_frame in enumerate(events):
        start_frame = hit_frame - pre_frames
        end_frame = hit_frame + post_frames + 1
        if args.allow_partial:
            start_frame = max(0, start_frame)
            end_frame = min(len(data_rows), end_frame)
        elif start_frame < 0 or end_frame > len(data_rows):
            skipped_count += 1
            continue

        output_path = output_dir / f"{args.trc_path.stem}_seg_{saved_count:03d}_hit_{hit_frame + 1:06d}.trc"
        write_segment(output_path, rows, header_index, markers, marker_names, start_frame, end_frame, fps)
        metadata["segments"].append({
            "segment_trc": str(output_path),
            "hit_frame_source_1based": hit_frame + 1,
            "start_frame_source_1based": start_frame + 1,
            "end_frame_source_1based": end_frame,
            "frame_count": end_frame - start_frame,
            "min_distance_m": float(distance[hit_frame]),
        })
        saved_count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "segments.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    print(f"Found {len(events)} contact events below {args.threshold:g} m")
    print(f"Saved {saved_count} segments to {output_dir}")
    if skipped_count:
        print(f"Skipped {skipped_count} edge events without full context; use --allow-partial to keep them")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
