#!/usr/bin/env python3
"""Retarget Vicon Plug-in Gait TRC marker data to a GMR robot-motion pickle."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent / "GMR"
REQUIRED_MARKERS = {
    "LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN",
    "LSHO", "LELB", "LWRA", "LWRB", "LFIN",
    "RSHO", "RELB", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LKNE", "LANK", "LTOE", "RKNE", "RANK", "RTOE",
}
RACKET_MARKERS = ("paddle_body1", "paddle_body2", "paddle_body3", "paddle_body4")


def _marker_name(raw_name: str) -> str:
    return raw_name.strip().split(":")[-1]


def _fill_missing(values: np.ndarray, marker_name: str) -> np.ndarray:
    result = values.copy()
    for axis in range(3):
        valid = np.isfinite(result[:, axis])
        if not valid.any():
            raise ValueError(f"Marker {marker_name} has no valid values for axis {axis}")
        if not valid.all():
            sample_ids = np.arange(len(result))
            result[:, axis] = np.interp(sample_ids, sample_ids[valid], result[valid, axis])
    return result


def load_trc(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """Load TRC markers in meters and convert Vicon Y-up coordinates to GMR Z-up."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0].strip() == "Frame#"),
        None,
    )
    if header_index is None:
        raise ValueError(f"{path} does not contain a TRC Frame# header")

    metadata = rows[header_index - 2]
    metadata_values = rows[header_index - 1]
    metadata_map = dict(zip(metadata, metadata_values))
    frame_rate = float(metadata_map["DataRate"])

    marker_names = []
    for column in range(2, len(rows[header_index]), 3):
        if rows[header_index][column].strip():
            marker_names.append(_marker_name(rows[header_index][column]))

    duplicates = sorted({name for name in marker_names if marker_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate marker names after prefix removal: {duplicates}")

    frames = []
    expected_columns = 2 + 3 * len(marker_names)
    for row in rows[header_index + 2 :]:
        if not row or not row[0].strip():
            continue
        padded = row + [""] * max(0, expected_columns - len(row))
        frames.append([
            float(value) if value.strip() else np.nan
            for value in padded[2:expected_columns]
        ])

    values = np.asarray(frames, dtype=np.float64).reshape(-1, len(marker_names), 3)
    # Vicon export: X=lateral, Y=up, Z=forward. GMR/MuJoCo: Z=up.
    # Negating the new Y axis preserves a right-handed coordinate system.
    values = values[:, :, [0, 2, 1]] * np.array([1.0, -1.0, 1.0]) / 1000.0
    markers = {
        marker_name: _fill_missing(values[:, index], marker_name)
        for index, marker_name in enumerate(marker_names)
    }

    missing = sorted(REQUIRED_MARKERS - markers.keys())
    if missing:
        raise ValueError(f"{path} is missing required body markers: {missing}")
    return frame_rate, markers


def _normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        vector = fallback
        norm = np.linalg.norm(vector)
    return vector / norm


def _frame_from_xz(x_hint: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    """Return a rotation matrix whose columns are orthonormal local x/y/z axes."""
    x_axis = _normalize(x_hint, np.array([1.0, 0.0, 0.0]))
    z_axis = z_hint - np.dot(z_hint, x_axis) * x_axis
    z_axis = _normalize(z_axis, np.array([0.0, 0.0, 1.0]))
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0]))
    z_axis = _normalize(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0]))
    return np.column_stack((x_axis, y_axis, z_axis))


def _quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized scalar-first quaternion."""
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = np.sqrt(1.0 + matrix[index, index] - matrix[next_index, next_index] - matrix[last_index, last_index]) * 2.0
        quat = np.zeros(4)
        quat[index + 1] = 0.25 * scale
        quat[0] = (matrix[last_index, next_index] - matrix[next_index, last_index]) / scale
        quat[next_index + 1] = (matrix[next_index, index] + matrix[index, next_index]) / scale
        quat[last_index + 1] = (matrix[last_index, index] + matrix[index, last_index]) / scale
    return quat / np.linalg.norm(quat)


def marker_frame(markers: dict[str, np.ndarray], index: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    point = lambda name: markers[name][index]
    midpoint = lambda left, right: 0.5 * (point(left) + point(right))

    pelvis = 0.25 * (point("LASI") + point("RASI") + point("LPSI") + point("RPSI"))
    shoulder_mid = midpoint("LSHO", "RSHO")
    torso_mid = 0.5 * (point("C7") + point("T10"))
    left_wrist = midpoint("LWRA", "LWRB")
    right_wrist = midpoint("RWRA", "RWRB")

    pelvis_matrix = _frame_from_xz(point("LASI") - point("RASI"), shoulder_mid - pelvis)
    torso_matrix = _frame_from_xz(point("LSHO") - point("RSHO"), point("C7") - point("T10"))
    up = pelvis_matrix[:, 2]
    forward = pelvis_matrix[:, 1]

    def limb(start: np.ndarray, end: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return _frame_from_xz(end - start, reference)

    transforms = {
        "Hips": (pelvis, pelvis_matrix),
        "Spine1": (torso_mid, torso_matrix),
        "LeftUpLeg": (point("LASI"), limb(point("LASI"), point("LKNE"), forward)),
        "LeftLeg": (point("LKNE"), limb(point("LKNE"), point("LANK"), forward)),
        "LeftToeBase": (point("LTOE"), limb(point("LANK"), point("LTOE"), up)),
        "RightUpLeg": (point("RASI"), limb(point("RASI"), point("RKNE"), forward)),
        "RightLeg": (point("RKNE"), limb(point("RKNE"), point("RANK"), forward)),
        "RightToeBase": (point("RTOE"), limb(point("RANK"), point("RTOE"), up)),
        "LeftArm": (point("LSHO"), limb(point("LSHO"), point("LELB"), up)),
        "LeftForeArm": (point("LELB"), limb(point("LELB"), left_wrist, up)),
        "LeftHand": (left_wrist, limb(left_wrist, point("LFIN"), up)),
        "RightArm": (point("RSHO"), limb(point("RSHO"), point("RELB"), up)),
        "RightForeArm": (point("RELB"), limb(point("RELB"), right_wrist, up)),
        "RightHand": (right_wrist, limb(right_wrist, point("RFIN"), up)),
    }

    if all(marker_name in markers for marker_name in RACKET_MARKERS):
        racket_points = {marker_name: point(marker_name) for marker_name in RACKET_MARKERS}
        transforms["Racket"] = racket_transform(racket_points)
    return {name: (position, _quat_wxyz(matrix)) for name, (position, matrix) in transforms.items()}


def racket_transform(points: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    center = np.mean(np.stack([points[marker_name] for marker_name in RACKET_MARKERS]), axis=0)
    # The four paddle markers form a rigid body. This convention makes local X run
    # across the broad marker pair and local Z follow the marker-plane normal.
    x_hint = 0.5 * (points["paddle_body3"] + points["paddle_body4"]) - 0.5 * (
        points["paddle_body1"] + points["paddle_body2"]
    )
    y_hint = 0.5 * (points["paddle_body1"] + points["paddle_body3"]) - 0.5 * (
        points["paddle_body2"] + points["paddle_body4"]
    )
    z_hint = np.cross(x_hint, y_hint)
    return center, _frame_from_xz(x_hint, z_hint)


def estimate_height(markers: dict[str, np.ndarray]) -> float:
    head = 0.25 * (markers["LFHD"] + markers["RFHD"] + markers["LBHD"] + markers["RBHD"])
    feet = np.minimum(markers["LTOE"][:, 2], markers["RTOE"][:, 2])
    # Head markers sit below the top of the skull. This correction is only a default;
    # use --human-height when a measured subject height is available.
    return float(np.median(head[:, 2] - feet) + 0.1)


def scaled_ground_height(retargeter, human_frames: list[dict[str, tuple[np.ndarray, np.ndarray]]]) -> float:
    lowest = np.inf
    for human_frame in human_frames:
        scaled = retargeter.scale_human_data(
            retargeter.to_numpy(human_frame),
            retargeter.human_root_name,
            retargeter.human_scale_table,
        )
        offset = retargeter.offset_human_data(scaled, retargeter.pos_offsets1, retargeter.rot_offsets1)
        lowest = min(lowest, *(position[2] for position, _ in offset.values()))
    return float(lowest)


def selected_frame_ids(source_fps: float, frame_count: int, target_fps: float, start_frame: int, max_frames: int | None) -> np.ndarray:
    frame_ids = np.arange(start_frame, frame_count)
    frame_ids = frame_ids[np.floor(frame_ids * target_fps / source_fps).astype(int) !=
                          np.floor((frame_ids - 1) * target_fps / source_fps).astype(int)]
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    if not len(frame_ids):
        raise ValueError("No frames selected; check --start-frame and --max-frames")
    return frame_ids


def retarget_trc_to_motion_data(
    trc_path: Path,
    robot: str = "unitree_g1",
    target_fps: float = 30.0,
    human_height: float | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    retargeter=None,
) -> tuple[dict, float, int, int]:
    source_fps, markers = load_trc(trc_path)
    height = human_height if human_height is not None else estimate_height(markers)
    frame_count = len(next(iter(markers.values())))
    frame_ids = selected_frame_ids(source_fps, frame_count, target_fps, start_frame, max_frames)

    if retargeter is None:
        sys.path.insert(0, str(GMR_ROOT))
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        retargeter = GMR(src_human="trc_vicon", tgt_robot=robot, actual_human_height=height)

    human_frames = [marker_frame(markers, int(index)) for index in frame_ids]
    retargeter.set_ground_offset(scaled_ground_height(retargeter, human_frames))
    qpos = np.asarray([retargeter.retarget(frame) for frame in human_frames])

    motion_data = {
        "fps": min(source_fps, target_fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # GMR pickle convention: xyzw
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
        "source_trc": str(trc_path),
    }
    return motion_data, height, frame_count, len(frame_ids)


def save_motion_data(motion_data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(motion_data, handle)


def convert_trc_to_gmr(
    trc_path: Path,
    output_path: Path,
    robot: str = "unitree_g1",
    target_fps: float = 30.0,
    human_height: float | None = None,
    start_frame: int = 0,
    max_frames: int | None = None,
    retargeter=None,
) -> tuple[dict, float, int, int]:
    motion_data, height, frame_count, selected_count = retarget_trc_to_motion_data(
        trc_path=trc_path,
        robot=robot,
        target_fps=target_fps,
        human_height=human_height,
        start_frame=start_frame,
        max_frames=max_frames,
        retargeter=retargeter,
    )
    save_motion_data(motion_data, output_path)
    return motion_data, height, frame_count, selected_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trc_path", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--human-height", type=float, help="Measured subject height in meters")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--inspect-only", action="store_true", help="Validate the TRC without loading GMR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_fps, markers = load_trc(args.trc_path)
    height = args.human_height if args.human_height is not None else estimate_height(markers)
    frame_ids = selected_frame_ids(
        source_fps,
        len(next(iter(markers.values()))),
        args.target_fps,
        args.start_frame,
        args.max_frames,
    )

    print(f"Loaded {args.trc_path}: {len(next(iter(markers.values())))} frames at {source_fps:g} Hz")
    print(f"Estimated subject height: {height:.3f} m; selected {len(frame_ids)} frames at <= {args.target_fps:g} Hz")
    if args.inspect_only:
        return

    motion_data, _, _, selected_count = convert_trc_to_gmr(
        trc_path=args.trc_path,
        output_path=args.output,
        robot=args.robot,
        target_fps=args.target_fps,
        human_height=height,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    print(f"Saved {selected_count} retargeted frames to {args.output}")


if __name__ == "__main__":
    main()
