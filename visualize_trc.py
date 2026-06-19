#!/usr/bin/env python3
"""Replay Vicon TRC markers and inferred GMR body frames in a MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from trc_to_gmr import load_trc, marker_frame


SKELETON_LINKS = [
    ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD"),
    ("LFHD", "C7"), ("RFHD", "C7"),
    ("C7", "LSHO"), ("C7", "RSHO"), ("C7", "T10"),
    ("LSHO", "LELB"), ("LELB", "LWRA"), ("LWRA", "LFIN"),
    ("RSHO", "RELB"), ("RELB", "RWRA"), ("RWRA", "RFIN"),
    ("T10", "LASI"), ("T10", "RASI"),
    ("LASI", "RASI"), ("RASI", "RPSI"), ("RPSI", "LPSI"), ("LPSI", "LASI"),
    ("LASI", "LKNE"), ("LKNE", "LANK"), ("LANK", "LTOE"),
    ("RASI", "RKNE"), ("RKNE", "RANK"), ("RANK", "RTOE"),
]
BODY_FRAME_NAMES = [
    "Hips", "Spine1",
    "LeftUpLeg", "LeftLeg", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightToeBase",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "Racket",
]
YELLOW_BALL_MARKERS = ("ball1", "ball2", "ball3", "ball4")
MARKER_RGBA = np.array([0.95, 0.95, 0.95, 1.0], dtype=np.float32)
YELLOW_BALL_RGBA = np.array([1.0, 0.86, 0.05, 1.0], dtype=np.float32)
LINK_RGBA = np.array([0.25, 0.75, 1.0, 1.0], dtype=np.float32)
AXIS_RGBA = [
    np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
    np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
    np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float32),
]
WORLD_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 4"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.12 0.12 0.12 1"/>
  </worldbody>
</mujoco>
"""


def add_sphere(viewer, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=position,
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom += 1


def add_line(viewer, start: np.ndarray, end: np.ndarray, width: float, rgba: np.ndarray) -> None:
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mj.mjv_initGeom(
        geom,
        type=mj.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([width, width, width]),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, width, start, end)
    viewer.user_scn.ngeom += 1


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def draw_frame_axes(viewer, position: np.ndarray, quaternion: np.ndarray, scale: float) -> None:
    matrix = quat_wxyz_to_matrix(quaternion)
    for axis in range(3):
        add_line(viewer, position, position + scale * matrix[:, axis], 0.006, AXIS_RGBA[axis])


def averaged_marker(markers: dict[str, np.ndarray], marker_names: tuple[str, ...]) -> np.ndarray | None:
    if not all(marker_name in markers for marker_name in marker_names):
        return None
    return np.mean(np.stack([markers[marker_name] for marker_name in marker_names]), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trc_path", type=Path)
    parser.add_argument("--show-body-frames", action="store_true")
    parser.add_argument("--frame-scale", type=float, default=0.12)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--once", action="store_true", help="Exit after playing the TRC once")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fps, markers = load_trc(args.trc_path)
    yellow_ball = averaged_marker(markers, YELLOW_BALL_MARKERS)
    marker_names = sorted(markers)
    frame_count = len(next(iter(markers.values())))
    model = mj.MjModel.from_xml_string(WORLD_XML)
    data = mj.MjData(model)
    viewer = mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
    viewer.cam.lookat[:] = np.mean(np.stack([values[args.start_frame] for values in markers.values()]), axis=0)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -12

    try:
        while viewer.is_running():
            for frame_index in range(args.start_frame, frame_count):
                if not viewer.is_running():
                    break
                start_time = time.perf_counter()
                viewer.user_scn.ngeom = 0
                for marker_name in marker_names:
                    add_sphere(viewer, markers[marker_name][frame_index], 0.015, MARKER_RGBA)
                if yellow_ball is not None:
                    add_sphere(viewer, yellow_ball[frame_index], 0.025, YELLOW_BALL_RGBA)
                for start_name, end_name in SKELETON_LINKS:
                    add_line(viewer, markers[start_name][frame_index], markers[end_name][frame_index], 0.008, LINK_RGBA)
                if args.show_body_frames:
                    body_frame = marker_frame(markers, frame_index)
                    for body_name in BODY_FRAME_NAMES:
                        if body_name not in body_frame:
                            continue
                        position, quaternion = body_frame[body_name]
                        draw_frame_axes(viewer, position, quaternion, args.frame_scale)
                viewer.sync()
                elapsed = time.perf_counter() - start_time
                time.sleep(max(0.0, 1.0 / (fps * args.speed) - elapsed))
            if args.once:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
