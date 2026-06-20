#!/usr/bin/env python3
"""Replay matched TRC clips and retargeted GMR pickle files in adjacent MuJoCo windows."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import re
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

from trc_to_gmr import load_trc, marker_frame
from visualize_trc import (
    BODY_FRAME_NAMES,
    LINK_RGBA,
    MARKER_RGBA,
    SKELETON_LINKS,
    WORLD_XML,
    YELLOW_BALL_MARKERS,
    YELLOW_BALL_RGBA,
    add_line,
    add_sphere,
    averaged_marker,
    draw_frame_axes,
)


HERE = Path(__file__).resolve().parent
GMR_ROOT = HERE.parent / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion  # noqa: E402

try:
    import glfw
except ImportError:
    glfw = None


SEGMENT_RE = re.compile(r"(?:^|_)seg_(\d+)(?:_|\.|$)")
KEY_LEFT = glfw.KEY_LEFT if glfw is not None else 263
KEY_RIGHT = glfw.KEY_RIGHT if glfw is not None else 262
KEY_N = glfw.KEY_N if glfw is not None else ord("N")
KEY_P = glfw.KEY_P if glfw is not None else ord("P")
KEY_H = glfw.KEY_H if glfw is not None else ord("H")


def segment_key(path: Path) -> str | None:
    match = SEGMENT_RE.search(path.stem)
    if match is None:
        return None
    return f"seg_{int(match.group(1)):03d}"


def find_pairs(trc_dir: Path, pkl_dir: Path) -> list[tuple[str, Path, Path]]:
    trc_by_segment = {}
    for trc_path in sorted(trc_dir.glob("*.trc")):
        key = segment_key(trc_path)
        if key is not None:
            trc_by_segment[key] = trc_path

    pkl_by_segment = {}
    for pkl_path in sorted(pkl_dir.glob("*.pkl")):
        key = segment_key(pkl_path)
        if key is not None:
            pkl_by_segment[key] = pkl_path

    keys = sorted(set(trc_by_segment) & set(pkl_by_segment))
    return [(key, trc_by_segment[key], pkl_by_segment[key]) for key in keys]


@dataclass
class TrcClip:
    fps: float
    markers: dict[str, np.ndarray]
    marker_names: list[str]
    yellow_ball: np.ndarray | None
    frame_count: int


@dataclass
class RobotClip:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self.original_attrs = None

    def __enter__(self) -> TerminalKeyReader:
        if sys.stdin.isatty():
            self.original_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
            self.enabled = True
            print("Terminal keys: n/right next, p/left previous, h help, q quit", flush=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.enabled and self.original_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.original_attrs)

    def poll_action(self) -> str | None:
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None

        key = sys.stdin.read(1)
        if key == "\x1b":
            sequence = key
            while select.select([sys.stdin], [], [], 0.001)[0]:
                sequence += sys.stdin.read(1)
            if sequence == "\x1b[C":
                return "next"
            if sequence == "\x1b[D":
                return "previous"
            return None

        key = key.lower()
        if key == "n":
            return "next"
        if key == "p":
            return "previous"
        if key == "h":
            return "help"
        if key == "q":
            return "quit"
        return None


def viewer_key_callback(command_queue, keycode: int) -> None:
    if keycode in (KEY_RIGHT, KEY_N):
        command_queue.put("next")
    elif keycode in (KEY_LEFT, KEY_P):
        command_queue.put("previous")
    elif keycode == KEY_H:
        command_queue.put("help")


def set_window_position(viewer, x: int, y: int) -> None:
    if glfw is None:
        return
    window = getattr(viewer, "_window", None) or getattr(viewer, "window", None)
    if window is not None:
        glfw.set_window_pos(window, x, y)


def load_trc_clip(path: Path) -> TrcClip:
    fps, markers = load_trc(path)
    return TrcClip(
        fps=fps,
        markers=markers,
        marker_names=sorted(markers),
        yellow_ball=averaged_marker(markers, YELLOW_BALL_MARKERS),
        frame_count=len(next(iter(markers.values()))),
    )


def load_robot_clip(path: Path) -> RobotClip:
    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(path)
    return RobotClip(fps=fps, root_pos=root_pos, root_rot=root_rot, dof_pos=dof_pos)


def set_trc_camera(viewer, markers: dict[str, np.ndarray], frame_index: int) -> None:
    viewer.cam.lookat[:] = np.mean(np.stack([values[frame_index] for values in markers.values()]), axis=0)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -12


def draw_trc_frame(viewer, clip: TrcClip, frame_index: int, show_body_frames: bool, frame_scale: float) -> None:
    viewer.user_scn.ngeom = 0
    for marker_name in clip.marker_names:
        add_sphere(viewer, clip.markers[marker_name][frame_index], 0.015, MARKER_RGBA)
    if clip.yellow_ball is not None:
        add_sphere(viewer, clip.yellow_ball[frame_index], 0.025, YELLOW_BALL_RGBA)
    for start_name, end_name in SKELETON_LINKS:
        add_line(viewer, clip.markers[start_name][frame_index], clip.markers[end_name][frame_index], 0.008, LINK_RGBA)
    if show_body_frames:
        body_frame = marker_frame(clip.markers, frame_index)
        for body_name in BODY_FRAME_NAMES:
            if body_name not in body_frame:
                continue
            position, quaternion = body_frame[body_name]
            draw_frame_axes(viewer, position, quaternion, frame_scale)


def current_version(version) -> int:
    with version.get_lock():
        return version.value


def current_index(index) -> int:
    with index.get_lock():
        return index.value


def trc_viewer_worker(
    pairs: list[tuple[str, str, str]],
    index,
    version,
    stop_event,
    command_queue,
    show_body_frames: bool,
    frame_scale: float,
    speed: float,
) -> None:
    model = mj.MjModel.from_xml_string(WORLD_XML)
    data = mj.MjData(model)
    viewer = mjv.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=lambda keycode: viewer_key_callback(command_queue, keycode),
    )
    set_window_position(viewer, 40, 80)

    clip = None
    seen_version = -1
    frame_index = 0
    try:
        while viewer.is_running() and not stop_event.is_set():
            if current_version(version) != seen_version:
                seen_version = current_version(version)
                segment, trc_path, _ = pairs[current_index(index)]
                clip = load_trc_clip(Path(trc_path))
                frame_index = 0
                set_trc_camera(viewer, clip.markers, 0)
                print(f"[TRC] {segment}: {Path(trc_path).name}", flush=True)

            if clip is None:
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()
            draw_trc_frame(viewer, clip, frame_index, show_body_frames, frame_scale)
            viewer.sync()
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, 1.0 / (clip.fps * speed) - elapsed))
            frame_index = (frame_index + 1) % clip.frame_count
    finally:
        stop_event.set()
        viewer.close()


def robot_viewer_worker(
    pairs: list[tuple[str, str, str]],
    index,
    version,
    stop_event,
    command_queue,
    robot: str,
    speed: float,
) -> None:
    _, _, first_pkl = pairs[0]
    first_clip = load_robot_clip(Path(first_pkl))
    robot_viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=first_clip.fps,
        camera_follow=False,
        keyboard_callback=lambda keycode: viewer_key_callback(command_queue, keycode),
    )
    set_window_position(robot_viewer.viewer, 760, 80)

    clip = None
    seen_version = -1
    frame_index = 0
    try:
        while robot_viewer.viewer.is_running() and not stop_event.is_set():
            if current_version(version) != seen_version:
                seen_version = current_version(version)
                segment, _, pkl_path = pairs[current_index(index)]
                clip = load_robot_clip(Path(pkl_path))
                frame_index = 0
                print(f"[Robot] {segment}: {Path(pkl_path).name}", flush=True)

            if clip is None:
                time.sleep(0.01)
                continue

            start_time = time.perf_counter()
            robot_viewer.step(
                clip.root_pos[frame_index],
                clip.root_rot[frame_index],
                clip.dof_pos[frame_index],
                rate_limit=False,
                follow_camera=True,
            )
            elapsed = time.perf_counter() - start_time
            time.sleep(max(0.0, 1.0 / (clip.fps * speed) - elapsed))
            frame_index = (frame_index + 1) % len(clip.root_pos)
    finally:
        stop_event.set()
        robot_viewer.close()


def print_help() -> None:
    print("Keys: Right/N next matched clip, Left/P previous matched clip, H help, Q quit", flush=True)


def print_status(pairs: list[tuple[str, str, str]], index: int) -> None:
    segment, trc_path, pkl_path = pairs[index]
    print(
        f"Playing [{index + 1}/{len(pairs)}] {segment}: "
        f"{Path(trc_path).name} <-> {Path(pkl_path).name}",
        flush=True,
    )


def apply_action(action: str, pairs: list[tuple[str, str, str]], index, version, stop_event) -> None:
    if action == "quit":
        stop_event.set()
        return
    if action == "help":
        print_help()
        return
    if action not in {"next", "previous"}:
        return

    with index.get_lock(), version.get_lock():
        step = 1 if action == "next" else -1
        index.value = (index.value + step) % len(pairs)
        version.value += 1
        new_index = index.value
    print_status(pairs, new_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trc_dir", type=Path, help="Folder containing segmented .trc clips")
    parser.add_argument("pkl_dir", type=Path, help="Folder containing matched seg_XXX.pkl files")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--show-body-frames", action="store_true")
    parser.add_argument("--frame-scale", type=float, default=0.12)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_pairs = find_pairs(args.trc_dir, args.pkl_dir)
    if not path_pairs:
        raise ValueError(f"No matching seg_XXX TRC/PKL pairs found in {args.trc_dir} and {args.pkl_dir}")

    pairs = [(segment, str(trc_path), str(pkl_path)) for segment, trc_path, pkl_path in path_pairs]
    index = mp.Value("i", 0)
    version = mp.Value("i", 0)
    stop_event = mp.Event()
    command_queue = mp.Queue()

    print_status(pairs, 0)
    print_help()

    workers = [
        mp.Process(
            target=trc_viewer_worker,
            args=(pairs, index, version, stop_event, command_queue, args.show_body_frames, args.frame_scale, args.speed),
            name="trc-viewer",
        ),
        mp.Process(
            target=robot_viewer_worker,
            args=(pairs, index, version, stop_event, command_queue, args.robot, args.speed),
            name="robot-viewer",
        ),
    ]
    for worker in workers:
        worker.start()

    try:
        with TerminalKeyReader() as terminal_keys:
            while not stop_event.is_set() and all(worker.is_alive() for worker in workers):
                terminal_action = terminal_keys.poll_action()
                if terminal_action is not None:
                    apply_action(terminal_action, pairs, index, version, stop_event)

                try:
                    action = command_queue.get(timeout=0.03)
                except queue.Empty:
                    continue
                apply_action(action, pairs, index, version, stop_event)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)


if __name__ == "__main__":
    main()
