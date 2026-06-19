#!/usr/bin/env python3
"""Replay TRC segments with keyboard navigation and archiving."""

from __future__ import annotations

import argparse
import select
import shutil
import sys
import termios
import time
import tty
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

try:
    import glfw
except ImportError:
    glfw = None


KEY_LEFT = glfw.KEY_LEFT if glfw is not None else 263
KEY_RIGHT = glfw.KEY_RIGHT if glfw is not None else 262
KEY_A = glfw.KEY_A if glfw is not None else ord("A")
KEY_N = glfw.KEY_N if glfw is not None else ord("N")
KEY_P = glfw.KEY_P if glfw is not None else ord("P")
KEY_H = glfw.KEY_H if glfw is not None else ord("H")


def unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique archive path for {path}")


class SegmentPlayer:
    def __init__(self, segment_dir: Path, archive_root: Path | None) -> None:
        self.segment_dir = segment_dir
        self.archive_dir = (archive_root if archive_root is not None else segment_dir.parent / "archived_segments") / segment_dir.name
        self.clip_paths = sorted(segment_dir.glob("*.trc"))
        if not self.clip_paths:
            raise ValueError(f"No .trc clips found in {segment_dir}")

        self.current_index = 0
        self.pending_action: str | None = None
        self.fps = 0.0
        self.markers: dict[str, np.ndarray] = {}
        self.marker_names: list[str] = []
        self.yellow_ball: np.ndarray | None = None
        self.frame_count = 0
        self.load_current_clip()

    @property
    def current_path(self) -> Path:
        return self.clip_paths[self.current_index]

    def request_action(self, action: str, source: str) -> None:
        self.pending_action = action
        print(f"{source} key: {action}", flush=True)

    def on_key(self, keycode: int) -> None:
        if keycode in (KEY_RIGHT, KEY_N):
            self.request_action("next", "Viewer")
        elif keycode in (KEY_LEFT, KEY_P):
            self.request_action("previous", "Viewer")
        elif keycode == KEY_A:
            self.request_action("archive", "Viewer")
        elif keycode == KEY_H:
            self.print_help()

    def print_help(self) -> None:
        print("Keys: Right/N next clip, Left/P previous clip, A archive current clip, H help", flush=True)

    def print_status(self) -> None:
        print(
            f"Playing [{self.current_index + 1}/{len(self.clip_paths)}]: "
            f"{self.current_path.name} ({self.frame_count} frames @ {self.fps:g} Hz)",
            flush=True,
        )

    def load_current_clip(self) -> None:
        self.fps, self.markers = load_trc(self.current_path)
        self.marker_names = sorted(self.markers)
        self.yellow_ball = averaged_marker(self.markers, YELLOW_BALL_MARKERS)
        self.frame_count = len(next(iter(self.markers.values())))
        self.print_status()

    def advance(self, step: int) -> None:
        self.current_index = (self.current_index + step) % len(self.clip_paths)
        self.load_current_clip()

    def archive_current_clip(self) -> None:
        source_path = self.current_path
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        destination_path = unique_archive_path(self.archive_dir / source_path.name)
        shutil.move(str(source_path), str(destination_path))
        print(f"Archived {source_path.name} -> {destination_path}", flush=True)

        del self.clip_paths[self.current_index]
        if not self.clip_paths:
            self.markers = {}
            self.marker_names = []
            self.yellow_ball = None
            self.frame_count = 0
            print(f"No clips remain in {self.segment_dir}", flush=True)
            return

        self.current_index %= len(self.clip_paths)
        self.load_current_clip()

    def apply_pending_action(self) -> bool:
        action = self.pending_action
        self.pending_action = None
        if action is None:
            return True
        if action == "next":
            self.advance(1)
        elif action == "previous":
            self.advance(-1)
        elif action == "archive":
            self.archive_current_clip()
        return bool(self.clip_paths)


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self.original_attrs = None

    def __enter__(self) -> TerminalKeyReader:
        if sys.stdin.isatty():
            self.original_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
            self.enabled = True
            print("Terminal key fallback enabled: n/right next, p/left previous, a archive", flush=True)
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
        if key == "a":
            return "archive"
        if key == "h":
            return "help"
        return None


def set_camera(viewer, markers: dict[str, np.ndarray], frame_index: int) -> None:
    viewer.cam.lookat[:] = np.mean(np.stack([values[frame_index] for values in markers.values()]), axis=0)
    viewer.cam.distance = 3.0
    viewer.cam.elevation = -12


def draw_clip_frame(viewer, player: SegmentPlayer, frame_index: int, show_body_frames: bool, frame_scale: float) -> None:
    viewer.user_scn.ngeom = 0
    for marker_name in player.marker_names:
        add_sphere(viewer, player.markers[marker_name][frame_index], 0.015, MARKER_RGBA)
    if player.yellow_ball is not None:
        add_sphere(viewer, player.yellow_ball[frame_index], 0.025, YELLOW_BALL_RGBA)
    for start_name, end_name in SKELETON_LINKS:
        add_line(viewer, player.markers[start_name][frame_index], player.markers[end_name][frame_index], 0.008, LINK_RGBA)
    if show_body_frames:
        body_frame = marker_frame(player.markers, frame_index)
        for body_name in BODY_FRAME_NAMES:
            if body_name not in body_frame:
                continue
            position, quaternion = body_frame[body_name]
            draw_frame_axes(viewer, position, quaternion, frame_scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("segment_dir", type=Path, help="Folder containing segmented .trc clips")
    parser.add_argument("--archive-root", type=Path, help="Archive root; defaults to <segment_dir parent>/archived_segments")
    parser.add_argument("--show-body-frames", action="store_true")
    parser.add_argument("--frame-scale", type=float, default=0.12)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = SegmentPlayer(args.segment_dir, args.archive_root)
    model = mj.MjModel.from_xml_string(WORLD_XML)
    data = mj.MjData(model)
    viewer = mjv.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=player.on_key,
    )
    player.print_help()
    set_camera(viewer, player.markers, 0)

    try:
        while viewer.is_running() and player.clip_paths:
            with TerminalKeyReader() as terminal_keys:
                while viewer.is_running() and player.clip_paths:
                    for frame_index in range(player.frame_count):
                        if not viewer.is_running():
                            break

                        terminal_action = terminal_keys.poll_action()
                        if terminal_action == "help":
                            player.print_help()
                        elif terminal_action is not None:
                            player.request_action(terminal_action, "Terminal")

                        action = player.pending_action
                        if not player.apply_pending_action():
                            break
                        if action is not None:
                            if player.clip_paths:
                                set_camera(viewer, player.markers, 0)
                            break

                        start_time = time.perf_counter()
                        draw_clip_frame(viewer, player, frame_index, args.show_body_frames, args.frame_scale)
                        viewer.sync()
                        elapsed = time.perf_counter() - start_time
                        time.sleep(max(0.0, 1.0 / (player.fps * args.speed) - elapsed))
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
