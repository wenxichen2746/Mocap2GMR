#!/usr/bin/env python3
"""Replay one GMR robot-motion pickle or browse a folder of pickles."""

from __future__ import annotations

import argparse
import select
import shutil
import sys
import termios
import tty
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "GMR"))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion

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


class TerminalKeyReader:
    def __init__(self) -> None:
        self.enabled = False
        self.original_attrs = None

    def __enter__(self) -> TerminalKeyReader:
        if sys.stdin.isatty():
            self.original_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
            self.enabled = True
            print("Terminal keys: n/right next, p/left previous, a archive, h help", flush=True)
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


class RobotPklPlayer:
    def __init__(self, motion_path: Path, archive_root: Path | None) -> None:
        self.single_file = motion_path.is_file()
        self.motion_dir = motion_path.parent if self.single_file else motion_path
        self.archive_dir = archive_root if archive_root is not None else self.motion_dir / "archived"
        self.clip_paths = [motion_path] if self.single_file else sorted(
            path for path in motion_path.glob("*.pkl") if path.parent.name != "archived"
        )
        if not self.clip_paths:
            raise ValueError(f"No .pkl files found in {motion_path}")

        self.current_index = 0
        self.pending_action: str | None = None
        self.fps = 30.0
        self.root_pos = None
        self.root_rot = None
        self.dof_pos = None
        self.load_current_clip()

    @property
    def current_path(self) -> Path:
        return self.clip_paths[self.current_index]

    def print_help(self) -> None:
        if self.single_file:
            print("Keys: H help", flush=True)
        else:
            print("Keys: Right/N next clip, Left/P previous clip, A archive current clip, H help", flush=True)

    def print_status(self) -> None:
        print(
            f"Playing [{self.current_index + 1}/{len(self.clip_paths)}]: "
            f"{self.current_path.name} ({len(self.root_pos)} frames @ {self.fps:g} Hz)",
            flush=True,
        )

    def load_current_clip(self) -> None:
        _, self.fps, self.root_pos, self.root_rot, self.dof_pos, _, _ = load_robot_motion(self.current_path)
        self.print_status()

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

    def advance(self, step: int) -> None:
        if self.single_file:
            return
        self.current_index = (self.current_index + step) % len(self.clip_paths)
        self.load_current_clip()

    def archive_current_clip(self) -> None:
        if self.single_file:
            print("Archive is disabled for single-file playback", flush=True)
            return

        source_path = self.current_path
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        destination_path = unique_archive_path(self.archive_dir / source_path.name)
        shutil.move(str(source_path), str(destination_path))
        print(f"Archived {source_path.name} -> {destination_path}", flush=True)

        del self.clip_paths[self.current_index]
        if not self.clip_paths:
            self.root_pos = None
            self.root_rot = None
            self.dof_pos = None
            print(f"No clips remain in {self.motion_dir}", flush=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_motion_path", type=Path, help="A .pkl file or a folder containing .pkl files")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--once", action="store_true", help="Exit after playing the current motion once")
    parser.add_argument("--archive-root", type=Path, help="Archive folder; defaults to <pkl folder>/archived")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = RobotPklPlayer(args.robot_motion_path, args.archive_root)
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=player.fps,
        camera_follow=False,
        keyboard_callback=player.on_key,
    )
    player.print_help()

    try:
        with TerminalKeyReader() as terminal_keys:
            while viewer.viewer.is_running() and player.clip_paths:
                for frame in range(len(player.root_pos)):
                    if not viewer.viewer.is_running():
                        break

                    terminal_action = terminal_keys.poll_action()
                    if terminal_action == "help":
                        player.print_help()
                    elif terminal_action is not None:
                        player.request_action(terminal_action, "Terminal")

                    previous_path = player.current_path
                    action = player.pending_action
                    if not player.apply_pending_action():
                        break
                    if action is not None or player.current_path != previous_path:
                        break

                    viewer.step(
                        player.root_pos[frame],
                        player.root_rot[frame],
                        player.dof_pos[frame],
                        rate_limit=True,
                    )
                if args.once:
                    break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
