#!/usr/bin/env python3
"""Replay a GMR robot-motion pickle in the MuJoCo viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "GMR"))

from general_motion_retargeting import RobotMotionViewer, load_robot_motion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_motion_path", type=Path)
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--once", action="store_true", help="Exit after playing the motion once")
    args = parser.parse_args()

    _, fps, root_pos, root_rot, dof_pos, _, _ = load_robot_motion(args.robot_motion_path)
    viewer = RobotMotionViewer(robot_type=args.robot, motion_fps=fps, camera_follow=False)
    try:
        while True:
            for frame in range(len(root_pos)):
                viewer.step(root_pos[frame], root_rot[frame], dof_pos[frame], rate_limit=True)
            if args.once:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
