# TT Smash TRC to GMR

Run these commands from `TT_smash_mocap` after activating the `gmr` conda environment.

Validate the TRC marker mapping without loading MuJoCo:

```bash
python trc_to_gmr.py mocap_data/footwork_03.trc -o retargeted/footwork_03_g1.pkl --inspect-only
```

Replay the raw TRC markers and a connected skeleton before retargeting:

```bash
python visualize_trc.py mocap_data/footwork_03.trc
```

Show the inferred body-frame axes passed into GMR. Red, green, and blue lines are the
local X, Y, and Z axes:

```bash
python visualize_trc.py mocap_data/footwork_03.trc --show-body-frames
```

Retarget the Vicon markers to Unitree G1 poses:

```bash
python trc_to_gmr.py mocap_data/footwork_03.trc -o retargeted/footwork_03_g1.pkl --robot unitree_g1
```

Pass the measured subject height when available:

```bash
python trc_to_gmr.py mocap_data/footwork_03.trc -o retargeted/footwork_03_g1.pkl --human-height 1.70
```

Replay the generated pickle in GMR's MuJoCo viewer:

```bash
python visualize_robot_pkl.py retargeted/footwork_03_g1.pkl --robot unitree_g1
```

Use `--once` on the viewer to stop after one playback. Use `--max-frames 100` on the
converter for a short test run.

The adapter reads the Plug-in Gait markers from `.trc`, converts Vicon Y-up coordinates
to MuJoCo Z-up coordinates, and estimates body transforms. It uses the dedicated
`trc_vicon_pos_only_to_g1.json` inverse-kinematics configuration. This mapping tracks
positions only: every orientation loss weight is zero and the stored quaternions are
valid identity placeholders. This initial adapter supports Unitree G1.
