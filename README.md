# TT Smash Mocap to GMR

This folder contains utilities for preparing table-tennis smash Vicon TRC data, segmenting the motion around hitting events, checking the segmented clips, retargeting the clips through GMR, and replaying the transformed robot trajectory.

## Environment

Set up the Python and MuJoCo environment by following the installation guide in [YanjieZe/GMR](https://github.com/YanjieZe/GMR). These scripts import the local GMR package directly, so place this folder beside the cloned `GMR` repository:

```text
workspace/
├── GMR/
└── TT_smash_mocap/
```

Run the commands below from `TT_smash_mocap` after activating the GMR environment.

## 1. Segment TRC Files

`segment_trc_smash.py` splits a full TRC recording into shorter clips centered on racket-ball contact events. It estimates the racket center and ball center from their marker sets, finds frames where their distance falls below a contact threshold, and writes one segment around each hit.

```bash
python segment_trc_smash.py mocap_data/session_001.trc \
  --output-root segments_trc \
  --threshold 0.1 \
  --pre-seconds 0.54 \
  --post-seconds 0.54
```

Outputs are written under:

```text
segments_trc/<source_trc_name>/
├── <source_trc_name>_seg_000_hit_000123.trc
├── <source_trc_name>_seg_001_hit_000456.trc
└── segments.json
```

Use `--allow-partial` if you want to keep events near the beginning or end of a recording even when the full pre/post context is not available.

## 2. Inspect Segmented Clips

`visualize_trc_segment.py` loads a directory of segmented TRC clips and replays them in a MuJoCo marker viewer. It is intended for checking whether the automatic segmentation found useful hits before running retargeting.

```bash
python visualize_trc_segment.py segments_trc/session_001 --show-body-frames
```

Useful controls:

- Right arrow or `n`: next clip
- Left arrow or `p`: previous clip
- `a`: archive the current clip
- `h`: print help

Archived clips are moved out of the active segment folder so the remaining directory can be used as a cleaned set for conversion.

## 3. Convert TRC to GMR Robot Motion

`trc_to_gmr.py` converts a TRC file into a GMR-compatible robot-motion pickle. It reads Vicon Plug-in Gait markers, converts Vicon Y-up coordinates to MuJoCo/GMR Z-up coordinates, estimates body target frames, and passes those frames to GMR.

```bash
python trc_to_gmr.py segments_trc/session_001/session_001_seg_000_hit_000123.trc \
  --output retargeted/session_001_seg_000_hit_000123_g1.pkl \
  --robot unitree_g1 \
  --target-fps 30
```

If measured subject height is available, pass it explicitly:

```bash
python trc_to_gmr.py segments_trc/session_001/session_001_seg_000_hit_000123.trc \
  --output retargeted/session_001_seg_000_hit_000123_g1.pkl \
  --robot unitree_g1 \
  --human-height 1.70
```

The script inserts `../GMR` into `sys.path` and creates:

```python
GeneralMotionRetargeting(
    src_human="trc_vicon",
    tgt_robot="unitree_g1",
    actual_human_height=height,
)
```

Inside GMR, `src_human="trc_vicon"` selects the TRC-specific inverse-kinematics config for Unitree G1. The generated pickle stores `root_pos`, `root_rot`, and `dof_pos` in the format expected by GMR's robot-motion viewer.

For a quick validation without loading GMR:

```bash
python trc_to_gmr.py mocap_data/session_001.trc \
  --output retargeted/test.pkl \
  --inspect-only
```

To convert every segmented TRC clip in a folder, use `trsegs_to_gmr`. It writes matched pickle names such as `seg_000.pkl`, `seg_001.pkl`, and `seg_002.pkl`, which makes it easy to pair the original segment with the retargeted robot trajectory.

```bash
python trsegs_to_gmr segments_trc/session_001 \
  --output-dir retargeted/session_001 \
  --robot unitree_g1 \
  --target-fps 30
```

## 4. Replay Transformed Robot Trajectories

`visualize_robot_pkl.py` loads the transformed GMR pickle and replays the Unitree G1 trajectory in the GMR MuJoCo viewer.

```bash
python visualize_robot_pkl.py retargeted/session_001_seg_000_hit_000123_g1.pkl \
  --robot unitree_g1
```

Use `--once` to play a trajectory one time and exit:

```bash
python visualize_robot_pkl.py retargeted/session_001_seg_000_hit_000123_g1.pkl \
  --robot unitree_g1 \
  --once
```

To compare segmented TRC markers and retargeted robot motion side by side, use `visualize_trcandpkl.py`. It matches files by the `seg_XXX` token, so `session_001_seg_000_hit_000123.trc` is paired with `seg_000.pkl`.

```bash
python visualize_trcandpkl.py segments_trc/session_001 retargeted/session_001 \
  --robot unitree_g1 \
  --show-body-frames
```

Useful controls:

- Right arrow or `n`: next matched clip
- Left arrow or `p`: previous matched clip
- `h`: print help

## 5. Train a Motion-VAE

`train_motion_vae.py` trains an autoregressive conditional Motion-VAE over retargeted GMR pickle clips. Each frame is represented as:

```text
[root_pos, root_rot, dof_pos]
```

The default config is [motion_vae_config.yml](motion_vae_config.yml). Its data roots can point either to retargeted PKL folders or to segment folders such as `segments_trc/trial_05`; segment folders are resolved to the matching `retargeted/<trial_name>` folder.

```bash
python train_motion_vae.py --config motion_vae_config.yml
```

Checkpoints and training history are written under:

```text
motion_vae/
├── checkpoints/
│   ├── latest.pt
│   └── epoch_XXXX.pt
└── history.json
```

Generate visualizable GMR pickle samples from a trained checkpoint:

```bash
python train_motion_vae.py \
  --config motion_vae_config.yml \
  --mode generate \
  --checkpoint motion_vae/checkpoints/latest.pt \
  --num-samples 20 \
  --output-dir motion_vae/generated
```

Generated samples keep the same pickle layout as retargeted clips, so they can be replayed with:

```bash
python visualize_robot_pkl.py motion_vae/generated/sample_000.pkl --robot unitree_g1
```

To browse every generated sample and filter out low-quality motions:

```bash
python visualize_robot_pkl.py motion_vae/generated --robot unitree_g1
```

Useful controls:

- Right arrow or `n`: next generated clip
- Left arrow or `p`: previous generated clip
- `a`: archive the current clip under `motion_vae/generated/archived`
- `h`: print help

## Working Data Folders

The local working folders below are ignored by git:

- `mocap_data/`: raw TRC recordings
- `segments_trc/`: generated TRC segments
- `retargeted/`: generated GMR robot-motion pickle files
- `motion_vae/`: generated VAE checkpoints, logs, and sampled pickle files
