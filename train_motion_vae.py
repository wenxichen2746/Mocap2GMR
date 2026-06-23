#!/usr/bin/env python3
"""Train or sample an autoregressive conditional Motion-VAE for GMR PKL clips."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it in the GMR environment with: pip install pyyaml") from exc


HERE = Path(__file__).resolve().parent
STATE_KEYS = ("root_pos", "root_rot", "dof_pos")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return HERE / path


def resolve_data_root(root: str | Path) -> Path:
    root_path = resolve_path(root)
    if list(root_path.glob("*.pkl")):
        return root_path

    # Allow configs to point at segments_trc/trial_xx while training from the
    # matching retargeted/trial_xx PKLs.
    if root_path.parent.name == "segments_trc":
        inferred = root_path.parent.parent / "retargeted" / root_path.name
        if list(inferred.glob("*.pkl")):
            return inferred

    raise ValueError(f"No .pkl clips found for data root {root_path}")


def list_motion_files(roots: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        data_root = resolve_data_root(root)
        files.extend(sorted(path for path in data_root.glob("*.pkl") if path.name != "segments_gmr.pkl"))
    if not files:
        raise ValueError("No motion PKL files found")
    return files


def load_motion_state(path: Path, sequence_length: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    with path.open("rb") as handle:
        motion = pickle.load(handle)

    state = np.concatenate([motion[key] for key in STATE_KEYS], axis=-1).astype(np.float32)
    if sequence_length is not None:
        if len(state) >= sequence_length:
            state = state[:sequence_length]
        else:
            pad = np.repeat(state[-1:], sequence_length - len(state), axis=0)
            state = np.concatenate([state, pad], axis=0)

    meta = {
        "fps": float(motion["fps"]),
        "source_trc": motion.get("source_trc"),
        "path": str(path),
        "root_pos_dim": motion["root_pos"].shape[-1],
        "root_rot_dim": motion["root_rot"].shape[-1],
        "dof_pos_dim": motion["dof_pos"].shape[-1],
    }
    return state, meta


def split_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_pos = state[:, :3]
    root_rot = state[:, 3:7]
    dof_pos = state[:, 7:]
    norm = np.linalg.norm(root_rot, axis=-1, keepdims=True)
    root_rot = root_rot / np.maximum(norm, 1e-8)
    return root_pos, root_rot, dof_pos


def save_motion_pkl(state: np.ndarray, path: Path, fps: float, source: str) -> None:
    root_pos, root_rot, dof_pos = split_state(state)
    motion = {
        "fps": fps,
        "root_pos": root_pos.astype(np.float64),
        "root_rot": root_rot.astype(np.float64),
        "dof_pos": dof_pos.astype(np.float64),
        "local_body_pos": None,
        "link_body_list": None,
        "source_trc": source,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(motion, handle)


def make_phase(batch_size: int, steps: int, device: torch.device) -> torch.Tensor:
    phase = torch.linspace(0.0, 2.0 * math.pi, steps, device=device)
    phase = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
    return phase.unsqueeze(0).expand(batch_size, -1, -1)


class MotionClipDataset(Dataset):
    def __init__(self, files: list[Path], sequence_length: int) -> None:
        self.files = files
        self.sequence_length = sequence_length
        states = []
        self.metadata = []
        for path in files:
            state, meta = load_motion_state(path, sequence_length)
            states.append(state)
            self.metadata.append(meta)
        self.states = np.stack(states).astype(np.float32)
        self.mean = self.states.reshape(-1, self.states.shape[-1]).mean(axis=0)
        self.std = self.states.reshape(-1, self.states.shape[-1]).std(axis=0)
        self.std = np.maximum(self.std, 1e-6)
        self.normalized = (self.states - self.mean) / self.std

    def __len__(self) -> int:
        return len(self.normalized)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.normalized[index])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[-1])


class MotionVAE(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        latent_size: int,
        encoder_layers: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_size = latent_size
        self.encoder = nn.GRU(
            input_size=state_dim + 2,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

        self.decoder_cells = nn.ModuleList()
        decoder_input = state_dim * 2 + latent_size + 2
        for layer in range(decoder_layers):
            self.decoder_cells.append(nn.GRUCell(decoder_input if layer == 0 else hidden_size, hidden_size))
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.phase_head = nn.Linear(hidden_size, 2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = make_phase(x.shape[0], x.shape[1], x.device)
        _, hidden = self.encoder(torch.cat((x, phase), dim=-1))
        last_hidden = hidden[-1]
        return self.mu(last_hidden), self.logvar(last_hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self,
        cond_state: torch.Tensor,
        z: torch.Tensor,
        steps: int,
        target: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = cond_state.shape[0]
        prev = cond_state
        hidden = [torch.zeros(batch_size, cell.hidden_size, device=cond_state.device) for cell in self.decoder_cells]
        phase = make_phase(batch_size, steps, cond_state.device)
        outputs = [cond_state]
        phase_outputs = []

        for step in range(1, steps):
            decoder_input = torch.cat((prev, cond_state, z, phase[:, step - 1]), dim=-1)
            for layer, cell in enumerate(self.decoder_cells):
                hidden[layer] = cell(decoder_input, hidden[layer])
                decoder_input = hidden[layer]
            delta = self.state_head(hidden[-1])
            pred = prev + delta
            outputs.append(pred)
            phase_outputs.append(self.phase_head(hidden[-1]))

            if target is not None and random.random() < teacher_forcing:
                prev = target[:, step]
            else:
                prev = pred

        return torch.stack(outputs, dim=1), torch.stack(phase_outputs, dim=1)

    def forward(self, x: torch.Tensor, teacher_forcing: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred, phase_pred = self.decode(x[:, 0], z, x.shape[1], target=x, teacher_forcing=teacher_forcing)
        return pred, phase_pred, mu, logvar


@dataclass
class Stats:
    mean: torch.Tensor
    std: torch.Tensor

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def kl_weight(epoch: int, beta_max: float, cycle_epochs: int) -> float:
    if beta_max <= 0.0:
        return 0.0
    if cycle_epochs <= 0:
        return beta_max
    progress = (epoch % cycle_epochs) / max(cycle_epochs - 1, 1)
    return beta_max * progress


def teacher_forcing(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    alpha = (epoch - 1) / (epochs - 1)
    return start + alpha * (end - start)


def compute_loss(
    batch: torch.Tensor,
    pred: torch.Tensor,
    phase_pred: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    stats: Stats,
    cfg: dict[str, Any],
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec = F.mse_loss(pred[:, 1:], batch[:, 1:])
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    phase_target = make_phase(batch.shape[0], batch.shape[1], batch.device)[:, 1:]
    phase_loss = F.mse_loss(phase_pred, phase_target)
    phase_vel_loss = F.mse_loss(phase_pred[:, 1:] - phase_pred[:, :-1], phase_target[:, 1:] - phase_target[:, :-1])

    smooth = F.mse_loss(pred[:, 1], batch[:, 0])

    denorm = stats.denormalize(pred)
    root_z = denorm[:, :, 2]
    ground = torch.relu(float(cfg["loss"]["ground_height"]) - root_z).mean()

    total = (
        rec
        + beta * kl
        + float(cfg["loss"]["phase_weight"]) * phase_loss
        + float(cfg["loss"]["phase_velocity_weight"]) * phase_vel_loss
        + float(cfg["loss"]["smooth_weight"]) * smooth
        + float(cfg["loss"]["ground_weight"]) * ground
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "rec": float(rec.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "beta": float(beta),
        "phase": float(phase_loss.detach().cpu()),
        "phase_vel": float(phase_vel_loss.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "ground": float(ground.detach().cpu()),
    }
    return total, metrics


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(
    path: Path,
    model: MotionVAE,
    optimizer: torch.optim.Optimizer,
    dataset: MotionClipDataset,
    cfg: dict[str, Any],
    epoch: int,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "mean": dataset.mean,
        "std": dataset.std,
        "state_dim": dataset.state_dim,
        "metadata": dataset.metadata,
        "history": history,
    }
    torch.save(payload, path)


def train(config_path: Path) -> None:
    cfg = load_config(config_path)
    random.seed(int(cfg["data"]["seed"]))
    np.random.seed(int(cfg["data"]["seed"]))
    torch.manual_seed(int(cfg["data"]["seed"]))

    files = list_motion_files(cfg["data"]["roots"])
    dataset = MotionClipDataset(files, int(cfg["data"]["sequence_length"]))
    train_count = max(1, int(len(dataset) * float(cfg["data"]["train_split"])))
    val_count = len(dataset) - train_count
    generator = torch.Generator().manual_seed(int(cfg["data"]["seed"]))
    train_set, val_set = random_split(dataset, [train_count, val_count], generator=generator)

    loader = DataLoader(train_set, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, drop_last=False) if val_count else None

    device = choose_device(str(cfg["training"]["device"]))
    model = MotionVAE(dataset.state_dim, **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    stats = Stats(torch.from_numpy(dataset.mean).float(), torch.from_numpy(dataset.std).float())

    output_dir = resolve_path(cfg["training"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    history: list[dict[str, float]] = []
    epochs = int(cfg["training"]["epochs"])
    print(f"Loaded {len(dataset)} clips from {len(files)} files; state_dim={dataset.state_dim}; device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        beta = kl_weight(epoch - 1, float(cfg["loss"]["beta_max"]), int(cfg["loss"]["kl_cycle_epochs"]))
        tf = teacher_forcing(
            epoch,
            epochs,
            float(cfg["training"]["teacher_forcing_start"]),
            float(cfg["training"]["teacher_forcing_end"]),
        )
        epoch_metrics = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, phase_pred, mu, logvar = model(batch, teacher_forcing=tf)
            loss, metrics = compute_loss(batch, pred, phase_pred, mu, logvar, stats, cfg, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            optimizer.step()
            epoch_metrics.append(metrics)

        avg = {key: float(np.mean([m[key] for m in epoch_metrics])) for key in epoch_metrics[0]}
        avg["epoch"] = epoch
        avg["teacher_forcing"] = tf

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred, phase_pred, mu, logvar = model(batch, teacher_forcing=0.0)
                    _, metrics = compute_loss(batch, pred, phase_pred, mu, logvar, stats, cfg, beta)
                    val_losses.append(metrics["loss"])
            avg["val_loss"] = float(np.mean(val_losses))

        history.append(avg)
        print(
            f"epoch {epoch:04d} loss={avg['loss']:.6f} rec={avg['rec']:.6f} "
            f"kl={avg['kl']:.6f} beta={avg['beta']:.5f} tf={tf:.3f}"
            + (f" val={avg['val_loss']:.6f}" if "val_loss" in avg else "")
        )

        if epoch % int(cfg["training"]["checkpoint_every"]) == 0 or epoch == epochs:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", model, optimizer, dataset, cfg, epoch, history)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, dataset, cfg, epoch, history)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
        handle.write("\n")


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[MotionVAE, dict[str, Any], Stats]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = MotionVAE(int(checkpoint["state_dim"]), **cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    stats = Stats(
        torch.as_tensor(checkpoint["mean"], dtype=torch.float32),
        torch.as_tensor(checkpoint["std"], dtype=torch.float32),
    )
    return model, cfg, stats


def generate(config_path: Path, checkpoint_path: Path | None, num_samples: int | None, output_dir: Path | None) -> None:
    cfg = load_config(config_path)
    checkpoint = resolve_path(checkpoint_path or cfg["generation"]["checkpoint"])
    device = choose_device(str(cfg["training"]["device"]))
    model, saved_cfg, stats = load_model_from_checkpoint(checkpoint, device)

    files = list_motion_files(saved_cfg["data"]["roots"])
    dataset = MotionClipDataset(files, int(saved_cfg["data"]["sequence_length"]))
    out_dir = resolve_path(output_dir or cfg["generation"]["output_dir"])
    count = int(num_samples or cfg["generation"]["num_samples"])
    temperature = float(cfg["generation"]["temperature"])
    fps = float(saved_cfg["data"]["fps"])

    for index in range(count):
        seed_state = torch.from_numpy(dataset.normalized[index % len(dataset), 0]).float().unsqueeze(0).to(device)
        z = torch.randn(1, model.latent_size, device=device) * temperature
        with torch.no_grad():
            pred, _ = model.decode(seed_state, z, int(saved_cfg["data"]["sequence_length"]), teacher_forcing=0.0)
            state = stats.denormalize(pred).squeeze(0).cpu().numpy()
        save_motion_pkl(state, out_dir / f"sample_{index:03d}.pkl", fps, source=f"motion_vae:{checkpoint}")
    print(f"Saved {count} generated PKLs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "motion_vae_config.yml")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args.config)
    else:
        generate(args.config, args.checkpoint, args.num_samples, args.output_dir)


if __name__ == "__main__":
    main()


"""
python train_motion_vae.py \
  --config motion_vae_config.yml \
  --mode train

python train_motion_vae.py \
  --config motion_vae_config.yml \
  --mode generate \
  --checkpoint motion_vae/checkpoints/latest.pt \
  --num-samples 20 \
  --output-dir motion_vae/generated
"""