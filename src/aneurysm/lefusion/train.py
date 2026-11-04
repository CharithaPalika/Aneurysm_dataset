"""Training loop for LeFusion on our aneurysm patches.

Usage (see 02_train_lefusion.ipynb):
    result = train(data_cfg, train_cfg)

Steps
-----
1. Seed everything (python, numpy, torch, CUDA, DataLoader) for reproducibility.
2. Get the data ready (split + cached patches) and build train / val loaders.
3. Build LeFusion's Unet3D + diffusion (code from LeFusion/LeFusion/ddpm, unchanged).
4. Optional memory check: one forward + backward, prints peak GPU memory.
5. Train with mixed precision, gradient accumulation and an EMA copy of the weights.
   Loss = LeFusion's lesion-focused L1 loss inside the *centred* aneurysm (center_mask).
6. Every `log_every` steps: train loss.  Every `eval_every`: validation loss at fixed
   noise levels.  Every `sample_every`: preview samples (PNG).  Every `save_every`: checkpoint.

Everything is written to  <output_dir>/<run_name>/ :
    config.yaml            data + train settings used for this run (config_resume_*.yaml when resumed)
    train_log.txt          everything printed during training
    metrics.csv            step, kind (train / val / sample), name, value
    checkpoints/latest.pt  last state (used to resume)
    checkpoints/best.pt    lowest validation loss
    checkpoints/step_XXXXXX.pt   periodic checkpoints (only the last `keep_last` are kept)
    samples/step_XXXXXX.png      preview samples

A checkpoint holds: network weights ("model"), EMA weights ("ema"), optimiser,
AMP scaler, random-number states, step, best val loss and both configs.
Stop any time with "Interrupt kernel": latest.pt is saved, and the next call
with resume=True continues from it.
"""
from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import DataConfig
from ..data.datasets import make_dataloaders
from ..data.pipeline import prepare_data
from ..env import default_num_workers, describe_environment, get_device
from .config import TrainConfig
from .model import build_diffusion, count_parameters


# ======================================================================
# Reproducibility
# ======================================================================
def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA). Optionally make cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ======================================================================
# Small helpers
# ======================================================================
class RunLogger:
    """print() that also appends to train_log.txt, plus a metrics.csv writer."""

    def __init__(self, run_dir: Path):
        self.log_path = run_dir / "train_log.txt"
        self.metrics_path = run_dir / "metrics.csv"
        new = not self.metrics_path.exists()
        self._mfile = open(self.metrics_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._mfile, fieldnames=["step", "kind", "name", "value"])
        if new:
            self._writer.writeheader()
        self.history: list[dict] = []

    def print(self, msg: str = "") -> None:
        print(msg, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def metrics(self, step: int, kind: str, values: dict) -> None:
        for name, value in values.items():
            row = {"step": step, "kind": kind, "name": name, "value": value}
            self._writer.writerow(row)
            self.history.append(row)
        self._mfile.flush()

    def close(self) -> None:
        self._mfile.close()


class EMA:
    """Exponential moving average of the network weights (smoother, used for samples)."""

    def __init__(self, model: torch.nn.Module, decay: float, start: int, every: int):
        self.decay, self.start, self.every = decay, start, every
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        if step % self.every != 0:
            return
        if step < self.start:                      # early on: just copy the weights
            self.model.load_state_dict(model.state_dict())
            return
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)


def _forever(loader):
    """Loop over a DataLoader forever (reshuffles every epoch)."""
    while True:
        for batch in loader:
            yield batch


def _to_hu(v: float, cfg: DataConfig) -> float:
    """Normalised value in [-1, 1] -> HU."""
    return (v + 1.0) / 2.0 * (cfg.hu_max - cfg.hu_min) + cfg.hu_min


def _gpu_mem_gb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m" if h else f"{m:d}m{s:02d}s"


def _plain(d: dict) -> dict:
    """Make config dicts YAML/torch friendly (Path -> str)."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


# ======================================================================
# Checkpoints
# ======================================================================
def save_checkpoint(path: Path, step: int, diffusion, ema: EMA, opt, scaler,
                    data_cfg: DataConfig, train_cfg: TrainConfig, best_val: float) -> None:
    """Write to a temporary file first, then rename -> never leaves a half-written checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": diffusion.denoise_fn.state_dict(),
        "ema": ema.model.denoise_fn.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
        "best_val": best_val,
        "patch_tag": data_cfg.patch_tag,
        "data_cfg": _plain(data_cfg.to_dict()),
        "train_cfg": train_cfg.to_dict(),
    }, tmp)
    os.replace(tmp, path)


def _prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    steps = sorted(ckpt_dir.glob("step_*.pt"))              # zero-padded names sort by step
    for old in (steps[:-keep_last] if keep_last > 0 else []):
        old.unlink(missing_ok=True)


# ======================================================================
# Evaluation and preview samples
# ======================================================================
@torch.no_grad()
def evaluate(diffusion, loader, device, timesteps: list[int], max_batches: int, amp: bool) -> dict:
    """Validation loss at fixed noise levels with fixed noise -> comparable between evaluations.

    Uses its own random generator, so it does not change the training random sequence.
    """
    diffusion.eval()
    per_t = {t: [] for t in timesteps}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device)
        m = batch["center_mask"].to(device)
        h = batch["hist"].to(device)
        for t in timesteps:
            g = torch.Generator().manual_seed(1000 * i + t)
            noise = torch.randn(x.shape, generator=g).to(device)
            tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = diffusion.p_losses(x_start=x, t=tt, mask=m, cond=h, noise=noise)
            per_t[t].append(float(loss))
    diffusion.train()
    out = {f"val_t{t}": float(np.mean(v)) for t, v in per_t.items() if v}
    out["val_loss"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


@torch.no_grad()
def save_samples(ema_diffusion, items: list[dict], data_cfg: DataConfig, train_cfg: TrainConfig,
                 device, out_path: Path, amp: bool, seed: int) -> list[dict]:
    """Rebuild the centred aneurysm of a few val patches (RePaint loop) and save one PNG.

    The real background is put back outside the mask afterwards. Runs with its own
    seed, so previews are comparable between steps and the training RNG is untouched.
    Returns mean HU inside the mask (real vs generated) for each sample.
    """
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt

    conf = OmegaConf.create({
        "schedule_jump_params": {"t_T": train_cfg.timesteps, "n_sample": 1,
                                 "jump_length": train_cfg.sample_jump_length,
                                 "jump_n_sample": train_cfg.sample_jump_n_sample},
        "inpa_inj_sched_prev_cumnoise": False,
        "inpa_inj_time_shift": 1,
    })
    gt = torch.stack([it["image"] for it in items]).to(device)
    msk = torch.stack([it["center_mask"] for it in items]).to(device)
    hist = torch.stack([it["hist"] for it in items]).to(device)

    saved = _rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ema_diffusion.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        gen = ema_diffusion.p_sample_loop_repaint(
            shape=tuple(gt.shape), model_kwargs={"gt": gt, "gt_keep_mask": msk},
            device=device, progress=False, conf=conf, cond=hist,
        )
    _set_rng_state(saved)
    gen = gen.float() * msk + gt * (1 - msk)             # real background outside the mask

    stats, n = [], len(items)
    fig, axs = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for k in range(n):
        g, r, m = gen[k, 0].cpu().numpy(), gt[k, 0].cpu().numpy(), msk[k, 0].cpu().numpy()
        z = int(np.argmax(m.sum(axis=(1, 2)))) if m.any() else g.shape[0] // 2
        real_hu = _to_hu(float(r[m > 0].mean()), data_cfg) if m.any() else float("nan")
        gen_hu = _to_hu(float(g[m > 0].mean()), data_cfg) if m.any() else float("nan")
        stats.append({"patch_id": items[k]["patch_id"], "real_hu": real_hu, "gen_hu": gen_hu})
        panels = [(r, f"real ({real_hu:.0f} HU)", "gray", -1), (g, f"generated ({gen_hu:.0f} HU)", "gray", -1),
                  (np.abs(g - r), "|difference|", "magma", 0)]
        for j, (img, title, cmap, vmin) in enumerate(panels):
            axs[k, j].imshow(img[z], cmap=cmap, vmin=vmin, vmax=1, origin="lower")
            axs[k, j].contour(m[z], levels=[0.5], colors="r", linewidths=0.8, origin="lower")
            axs[k, j].set_title(f"{items[k]['patch_id']}: {title}", fontsize=8)
            axs[k, j].axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return stats


# ======================================================================
# Main entry point
# ======================================================================
def train(data_cfg: DataConfig, train_cfg: TrainConfig) -> dict:
    """Train LeFusion. Returns {'run_dir', 'history' (DataFrame), 'best_val', 'step'}."""
    import pandas as pd

    train_cfg.validate()
    seed_everything(train_cfg.seed, train_cfg.deterministic)
    device = get_device()
    use_amp = bool(train_cfg.amp and device.type == "cuda")

    # ---- run folder: <output_dir>/<run_name>/ ----
    run_dir = data_cfg.project_root / train_cfg.output_dir / train_cfg.run_name
    ckpt_dir, sample_dir = run_dir / "checkpoints", run_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(run_dir)
    say = logger.print

    env = describe_environment()
    say(f"=== Run '{train_cfg.run_name}'  |  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  seed {train_cfg.seed} ===")
    say(f"Device: {device}" + (f" ({env.get('gpu')}, {env.get('gpu_memory_gb')} GB)" if device.type == "cuda" else ""))
    if device.type != "cuda":
        say("Note: no CUDA GPU - training will be very slow. Use the Windows GPU machine for real runs.")

    # ---- data ----
    data = prepare_data(data_cfg)
    workers = default_num_workers() if train_cfg.num_workers is None else train_cfg.num_workers
    loaders = make_dataloaders(data["index"], data_cfg, batch_size=train_cfg.batch_size,
                               num_workers=workers, seed=train_cfg.seed)
    if "train" not in loaders:
        raise RuntimeError("No training patches found. Check Dataset/ and run 01_dataset.ipynb.")
    val_loader = loaders.get("val")
    sample_items = []
    if val_loader is not None and train_cfg.sample_every > 0:
        sample_items = [val_loader.dataset[i] for i in range(min(train_cfg.n_samples, len(val_loader.dataset)))]
    say(f"Patches: {data_cfg.patch_tag} | train {len(loaders['train'].dataset)} | "
        f"val {len(val_loader.dataset) if val_loader else 0} | loader workers {workers}")

    # ---- model, optimiser, EMA ----
    diffusion = build_diffusion(data_cfg, train_cfg, device)
    say(f"LeFusion U-Net: {count_parameters(diffusion.denoise_fn):.1f} M parameters | "
        f"input (D, H, W) = ({data_cfg.depth}, {data_cfg.patch_size}, {data_cfg.patch_size}) | "
        f"base channels {train_cfg.base_channels} | AMP {use_amp}")
    opt = torch.optim.Adam(diffusion.parameters(), lr=train_cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(diffusion, train_cfg.ema_decay, train_cfg.ema_start, train_cfg.ema_every)

    # ---- resume ----
    step, best_val = 0, float("inf")
    latest = ckpt_dir / "latest.pt"
    if train_cfg.resume and latest.exists():
        ck = torch.load(latest, map_location=device)
        if ck.get("patch_tag") != data_cfg.patch_tag:
            say(f"Warning: checkpoint used patches '{ck.get('patch_tag')}', now '{data_cfg.patch_tag}'.")
        diffusion.denoise_fn.load_state_dict(ck["model"])
        ema.model.denoise_fn.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if "rng" in ck:
            _set_rng_state(ck["rng"])
        step, best_val = int(ck["step"]), float(ck.get("best_val", best_val))
        say(f"Resumed from checkpoints/latest.pt at step {step} (best val loss {best_val:.4f})")
    elif latest.exists():
        logger.close()
        raise FileExistsError(
            f"{data_cfg.rel(run_dir)} already has checkpoints. Use resume=True to continue it, "
            f"or set a new run_name to start a fresh run (nothing was overwritten).")

    # settings used for this run (a resumed run gets its own file, the first one is kept)
    cfg_name = "config.yaml" if step == 0 else f"config_resume_step{step:06d}.yaml"
    with open(run_dir / cfg_name, "w", encoding="utf-8") as f:
        yaml.safe_dump({"data": _plain(data_cfg.to_dict()), "train": train_cfg.to_dict()}, f, sort_keys=False)

    if step >= train_cfg.max_steps:
        say(f"Already at step {step} >= max_steps {train_cfg.max_steps}. Increase max_steps to continue.")
        logger.close()
        return {"run_dir": run_dir, "history": pd.DataFrame(logger.history), "best_val": best_val, "step": step}

    # ---- memory check (one forward + backward, no weight update) ----
    train_iter = _forever(loaders["train"])
    if train_cfg.memory_check and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        b = next(train_iter)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = diffusion(x=(b["image"].to(device), b["hist"].to(device)), mask=b["center_mask"].to(device))
            scaler.scale(loss).backward()
            opt.zero_grad(set_to_none=True)
            say(f"Memory check OK: peak GPU memory {_gpu_mem_gb(device):.2f} GB of "
                f"{env.get('gpu_memory_gb')} GB (batch {train_cfg.batch_size})")
        except torch.cuda.OutOfMemoryError:
            logger.close()
            raise RuntimeError(
                "Out of GPU memory in the memory check. Try, in this order: base_channels=32 "
                "(TRAIN_OVERRIDES) -> patch_depth=24 or patch_size=48 (DATA_OVERRIDES)."
            ) from None
        torch.cuda.reset_peak_memory_stats(device)

    # ---- training loop ----
    say(f"\nTraining steps {step} -> {train_cfg.max_steps} | effective batch "
        f"{train_cfg.batch_size * train_cfg.grad_accum} | outputs: {data_cfg.rel(run_dir)}\n")
    diffusion.train()
    window, t_start, t_window, start_step = [], time.time(), time.time(), step

    def checkpoint(name: str) -> None:
        save_checkpoint(ckpt_dir / name, step, diffusion, ema, opt, scaler, data_cfg, train_cfg, best_val)

    try:
        while step < train_cfg.max_steps:
            # -- one optimiser step = grad_accum mini-batches --
            opt.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(train_cfg.grad_accum):
                b = next(train_iter)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    loss = diffusion(x=(b["image"].to(device, non_blocking=True), b["hist"].to(device)),
                                     mask=b["center_mask"].to(device, non_blocking=True))
                scaler.scale(loss / train_cfg.grad_accum).backward()
                step_loss += float(loss) / train_cfg.grad_accum

            if not math.isfinite(step_loss):
                say(f"step {step}: loss is {step_loss}, skipping this update")
                opt.zero_grad(set_to_none=True)
                continue

            if train_cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), train_cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            ema.update(diffusion, step)
            window.append(step_loss)

            # -- train loss --
            if step % train_cfg.log_every == 0 or step == train_cfg.max_steps:
                sec = (time.time() - t_window) / max(1, len(window))
                avg = float(np.mean(window))
                say(f"step {step:6d}/{train_cfg.max_steps} | train loss {avg:.4f} | {sec:.2f} s/step | "
                    f"ETA {_fmt_time(sec * (train_cfg.max_steps - step))} | GPU {_gpu_mem_gb(device):.2f} GB")
                logger.metrics(step, "train", {"loss": avg, "sec_per_step": sec})
                window, t_window = [], time.time()

            # -- validation loss --
            if val_loader is not None and (step % train_cfg.eval_every == 0 or step == train_cfg.max_steps):
                res = evaluate(diffusion, val_loader, device, train_cfg.eval_timesteps,
                               train_cfg.eval_max_batches, use_amp)
                improved = res["val_loss"] < best_val
                if improved:
                    best_val = res["val_loss"]
                    checkpoint("best.pt")
                per_t = "  ".join(f"t={k[5:]}: {v:.4f}" for k, v in res.items() if k.startswith("val_t"))
                say(f"  [eval]    step {step} | val loss {res['val_loss']:.4f} ({per_t})"
                    + ("  <- best, saved best.pt" if improved else ""))
                logger.metrics(step, "val", res)

            # -- preview samples --
            if sample_items and step % train_cfg.sample_every == 0:
                t0 = time.time()
                png = sample_dir / f"step_{step:06d}.png"
                stats = save_samples(ema.model, sample_items, data_cfg, train_cfg, device, png,
                                     use_amp, seed=train_cfg.seed)
                txt = "  ".join(f"{s['patch_id']}: real {s['real_hu']:.0f} / gen {s['gen_hu']:.0f} HU" for s in stats)
                say(f"  [samples] step {step} | mean HU in mask -> {txt} | {png.name} ({time.time() - t0:.0f} s)")
                logger.metrics(step, "sample", {f"{s['patch_id']}_gen_hu": s["gen_hu"] for s in stats})

            # -- checkpoints --
            if step % train_cfg.save_every == 0 or step == train_cfg.max_steps:
                checkpoint(f"step_{step:06d}.pt")
                checkpoint("latest.pt")
                _prune_checkpoints(ckpt_dir, train_cfg.keep_last)

    except KeyboardInterrupt:
        say(f"\nStopped by user at step {step}. Saving checkpoints/latest.pt ...")
        checkpoint("latest.pt")
    finally:
        logger.close()

    say(f"\nFinished: {step - start_step} steps in {_fmt_time(time.time() - t_start)} | "
        f"best val loss {best_val:.4f}")
    say(f"Saved in {data_cfg.rel(ckpt_dir)}: latest.pt, best.pt, step_*.pt")
    return {"run_dir": run_dir, "history": pd.DataFrame(logger.history), "best_val": best_val, "step": step}
