"""Figures shared by training previews and evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MASK_COLORS = {"actual": "yellow", "dilated": "red"}


def _contour(ax, m2d: np.ndarray, color: str) -> None:
    if m2d.any():
        ax.contour(m2d, levels=[0.5], colors=color, linewidths=0.8, origin="lower")


def plot_rows(rows: list[dict], out_path: Path | None = None, show: bool = False, title: str = "",
              names: dict | None = None):
    """One row per sample: real | (input) | generated | |difference|, middle slice through the mask.

    rows: dicts with
        'real', 'gen', 'mask'  (D, H, W) numpy, [-1, 1] / 0-1
        'label'                str
        'input'      optional  background given to the model -> extra panel
        'outer_mask' optional  larger mask: then 'mask' is drawn yellow and 'outer_mask' red
                               (without it, 'mask' is drawn red)
    names: rename panels, e.g. {"generated": "pseudo-healthy"}
    """
    import matplotlib.pyplot as plt

    names = {"real": "real", "input": "input", "generated": "generated", "difference": "|difference|",
             **(names or {})}
    with_input = any("input" in r for r in rows)
    n, cols = len(rows), (4 if with_input else 3)
    fig, axs = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for k, r in enumerate(rows):
        m, outer = r["mask"], r.get("outer_mask")
        ref = outer if outer is not None else m
        z = int(np.argmax(ref.sum(axis=(1, 2)))) if ref.any() else ref.shape[0] // 2
        panels = [(r["real"], names["real"], "gray", -1)]
        if with_input:
            panels.append((r.get("input", r["real"]), names["input"], "gray", -1))
        panels += [(r["gen"], names["generated"], "gray", -1),
                   (np.abs(r["gen"] - r["real"]), names["difference"], "magma", 0)]
        for j, (img, name, cmap, vmin) in enumerate(panels):
            ax = axs[k, j]
            ax.imshow(img[z], cmap=cmap, vmin=vmin, vmax=1, origin="lower")
            if outer is None:
                _contour(ax, m[z], "r")
            else:
                _contour(ax, outer[z], "r")
                _contour(ax, m[z], "y")
            ax.set_title(f"{r['label']}  {name}", fontsize=8)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def plot_panels(rows: list[dict], columns: list[dict], out_path: Path | None = None, show: bool = False,
                title: str = "", value_range: tuple = (-100, 900), diff_range: tuple = (0, 500),
                unit: str = "HU") -> None:
    """Flexible grid: one row per sample, one column per entry of `columns`.

    rows: dicts of (D, H, W) arrays (images in `unit`, masks 0/1) + 'label' (str).
          Mask keys are the keys of MASK_COLORS ('actual' = yellow, 'dilated' = red).
    columns: list of dicts, e.g.
        {"title": "real",               "show": "real",          "masks": ["actual", "dilated"]}
        {"title": "|generated - real|", "diff": ("gen", "real"), "masks": ["actual"]}
    The slice shown is the one where the outer mask is largest (dilated if present, else actual).
    Difference panels get a colour bar.
    """
    import matplotlib.pyplot as plt

    n, c = len(rows), len(columns)
    fig, axs = plt.subplots(n, c, figsize=(3.2 * c, 3.1 * n), squeeze=False)
    for k, r in enumerate(rows):
        ref = r["dilated"] if r.get("dilated") is not None else r["actual"]
        z = int(np.argmax(ref.sum(axis=(1, 2)))) if ref.any() else ref.shape[0] // 2
        for j, col in enumerate(columns):
            ax = axs[k, j]
            if "diff" in col:
                a, b = col["diff"]
                im = ax.imshow(np.abs(r[a][z] - r[b][z]), cmap="magma", vmin=diff_range[0], vmax=diff_range[1],
                               origin="lower")
                bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                bar.ax.tick_params(labelsize=6)
                bar.set_label(unit, fontsize=6)
            else:
                ax.imshow(r[col["show"]][z], cmap="gray", vmin=value_range[0], vmax=value_range[1], origin="lower")
            for m in col.get("masks", []):
                if r.get(m) is not None:
                    _contour(ax, r[m][z], MASK_COLORS[m])
            ax.set_title(f"{r['label']}\n{col['title']}", fontsize=8)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
