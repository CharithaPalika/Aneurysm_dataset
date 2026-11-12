"""Fixed folder and file names, in one place.

Every path in the code is built as  <project root> / <name from here or from a YAML file>.
The project root is the folder that contains src/aneurysm/ and configs/
(scripts take the parent of scripts/, notebooks search upwards from their folder).

    <root>/configs/{data,train,eval}/*.yaml     settings
    <root>/LeFusion/LeFusion/ddpm/              LeFusion code (imported, unchanged)
    <dataset root>/...                          read-only raw data: configs/data/sources/<name>.yaml
    <root>/<work_subdir>/<dataset>/             preprocessed data (splits.csv + patches/)
    <root>/<output_dir>/<run_name>/             one training run (layout: RunPaths)
"""
from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path("configs")
SOURCES_CONFIG_DIR = CONFIG_DIR / "data" / "sources"         # one YAML per dataset: <name>.yaml
LEFUSION_CODE_DIR = Path("LeFusion") / "LeFusion"          # folder that contains ddpm/
ROOT_MARKERS = (Path("src") / "aneurysm", CONFIG_DIR)      # how the project root is recognised


class RunPaths:
    """Layout of one run folder (<output_dir>/<run_name>/), used by training and evaluation."""

    def __init__(self, run_dir: str | Path):
        self.root = Path(run_dir)
        self.checkpoints = self.root / "checkpoints"
        self.samples = self.root / "samples"
        self.eval = self.root / "eval"
        self.log = self.root / "train_log.txt"
        self.metrics = self.root / "metrics.csv"
        self.config = self.root / "config.yaml"

    def checkpoint(self, name: str) -> Path:
        """'best' / 'latest' / 'step_001000' (with or without .pt) -> checkpoints/<name>.pt"""
        return self.checkpoints / (name if name.endswith(".pt") else f"{name}.pt")

    def eval_dir(self, checkpoint: str, split: str) -> Path:
        return self.eval / f"{Path(checkpoint).stem}_{split}"

    def is_run(self) -> bool:
        return self.checkpoints.is_dir()
