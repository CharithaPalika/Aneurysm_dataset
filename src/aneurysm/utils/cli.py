"""Command-line helpers shared by the scripts in scripts/ (run_preprocess, run_train, run_eval)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def _read_value(text: str):
    """Read one --set value like YAML: 32 -> int, true -> bool, [1,2,4] -> list, null -> None.
    Plain YAML reads '1e-4' as text, so numbers like that are converted to float here."""
    value = yaml.safe_load(text) if text.strip() else None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_overrides(pairs: list[str] | None, option: str = "--set") -> dict:
    """['lr=2e-4', 'dim_mults=[1,2,4,4]', 'run_dir=null'] -> {'lr': 0.0002, 'dim_mults': [1, 2, 4, 4], ...}"""
    out = {}
    for pair in pairs or []:
        key, sep, text = pair.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"{option} expects KEY=VALUE, got '{pair}'")
        out[key.strip()] = _read_value(text)
    return out


def add_config_arg(parser: argparse.ArgumentParser, flag: str, default: Path, what: str) -> None:
    parser.add_argument(flag, default=None, metavar="PATH",
                        help=f"{what} YAML file (default: {Path(default).as_posix()}). "
                             f"Relative paths work from the current folder or the project folder.")


def add_set_arg(parser: argparse.ArgumentParser, flag: str, what: str) -> None:
    parser.add_argument(flag, nargs="+", default=[], metavar="KEY=VALUE",
                        help=f"override {what} values from the file, e.g. {flag} run_name=test lr=2e-4. "
                             f"In PowerShell quote lists: \"dim_mults=[1,2,4,4]\"")


def print_header(title: str, configs: dict[str, Path], overrides: dict[str, dict]) -> None:
    """Print which config files and overrides a script uses (also ends up in the terminal log).
    Multi-GPU (torchrun): only the first process prints."""
    from .env import describe_environment

    if not is_main_process():
        return

    print(f"===== {title} =====")
    for name, path in configs.items():
        print(f"{name:12s}: {Path(path).as_posix()}")
    for name, values in overrides.items():
        if values:
            print(f"{name:12s}: {values}")
    env = describe_environment()
    print("environment : " + ", ".join(f"{k}={v}" for k, v in env.items() if k != "platform"))
    print()


def is_main_process() -> bool:
    """False for the extra processes of a multi-GPU run (torchrun sets RANK)."""
    return int(os.environ.get("RANK", "0")) == 0
