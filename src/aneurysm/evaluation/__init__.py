"""Evaluation of a trained run: training curves, cases 1-4, histogram control, copy check, full scan."""
from .evaluate import CASES, resolve_run_dir, run_evaluation

__all__ = ["run_evaluation", "resolve_run_dir", "CASES"]
