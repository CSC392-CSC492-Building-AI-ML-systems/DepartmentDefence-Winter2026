"""Filesystem paths used by stack evaluation scripts."""

from pathlib import Path

STACK_EVAL_DIR = Path(__file__).resolve().parent
EVAL_DIR = STACK_EVAL_DIR.parent
REPO_ROOT = EVAL_DIR.parent
CASES_DIR = EVAL_DIR / "cases"
RUNS_DIR = EVAL_DIR / "runs"
