"""Entry point for the structured PolicyRAG evaluation stack.

This file intentionally stays small. Core logic now lives under:
- evaluation/stack_eval/runner.py
- evaluation/stack_eval/*.py helper modules
"""

from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from stack_eval.runner import main  # noqa: E402


if __name__ == "__main__":
    main()
