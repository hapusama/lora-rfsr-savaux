#!/usr/bin/env python3
"""Compatibility wrapper for the standalone UniChirp evaluator.

The old version compared UniChirp with retired phase-line island experiments.
Those experiments are no longer part of the baseline package; use
``evaluate_unichirp.py`` for the actual implementation.
"""

from __future__ import annotations

try:
    from .evaluate_unichirp import *  # noqa: F401,F403
    from .evaluate_unichirp import main
except ImportError:  # pragma: no cover - direct script execution fallback
    from evaluate_unichirp import *  # type: ignore # noqa: F401,F403
    from evaluate_unichirp import main  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main())
