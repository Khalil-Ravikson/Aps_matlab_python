"""Gravação de resultados (métricas JSON e figuras) em ``results/<exercício>/``."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def exercise_dir(exercise: str) -> Path:
    """``"P7.9"`` → ``results/p7_09/`` (criado se não existir)."""
    chapter, number = exercise.lstrip("Pp").split(".")
    d = RESULTS_DIR / f"p{chapter}_{int(number):02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"não serializável: {type(obj)}")


def save_metrics(exercise: str, metrics: dict) -> Path:
    path = exercise_dir(exercise) / "metrics.json"
    payload = metrics | {"environment": environment()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable) + "\n")
    return path


def save_figure(exercise: str, fig, name: str) -> Path:
    path = exercise_dir(exercise) / f"{name}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    return path
