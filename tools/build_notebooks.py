"""Gera os notebooks em notebooks/ a partir de definições em Python.

Cada notebook é definido em ``tools/nb/<exercício>.py`` (função ``cells()``),
o que mantém o conteúdo revisável em diff. Uso:

    python tools/build_notebooks.py            # todos
    python tools/build_notebooks.py p7_11      # apenas um

Depois, execute com ``jupyter nbconvert --to notebook --execute --inplace``.
"""

import importlib
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT), str(ROOT / "src")]


def available():
    return sorted(p.stem for p in (ROOT / "tools" / "nb").glob("p[0-9]*.py"))


def build(name: str) -> Path:
    module = importlib.import_module(f"nb.{name}")
    nb = nbf.v4.new_notebook()
    nb.cells = module.cells()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "notebooks" / f"{name}.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    return out


def main():
    for name in sys.argv[1:] or available():
        print("gerado:", build(name))


if __name__ == "__main__":
    main()
