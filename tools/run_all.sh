#!/usr/bin/env bash
# Regenera todos os resultados: testes, scripts dos exercícios (results/) e notebooks.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest -q
for f in exercises/p*.py; do
  m=$(basename "$f" .py)
  echo "== $m"; .venv/bin/python -m "exercises.$m" > "results/${m}/stdout.txt"
done
tools/run_notebook.sh
