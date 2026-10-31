#!/usr/bin/env bash
# Gera e executa notebooks: tools/run_notebook.sh p7_11 [p7_12 ...]  (sem argumentos: todos)
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python tools/build_notebooks.py "$@"
names=("$@"); [ ${#names[@]} -eq 0 ] && names=($(ls tools/nb/p*.py | xargs -n1 basename | sed 's/\.py$//'))
for n in "${names[@]}"; do
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace "notebooks/$n.ipynb" 2>&1 | grep -E "Writing|Error" || true
done
