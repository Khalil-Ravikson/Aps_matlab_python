"""Mostra as saídas de texto de um notebook executado (para revisão rápida)."""
import sys

import nbformat

nb = nbformat.read(sys.argv[1], 4)
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.outputs:
            if o.output_type == "stream":
                print(o.text)
            elif o.output_type == "error":
                print("ERRO", o.ename, o.evalue)
            else:
                print("[", o.output_type, "]")
