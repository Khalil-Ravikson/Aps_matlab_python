"""Gera tests/golden/results.json: os resultados oficiais dos 12 exercícios.

Executado UMA vez a partir do commit validado (70c8e0c). O teste
tests/test_regression.py compara o núcleo atual com este arquivo, garantindo
que a aplicação (ou qualquer mudança futura) não altera os cálculos.
Regenerar só com justificativa documentada.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from exercises import (p7_09, p7_10, p7_11, p7_12, p7_13, p7_14, p7_15, p7_16,
                       p8_12, p8_13, p8_14, p8_15)

ROOT = Path(__file__).resolve().parents[1]


def checks(report):
    return {c.name: c.measured for c in report.checks} | {"passed": report.passed}


def fir(mod):
    d = mod.design()
    return {"window": d.window, "M": d.M, "cutoffs_pi": list(d.cutoffs),
            "params": {k: float(v) for k, v in d.params.items()},
            "h": d.h.tolist(), "validation": checks(mod.validate(d))}


def iir(mod, form):
    p = mod.design()
    out = {"N": p.N, "params": {k: float(v) for k, v in p.params.items()},
           "poles": [[float(z.real), float(z.imag)] for z in sorted(p.p, key=lambda q: (q.imag, q.real))],
           "k": float(p.k), "validation": checks(mod.validate(p))}
    if form == "rational":
        b, a = mod.rational_form(p)
        out |= {"B": b.tolist(), "A": a.tolist()}
    else:
        c = mod.cascade_form(p)
        out |= {"gain": c.gain, "sections": [{"num": s.num.tolist(), "den": s.den.tolist()} for s in c.sections]}
    return out


def main():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    data = {
        "_meta": {"commit": commit, "numpy": np.__version__},
        "P7.9": fir(p7_09), "P7.10": fir(p7_10), "P7.11": fir(p7_11), "P7.12": fir(p7_12),
        "P7.13": fir(p7_13) | {"M_estimate": p7_13.estimate().M},
        "P7.14": fir(p7_14) | {"M_estimate": p7_14.estimate().M},
        "P7.15": fir(p7_15) | {"M_estimate": p7_15.estimate().M, "remez_M_min": p7_15.remez_search().M_min},
        "P7.16": fir(p7_16),
        "P8.12": iir(p8_12, "rational"), "P8.13": iir(p8_13, "cascade"),
        "P8.14": iir(p8_14, "rational"), "P8.15": iir(p8_15, "cascade"),
    }
    out = ROOT / "tests" / "golden" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print("gravado:", out)


if __name__ == "__main__":
    main()
