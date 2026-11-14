"""Regressão: o núcleo matemático continua produzindo os resultados oficiais.

Referência: tests/golden/results.json, gerado por ``python -m tools.make_golden``
a partir do commit validado (70c8e0c). Se este teste falhar, algum cálculo
mudou: investigar antes de regenerar o golden.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from exercises import (p7_09, p7_10, p7_11, p7_12, p7_13, p7_14, p7_15, p7_16,
                       p8_12, p8_13, p8_14, p8_15)
from tools.make_golden import fir, iir

GOLDEN = json.loads((Path(__file__).parent / "golden" / "results.json").read_text())
RTOL = 1e-9

FIR = {"P7.9": p7_09, "P7.10": p7_10, "P7.11": p7_11, "P7.12": p7_12,
       "P7.13": p7_13, "P7.14": p7_14, "P7.15": p7_15, "P7.16": p7_16}
IIR = {"P8.12": (p8_12, "rational"), "P8.13": (p8_13, "cascade"),
       "P8.14": (p8_14, "rational"), "P8.15": (p8_15, "cascade")}


def _compare(got, ref, path=""):
    if isinstance(ref, dict):
        assert set(got) == set(ref), f"{path}: chaves diferentes"
        for k in ref:
            _compare(got[k], ref[k], f"{path}.{k}")
    elif isinstance(ref, list) and ref and isinstance(ref[0], dict):
        assert len(got) == len(ref), path
        for i, (g, r) in enumerate(zip(got, ref)):
            _compare(g, r, f"{path}[{i}]")
    elif isinstance(ref, list):
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(ref, float), rtol=RTOL, atol=1e-12,
                                   err_msg=path)
    elif isinstance(ref, (bool, str)):
        assert got == ref, path
    else:
        assert got == pytest.approx(ref, rel=RTOL, abs=1e-12), path


@pytest.mark.parametrize("ex", FIR)
def test_fir_matches_golden(ex):
    got = fir(FIR[ex])
    _compare(got, {k: v for k, v in GOLDEN[ex].items() if k in got}, ex)


@pytest.mark.parametrize("ex", IIR)
def test_iir_matches_golden(ex):
    mod, form = IIR[ex]
    _compare(iir(mod, form), GOLDEN[ex], ex)


def test_kaiser_estimates_and_remez_match_golden():
    assert p7_13.estimate().M == GOLDEN["P7.13"]["M_estimate"]
    assert p7_14.estimate().M == GOLDEN["P7.14"]["M_estimate"]
    assert p7_15.estimate().M == GOLDEN["P7.15"]["M_estimate"]
    assert p7_15.remez_search().M_min == GOLDEN["P7.15"]["remez_M_min"]


def test_all_official_results_pass():
    for ex, data in GOLDEN.items():
        if not ex.startswith("_"):
            assert data["validation"]["passed"], ex
