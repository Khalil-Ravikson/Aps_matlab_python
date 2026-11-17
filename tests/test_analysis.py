"""Seções de análise (Etapa 5): afirmações do texto conferidas com os cálculos."""

import json
from pathlib import Path

import numpy as np
import pytest

from content import analysis as A

GOLDEN = json.loads((Path(__file__).parent / "golden" / "results.json").read_text())


@pytest.fixture(scope="module")
def comp():
    return A.comparison_data()


def test_chebyshev_never_needs_higher_order_than_butterworth(comp):
    for ex, ratio, nb, nc, ne in comp["orders"]:
        assert nc <= nb and ne <= nc, ex


def test_orders_match_official_results(comp):
    by = {r[0]: r for r in comp["orders"]}
    assert by["P8.12"][2] == GOLDEN["P8.12"]["N"] and by["P8.13"][2] == GOLDEN["P8.13"]["N"]
    assert by["P8.14"][3] == GOLDEN["P8.14"]["N"] and by["P8.15"][3] == GOLDEN["P8.15"]["N"]


def test_p8_14_largest_chebyshev_advantage_is_narrowest_transition(comp):
    gain = {r[0]: r[2] - r[3] for r in comp["orders"]}
    ratio = {r[0]: r[1] for r in comp["orders"]}
    assert max(gain, key=gain.get) == min(ratio, key=ratio.get) == "P8.14"


def test_digital_iir_illustration_meets_p7_9_spec(comp):
    assert all(r["ok"] for r in comp["digital"])
    best = min(comp["digital"], key=lambda r: r["mults"])
    assert best["mults"] < comp["fir"]["mults_sym"]          # "IIR com menos multiplicações"
    assert best["gd_max"] - best["gd_min"] > 1               # atraso de grupo varia


def test_fir_group_delay_is_constant_alpha(comp):
    f = comp["fir"]
    assert f["gd_min"] == pytest.approx(f["alpha"], abs=1e-6) and f["gd_max"] == pytest.approx(f["alpha"], abs=1e-6)


def test_p7_15_is_the_most_expensive_fir(comp):
    assert max(comp["fir_rows"], key=lambda r: r[3])[0] == "P7.15"


def test_validation_all_pass_and_match_golden():
    v = A.validation_data()
    assert len(v["rows"]) == 12
    assert all(r["passed"] and r["golden"] for r in v["rows"])
    assert all(r["min_rel_margin"] > 0 for r in v["rows"])


def test_validation_lists_every_errata_item():
    from content.errata import items
    text = A.validation_blocks()[3].text
    for i in items():
        assert f"| {i} |" in text


def test_blocks_have_no_unfilled_markers_or_typed_failures():
    for blocks in (A.comparison_blocks(), A.validation_blocks(), A.methodology_blocks()):
        for b in blocks:
            assert "[[" not in b.text
            if b.kind == "fig":
                for y in b.data.get("curves", {}).values():
                    assert np.all(np.isfinite(np.asarray(y)[np.isfinite(np.asarray(y))]))
