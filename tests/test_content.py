"""Consistência entre conteúdo didático, cálculos, resultados oficiais e errata.

Garante que:
- todo marcador [[...]] do texto tem valor e nenhum fica sem preenchimento;
- nenhum resultado numérico é digitado no texto (vem sempre do cálculo);
- os valores do conteúdo coincidem com os resultados oficiais (golden);
- as afirmações qualitativas do texto (ex.: "a estimativa falha") são verdadeiras;
- cada correção da errata citada existe e concorda com o cálculo.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from content import registry
from content.errata import items as errata_items
from content.model import fill, markers, table_cell

GOLDEN = json.loads((Path(__file__).parent / "golden" / "results.json").read_text())
WRITTEN = [e.id for e in registry.EXERCISES if registry.has_content(e.id)]


@pytest.fixture(scope="module", params=WRITTEN)
def ex(request):
    mod = registry.load(request.param)
    return mod, mod.CONTENT, mod.context()


def test_pilots_are_written():
    assert {"p7_13", "p8_13"} <= set(WRITTEN)


def test_every_marker_has_a_value(ex):
    mod, C, ctx = ex
    for text in C.texts():
        out = fill(text, ctx)
        assert "[[" not in out and "]]" not in out, text[:80]


def test_all_ten_sections_present(ex):
    _, C, _ = ex
    assert C.asked and C.data_rows and C.theory and C.equations and C.steps
    assert C.implementation and C.code_roles and C.figures and C.verification and C.conclusion
    assert C.highlights


def test_result_values_are_never_typed(ex):
    """Os resultados aparecem no texto só via marcador, nunca digitados."""
    _, C, ctx = ex
    raw = "\n".join(C.texts())
    for key in C.result_keys:
        val = ctx[key]
        candidates = {str(val)} if isinstance(val, int) else {f"{val:.2f}", f"{val:.4f}"}
        for c in candidates:
            assert not re.search(rf"(?<![\w.\[]){re.escape(c)}(?![\w])", raw), f"{key}={c} digitado no texto"


def test_errata_items_exist_and_match_exercise(ex):
    _, C, _ = ex
    for eid in C.errata:
        item = errata_items()[eid]
        assert item.exercise == C.number


def test_table_cell_never_contains_bare_pipe():
    out = table_cell(r"tol: $0.975 \le |H| \le 1$ e ||H| − 1|")
    assert "|" not in out


# ── P7.13 ───────────────────────────────────────────────────────────────────

def test_p7_13_values_match_golden():
    from content import p7_13
    ctx, g = p7_13.context(), GOLDEN["P7.13"]
    assert ctx["M"] == g["M"] and ctx["M_est"] == g["M_estimate"]
    assert ctx["beta"] == pytest.approx(g["params"]["beta"], rel=1e-12)
    assert ctx["As"] == pytest.approx(g["params"]["As_design_dB"], rel=1e-12)


def test_p7_13_narrative_claims_hold():
    """O texto afirma que a estimativa falha e que a busca termina no primeiro que atende."""
    from content import p7_13
    ctx, o = p7_13.context(), p7_13.objects()
    assert ctx["est_dev_stop"] > ctx["d2"]                      # passo 8: "não atende"
    assert ctx["M_est"] % 2 == 1 and ctx["M"] % 2 == 1          # "só ímpares"
    assert o["search"].tried[-1] == (ctx["M"], True)
    assert all(not ok for _, ok in o["search"].tried[:-1])
    assert ctx["delta"] == min(ctx["d1"], ctx["d2"])
    assert 21 < ctx["As"] < 50                                  # passo 4: "segundo ramo"
    assert ctx["dev_pass"] < ctx["d1"] / 2                      # "folga desperdiçada"


def test_p7_13_errata_E1_agrees_with_computation():
    from content import p7_13
    ctx = p7_13.context()
    e1 = errata_items()["E1"]
    assert f"M = {ctx['M']}" in e1.correction
    assert f"M = {ctx['M_est']}" in e1.correction
    assert str(ctx["M_est"]) in e1.claim


# ── P8.13 ───────────────────────────────────────────────────────────────────

def test_p8_13_values_match_golden():
    from content import p8_13
    ctx, g = p8_13.context(), GOLDEN["P8.13"]
    assert ctx["N"] == g["N"]
    assert ctx["Omega_c"] == pytest.approx(g["params"]["Omega_c"], rel=1e-12)
    assert ctx["gain"] == pytest.approx(g["gain"], rel=1e-12)


def test_p8_13_narrative_claims_hold():
    from content import p8_13
    ctx = p8_13.context()
    assert ctx["N"] % 2 == 1                                    # "N ímpar → polo real"
    assert ctx["att_lower"] < ctx["As"] <= ctx["att_ws"]        # minimalidade
    assert ctx["margin_ws"] < 0.1                               # "pouca folga"
    assert ctx["loss_wp"] == pytest.approx(ctx["Rp"], abs=1e-9)  # "igualdade por construção"
    assert ctx["N_real"] > ctx["N"] - 0.05                      # "muito perto do inteiro"


def test_p8_13_errata_E6_agrees_with_computation():
    from content import p8_13
    ctx, o = p8_13.context(), p8_13.objects()
    wc = ctx["Omega_c"]
    thetas = sorted(np.degrees(np.arccos(s.den[1] / (2 * wc))) for s in o["cascade"].sections if s.order == 2)
    np.testing.assert_allclose(thetas, [k * 180 / ctx["N"] for k in range(1, ctx["n_pairs"] + 1)], atol=1e-9)
    assert rf"k\pi/{ctx['N']}" in errata_items()["E6"].correction
