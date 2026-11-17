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


def test_all_twelve_are_written():
    assert set(WRITTEN) == {e.id for e in registry.EXERCISES}


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


# ── Demais exercícios: afirmações do texto e errata ─────────────────────────

def _c(ex_id):
    return registry.load(ex_id).context()


def test_p7_09_claims():
    c = _c("p7_09")
    assert c["dw"] == pytest.approx(c["t1"]) and c["t1"] < c["t2"]        # "vale a menor transição"
    assert c["M"] % 2 == 1 and c["M"] == GOLDEN["P7.9"]["M"]
    assert abs(c["as_"] - c["As_win"]) < 0.5                             # "coincide com o nominal"
    assert "44.05" in errata_items()["V5"].correction and f"{c['as_']:.2f}" == "44.05"


def test_p7_10_claims():
    c, o = _c("p7_10"), registry.load("p7_10").objects()
    assert c["worst_w"] == pytest.approx(0.4) or c["worst_w"] == pytest.approx(0.6)   # pior ponto = borda
    assert 0 < c["as_margin"] < 0.1
    assert c["odd_zero"] < 1e-12                                          # coeficientes nulos
    assert c["as_prev"] < c["As_spec"] and c["Mprev"] == c["M"] - 2      # mínimo ímpar
    assert c["as_2049"] > c["as_"] + 0.5                                  # grade sem bordas superestima
    assert f"{c['as_']:.2f} dB" in errata_items()["V1"].correction


def test_p7_11_claims():
    c = _c("p7_11")
    assert c["M_first"] < c["M"] and c["type_first"] == 2 and c["as_first"] > c["as_"]
    assert c["worst_w"] == pytest.approx(0.6) or c["worst_w"] == pytest.approx(0.3)
    assert f"{c['as_']:.2f} dB" in errata_items()["V2"].correction


def test_p7_12_claims():
    c, o = _c("p7_12"), registry.load("p7_12").objects()
    assert o["final"].window == "blackman"
    assert all(rp > c["Rp_spec"] for _, rp, _ in o["hamming_rows"])      # Hamming nunca atende Rp
    assert c["M_table"] % 2 == 0 and c["M"] == c["M_table"] + 1           # paridade
    assert c["M_bla_first"] < c["M"] and c["rp_bla_first_margin"] < 1e-3
    assert f"**{c['M_table']}**" in errata_items()["V3"].correction


def test_p7_14_claims():
    c = _c("p7_14")
    assert c["M"] % 2 == 0 and c["type"] == 2 and c["M"] == c["M_est"] == GOLDEN["P7.14"]["M"]
    assert abs(c["Hr_pi"]) < 1e-12
    assert c["dev_hi"] > c["dev_lo"]                                      # faixa crítica: superior
    assert c["dta_passes"]                                                # "47 também atende"
    assert f"M = {c['M']}" in errata_items()["E2"].correction


def test_p7_15_claims():
    c = _c("p7_15")
    assert c["M"] == GOLDEN["P7.15"]["M"] and c["M_pm"] == c["M_book"] == GOLDEN["P7.15"]["remez_M_min"]
    devs = [c["dev1"], c["dev2"], c["dev3"]]
    assert max(devs) / min(devs) < 1.2                                     # ondulações ~iguais
    assert c["best_M"] > c["M_pm"]                                        # ajuste de A_s não basta
    assert f"M = {c['M']}" in errata_items()["E3"].correction
    assert f"{c['pct']:.0f}%" in errata_items()["E4"].correction
    assert f"{c['pct_dta_red']:.0f}%" in errata_items()["E4"].correction


def test_p7_16_claims():
    c = _c("p7_16")
    assert c["win"] == "Hamming" and c["M_min_ham"] < c["M_min_hann"] < c["M_tab_hann"]
    assert c["tab_best"] == "Hann"
    assert f"Hamming, M = {c['M']}" in errata_items()["E5"].correction


def test_p8_12_claims():
    c = _c("p8_12")
    assert c["N"] % 2 == 0                                                # "N par: sem polo real"
    assert c["B"] == pytest.approx(c["a0"]) and c["H0"] == pytest.approx(1.0)
    assert c["att_lower"] < c["As"] <= c["att_ws"]
    assert c["closest_deg"] == pytest.approx(105.0)


def test_p8_14_claims():
    c = _c("p8_14")
    assert c["extrema"] == c["N"] + 1
    assert c["dc"] == pytest.approx(1 / np.sqrt(1 + c["eps"] ** 2))
    assert c["Nb"] == 2 * c["N"]                                          # "reduz a ordem pela metade"
    assert c["att_lower"] < c["As"] <= c["att_ws"]
    assert c["ab_ratio"] < 0.25                                           # "elipse muito achatada"


def test_p8_15_claims():
    from dsp_av2.iir import analysis as ia
    c, o = _c("p8_15"), registry.load("p8_15").objects()
    assert c["N"] % 2 == 0 and c["n_sec"] == c["N"] // 2
    assert c["gain"] == pytest.approx(1 / np.sqrt(1 + c["eps"] ** 2))
    assert c["N_real"] - (c["N"] - 1) < 0.1                               # "logo acima de um inteiro"
    assert c["att_lower"] < c["As"] <= c["att_ws"]
    w = np.linspace(0, o["spec"].wp, 200001)
    d = np.diff(np.abs(ia.frequency_response(o["proto"], w)))
    assert int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) + 2 == c["Np1"]  # N+1 extremos


def test_every_exercise_errata_item_is_shown_on_its_page():
    """Cada item da errata ligado a um exercício aparece na página desse exercício."""
    by_number = {registry.load(e.id).CONTENT.number: set(registry.load(e.id).CONTENT.errata) for e in registry.EXERCISES}
    for item in errata_items().values():
        if item.exercise in by_number:
            assert item.id in by_number[item.exercise], f"{item.id} ({item.exercise}) não citado"
