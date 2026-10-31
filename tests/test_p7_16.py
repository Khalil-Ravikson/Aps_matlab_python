"""P7.16 — janela fixa de menor comprimento (decisão D5: critério medido)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import analysis
from dsp_av2.specs import P7_16 as SPEC
from dsp_av2.validation import validate_fir_db
from exercises import p7_16


@pytest.fixture(scope="module")
def results():
    return {r.window: r for r in p7_16.evaluate_windows()}


def test_table_lengths(results):
    """Critério da Tabela 7.1 (o do DTA): Hann 63 é o menor."""
    assert {k: r.M_table for k, r in results.items()} == {"hann": 63, "hamming": 67, "blackman": 111}
    assert all(r.table_design_passes for r in results.values())


def test_measured_minimum_lengths(results):
    assert {k: r.M_min for k, r in results.items()} == {"hann": 62, "hamming": 61, "blackman": 82}
    for r in results.values():
        assert all(not ok for _, ok in r.search.tried[:-1])
        assert len({M % 2 for M, _ in r.search.tried}) == 2  # pares e ímpares testados (D4)


def test_final_design_is_hamming_61():
    d = p7_16.design()
    assert (d.window, d.M) == ("hamming", 61)
    assert d.cutoffs == pytest.approx((0.35, 0.55))
    assert analysis.linear_phase_type(d.h) == 1


def test_meets_specification_on_very_dense_grid():
    d = p7_16.design()
    w = analysis.frequency_grid(SPEC.edges, 200001)
    rp, as_ = analysis.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, w)
    assert rp <= SPEC.rp_db and as_ >= SPEC.as_db
    assert validate_fir_db(d).passed


def test_cross_check_with_firwin():
    d = p7_16.design()
    ref = firwin(61, [0.35, 0.55], window="hamming", pass_zero=False, scale=False)
    np.testing.assert_allclose(d.h, ref, atol=1e-14)


def test_figure_has_requested_panels():
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_16.plot(p7_16.design()).axes]
    assert any("impulso" in t for t in titles)
    assert any("dB" in t for t in titles)
