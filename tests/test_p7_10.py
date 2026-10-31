"""P7.10 — validação contra a especificação do enunciado (não contra o DTA)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.specs import P7_10 as SPEC
from exercises import p7_10


@pytest.fixture(scope="module")
def design():
    return p7_10.design()


def test_design_parameters(design):
    assert design.transition_width == pytest.approx(0.10)
    assert design.cutoffs == pytest.approx((0.35, 0.65))
    assert design.M_table == 67  # robusto ao ruído de ponto flutuante em 6.6π/0.1π
    assert design.M == 67
    assert design.alpha == 33


def test_linear_phase_type_1(design):
    """Rejeita-faixa exige Tipo 1 (H(π) ≠ 0)."""
    assert analysis.linear_phase_type(design.h) == 1


def test_meets_specification(design):
    report = p7_10.validate(design)
    assert report.passed, str(report)


def test_stopband_margin_is_narrow_and_worst_point_is_the_edge(design):
    """A_s ≈ 50.04 dB e o pior ponto é a própria borda 0.4π (fim da transição)."""
    w = analysis.frequency_grid(SPEC.edges, 200001)
    db = analysis.magnitude_db(analysis.frequency_response(design.h, w))
    stop = (w >= 0.4 * np.pi - 1e-12) & (w <= 0.6 * np.pi + 1e-12)
    worst = w[stop][np.argmax(db[stop])] / np.pi
    assert -db[stop].max() == pytest.approx(50.044, abs=1e-3)
    assert worst == pytest.approx(0.4, abs=1e-9) or worst == pytest.approx(0.6, abs=1e-9)


def test_table_length_is_also_the_minimum_odd_length():
    """Com M = 65 (próximo ímpar abaixo) a atenuação cai para ≈ 44.9 dB."""
    d = design_fixed_window(SPEC, "hamming", M=65)
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    _, as_ = analysis.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, w)
    assert as_ < SPEC.as_db


def test_cross_check_with_firwin(design):
    """firwin é proibido na solução, mas serve de verificação independente."""
    ref = firwin(67, [0.35, 0.65], window="hamming", pass_zero=True, scale=False)
    np.testing.assert_allclose(design.h, ref, atol=1e-14)


def test_figure_has_requested_panels(design):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_10.plot(design).axes]
    assert any("impulso" in t for t in titles)
    assert any("dB" in t for t in titles)
