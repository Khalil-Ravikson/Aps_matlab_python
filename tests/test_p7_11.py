"""P7.11 — validação contra a especificação do enunciado (não contra o DTA)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.specs import P7_11 as SPEC
from exercises import p7_11


@pytest.fixture(scope="module")
def design():
    return p7_11.design()


def test_design_parameters(design):
    assert design.transition_width == pytest.approx(0.10)
    assert design.cutoffs == pytest.approx((0.35, 0.55))
    assert design.M == 67
    assert design.alpha == 33


def test_linear_phase_type_1(design):
    assert analysis.linear_phase_type(design.h) == 1


def test_meets_specification(design):
    report = p7_11.validate(design)
    assert report.passed, str(report)


def test_even_length_66_also_meets_specification():
    """Observação: passa-faixa admite Tipo 2, e M = 66 também atende (≈ 52.0 dB)."""
    d = design_fixed_window(SPEC, "hamming", M=66)
    assert analysis.linear_phase_type(d.h) == 2
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    rp, as_ = analysis.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, w)
    assert rp <= SPEC.rp_db and as_ >= SPEC.as_db
    d65 = design_fixed_window(SPEC, "hamming", M=65)
    assert analysis.measure_rp_as(d65.h, SPEC.passbands, SPEC.stopbands, w)[1] < SPEC.as_db


def test_cross_check_with_firwin(design):
    ref = firwin(67, [0.35, 0.55], window="hamming", pass_zero=False, scale=False)
    np.testing.assert_allclose(design.h, ref, atol=1e-14)


def test_figure_has_requested_panels(design):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_11.plot(design).axes]
    assert any("impulso" in t for t in titles)
    assert any("dB" in t for t in titles)
