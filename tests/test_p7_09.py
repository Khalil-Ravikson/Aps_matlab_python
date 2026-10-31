"""P7.9 — validação contra a especificação do enunciado (não contra o DTA)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import analysis
from dsp_av2.specs import P7_9 as SPEC
from exercises import p7_09


@pytest.fixture(scope="module")
def design():
    return p7_09.design()


def test_design_parameters(design):
    assert design.transition_width == pytest.approx(0.15)
    assert design.cutoffs == pytest.approx((0.275, 0.65))
    assert design.M == 43
    assert design.alpha == 21


def test_linear_phase_type_1(design):
    assert analysis.linear_phase_type(design.h) == 1


def test_meets_specification(design):
    report = p7_09.validate(design)
    assert report.passed, str(report)


def test_spec_values_on_dense_grid(design):
    w = analysis.frequency_grid(SPEC.edges, 200001)
    rp, as_ = analysis.measure_rp_as(design.h, SPEC.passbands, SPEC.stopbands, w)
    assert rp <= SPEC.rp_db
    assert as_ >= SPEC.as_db


def test_cross_check_with_firwin(design):
    """firwin é proibido na solução, mas serve de verificação independente."""
    ref = firwin(43, [0.275, 0.65], window="hann", pass_zero=False, scale=False)
    np.testing.assert_allclose(design.h, ref, atol=1e-14)


def test_figure_has_requested_panels(design):
    import matplotlib
    matplotlib.use("Agg")
    fig = p7_09.plot(design)
    titles = [ax.get_title(loc="left") for ax in fig.axes]
    assert any("impulso" in t for t in titles)
    assert any("dB" in t for t in titles)
