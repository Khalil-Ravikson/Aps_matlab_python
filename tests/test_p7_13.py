"""P7.13 — Kaiser, rejeita-faixa de comprimento mínimo (validação contra o enunciado)."""

import numpy as np
import pytest

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_kaiser
from dsp_av2.specs import P7_13 as SPEC
from dsp_av2.validation import validate_fir_tolerance
from exercises import p7_13


@pytest.fixture(scope="module")
def search():
    return p7_13.search()


def test_kaiser_parameters():
    e = p7_13.estimate()
    assert e.params["As_design_dB"] == pytest.approx(46.0206, abs=1e-4)
    assert e.params["beta"] == pytest.approx(4.0909, abs=1e-4)
    assert e.transition_width == pytest.approx(0.1)
    assert e.cutoffs == pytest.approx((0.3, 0.7))
    assert e.M == 55


def test_kaiser_estimate_fails_the_stopband_tolerance():
    """Correção do DTA: com M = 55, max|H| na rejeição ≈ 0.0059 > 0.005."""
    report = validate_fir_tolerance(p7_13.estimate())
    assert not report.passed
    assert report.checks[1].measured == pytest.approx(0.0059, abs=1e-4)


def test_minimum_length_is_57(search):
    assert search.M_min == 57
    assert all(not ok for M, ok in search.tried[:-1])
    assert all(M % 2 == 1 for M, _ in search.tried)  # rejeita-faixa: só Tipo 1


def test_minimum_design_meets_specification_on_very_dense_grid(search):
    d = search.best
    assert analysis.linear_phase_type(d.h) == 1
    report = validate_fir_tolerance(d, n_points=200001)
    assert report.passed, str(report)


def test_beta_is_kept_fixed_during_search(search):
    assert search.best.params["beta"] == pytest.approx(p7_13.estimate().params["beta"])
    np.testing.assert_allclose(search.best.w, np.kaiser(57, search.best.params["beta"]), atol=1e-12)


def test_even_lengths_are_not_admissible():
    with pytest.raises(ValueError, match="M ímpar"):
        design_kaiser(SPEC, M=56)


def test_figure_has_requested_panels(search):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_13.plot(search.best).axes]
    assert any("amplitude" in t for t in titles)
    assert any("dB" in t for t in titles)
