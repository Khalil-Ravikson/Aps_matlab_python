"""P7.14 — Kaiser, passa-faixa de comprimento mínimo (validação contra o enunciado)."""

import numpy as np
import pytest

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_kaiser
from dsp_av2.specs import P7_14 as SPEC
from dsp_av2.validation import validate_fir_tolerance
from exercises import p7_14


@pytest.fixture(scope="module")
def search():
    return p7_14.search()


def test_kaiser_parameters():
    e = p7_14.estimate()
    assert e.params["As_design_dB"] == pytest.approx(40.0)
    assert e.params["beta"] == pytest.approx(3.3953, abs=1e-4)
    assert e.cutoffs == pytest.approx((0.3, 0.7))
    assert e.M == 46  # a estimativa não é forçada a ímpar no passa-faixa (D4)


def test_minimum_length_is_46_type_2(search):
    assert search.M_min == 46
    assert analysis.linear_phase_type(search.best.h) == 2
    assert [M for M, _ in search.tried][-4:] == [43, 44, 45, 46]  # pares e ímpares testados
    assert all(not ok for _, ok in search.tried[:-1])


def test_minimum_design_meets_specification_on_very_dense_grid(search):
    report = validate_fir_tolerance(search.best, n_points=200001)
    assert report.passed, str(report)


def test_type_2_has_zero_at_pi(search):
    Hr = analysis.amplitude_response(search.best.h, np.array([np.pi]))
    assert Hr[0] == pytest.approx(0, abs=1e-12)


def test_dta_length_47_also_meets_but_is_not_minimum():
    assert validate_fir_tolerance(design_kaiser(SPEC, M=47)).passed


def test_figure_has_requested_panels(search):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_14.plot(search.best).axes]
    assert any("amplitude" in t for t in titles)
    assert any("dB" in t for t in titles)
