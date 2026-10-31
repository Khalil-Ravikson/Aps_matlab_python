"""P7.15 — filtro em escada com Kaiser e comparação com o Exemplo 7.26."""

import numpy as np
import pytest

from dsp_av2.fir import analysis, ideal
from dsp_av2.specs import EXAMPLE_7_26_M, P7_15 as SPEC
from dsp_av2.validation import validate_fir_tolerance
from exercises import p7_15

PI = np.pi


@pytest.fixture(scope="module")
def search():
    return p7_15.search()


def test_kaiser_parameters():
    e = p7_15.estimate()
    assert e.params["As_design_dB"] == pytest.approx(60.0)
    assert e.params["beta"] == pytest.approx(5.65326, abs=1e-5)
    assert e.cutoffs == pytest.approx((0.35, 0.75))
    assert e.M == 74


def test_ideal_response_is_dta_staircase():
    """h_d = 0.5 h_LP(0.35π) + 0.5 h_LP(0.75π) (DTA §4, Resolução P7.15)."""
    e = p7_15.estimate()
    expected = 0.5 * ideal.ideal_lowpass(0.35 * PI, e.M) + 0.5 * ideal.ideal_lowpass(0.75 * PI, e.M)
    np.testing.assert_allclose(e.hd, expected, atol=1e-15)


def test_minimum_length_is_72(search):
    assert search.M_min == 72
    assert all(not ok for _, ok in search.tried[:-1])


def test_minimum_design_meets_specification_on_very_dense_grid(search):
    report = validate_fir_tolerance(search.best, n_points=200001)
    assert report.passed, str(report)


def test_remez_reproduces_example_7_26():
    """Equiripple independente: M mínimo = 49, como no livro (decisão D6)."""
    r = p7_15.remez_search()
    assert r.M_min == EXAMPLE_7_26_M == 49
    by_M = {M: a for M, _, a in r.tried}
    # Valores do livro: 49 → 60.61 dB, 50 → 60.03 dB (aqui medidos em grade densa)
    assert by_M[49] == pytest.approx(60.61, abs=0.2)


def test_remez_weights():
    assert p7_15.remez_weights() == pytest.approx([0.1, 0.2, 1.0])


def test_kaiser_is_about_47_percent_longer(search):
    assert search.M_min / EXAMPLE_7_26_M == pytest.approx(72 / 49)


def test_figure_has_magnitude_db(search):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_15.plot(search.best).axes]
    assert any("dB" in t for t in titles)
