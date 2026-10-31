"""P8.14 — Chebyshev-I racional (validação contra o enunciado e valores confirmados no plano §4.2)."""

import numpy as np
import pytest

from dsp_av2.iir import analysis
from dsp_av2.specs import P8_14 as SPEC
from exercises import p8_14


@pytest.fixture(scope="module")
def proto():
    return p8_14.design()


def test_order_and_ripple_factor(proto):
    assert proto.params["epsilon"] == pytest.approx(0.508847, abs=1e-6)
    assert proto.params["N_real"] == pytest.approx(7.403, abs=1e-3)
    assert proto.N == 8


def test_rational_coefficients(proto):
    b, a = p8_14.rational_form(proto)
    assert len(a) == 9 and a[0] == 1.0
    assert b[0] == pytest.approx(1.535333e6, rel=1e-6)
    assert a[-1] == pytest.approx(1.722672e6, rel=1e-6)
    assert b[0] / a[-1] == pytest.approx(1 / np.sqrt(1 + proto.params["epsilon"] ** 2))  # N par
    assert np.all(a > 0)


def test_meets_specification(proto):
    report = p8_14.validate(proto)
    assert report.passed, str(report)


def test_equiripple_passband(proto):
    """N = 8: N + 1 = 9 extremos em [0, Ωp], alternando entre 1 e 10^(−Rp/20)."""
    w = np.linspace(0, SPEC.wp, 200001)
    mag = np.abs(analysis.frequency_response(proto, w))
    assert mag.max() == pytest.approx(1.0, abs=1e-6)
    assert mag.min() == pytest.approx(10 ** (-SPEC.rp_db / 20), abs=1e-6)
    d = np.diff(mag)
    extrema_inside = np.sum(np.sign(d[1:]) != np.sign(d[:-1]))
    assert extrema_inside == 7  # + as duas extremidades (DC e Ωp) = 9


def test_figure_has_requested_panels(proto):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p8_14.plot(proto).axes]
    assert titles == ["Magnitude", "Log-magnitude", "Atraso de grupo", "Resposta ao impulso"]
