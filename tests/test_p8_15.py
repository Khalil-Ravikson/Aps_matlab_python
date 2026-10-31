"""P8.15 — Chebyshev-I em cascata (validação contra o enunciado e valores confirmados no plano §4.2)."""

import numpy as np
import pytest
from scipy import signal

from dsp_av2.specs import P8_15 as SPEC
from exercises import p8_15


@pytest.fixture(scope="module")
def proto():
    return p8_15.design()


def test_order(proto):
    assert proto.params["epsilon"] == pytest.approx(0.349311, abs=1e-6)
    assert proto.params["N_real"] == pytest.approx(3.021, abs=1e-3)
    assert proto.N == 4


def test_poles_match_dta(proto):
    expected = np.array([-4407.10 + 25541.22j, -10639.69 + 10579.52j])
    upper = np.sort_complex(proto.p[proto.p.imag > 0])
    np.testing.assert_allclose(np.sort_complex(upper), np.sort_complex(expected), atol=0.01)


def test_cascade_matches_dta(proto):
    c = p8_15.cascade_form(proto)
    assert c.gain == pytest.approx(0.94406, abs=1e-5)
    dens = sorted((s.den for s in c.sections), key=lambda d: d[1])
    np.testing.assert_allclose(dens[0], [1, 8814.21, 6.7178e8], rtol=1e-4)
    np.testing.assert_allclose(dens[1], [1, 21279.38, 2.2513e8], rtol=1e-4)
    for s in c.sections:
        assert s.num[0] == pytest.approx(s.den[-1])  # ganho DC unitário por seção


def test_cascade_equals_zpk_response(proto):
    c = p8_15.cascade_form(proto)
    w = np.geomspace(1e2, 1e7, 400)
    _, H = signal.freqs_zpk(proto.z, proto.p, proto.k, worN=w)
    Hc = c.gain * np.ones_like(w, dtype=complex)
    for s in c.sections:
        Hc *= signal.freqs(s.num, s.den, worN=w)[1]
    np.testing.assert_allclose(Hc, H, rtol=1e-9, atol=1e-30)


def test_meets_specification(proto):
    report = p8_15.validate(proto)
    assert report.passed, str(report)


def test_figure_has_requested_panels(proto):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p8_15.plot(proto).axes]
    assert titles == ["Magnitude", "Log-magnitude", "Fase", "Resposta ao impulso"]
