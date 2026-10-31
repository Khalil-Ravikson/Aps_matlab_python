"""P8.13 — Butterworth em cascata (validação contra o enunciado + errata θ_k)."""

import numpy as np
import pytest
from scipy import signal

from dsp_av2.iir import analysis, forms
from dsp_av2.specs import P8_13 as SPEC
from exercises import p8_13


@pytest.fixture(scope="module")
def proto():
    return p8_13.design()


def test_hz_to_rad_conversion():
    assert SPEC.wp == pytest.approx(2 * np.pi * 10e3)
    assert SPEC.ws == pytest.approx(2 * np.pi * 20e3)


def test_order_and_cutoff(proto):
    assert proto.N == 9
    assert proto.params["N_real"] == pytest.approx(8.99, abs=0.005)
    assert proto.params["Omega_c"] == pytest.approx(70621.0233, abs=1e-3)


def test_meets_specification(proto):
    report = p8_13.validate(proto)
    assert report.passed, str(report)


def test_cascade_structure_and_errata(proto):
    c = p8_13.cascade_form(proto)
    assert [s.order for s in c.sections] == [1, 2, 2, 2, 2]
    assert c.gain == pytest.approx(1.0)
    np.testing.assert_allclose(np.sort(p8_13.section_angles(proto, c)), [20, 40, 60, 80], atol=1e-9)
    wc = proto.params["Omega_c"]
    for s in c.sections:
        assert s.num[0] == pytest.approx(wc ** s.order, rel=1e-12)  # ganho DC unitário por seção


def test_cascade_equals_zpk_response(proto):
    c = p8_13.cascade_form(proto)
    w = np.geomspace(1e2, 1e7, 400)
    _, H = signal.freqs_zpk(proto.z, proto.p, proto.k, worN=w)
    Hc = c.gain * np.ones_like(w, dtype=complex)
    for s in c.sections:
        Hc *= signal.freqs(s.num, s.den, worN=w)[1]
    np.testing.assert_allclose(Hc, H, rtol=1e-9, atol=1e-30)


def test_group_delay_dc_value(proto):
    """τ(0) = Σ −σ_k/|p_k|² = (1/Ωc) Σ cos(ângulo ao eixo real negativo)."""
    expected = (1 + 2 * np.sum(np.cos(np.radians([20, 40, 60, 80])))) / proto.params["Omega_c"]
    assert analysis.group_delay(proto, np.array([0.0]))[0] == pytest.approx(expected, rel=1e-12)


def test_figure_has_requested_panels(proto):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p8_13.plot(proto).axes]
    assert titles == ["Magnitude", "Log-magnitude", "Atraso de grupo", "Resposta ao impulso"]
