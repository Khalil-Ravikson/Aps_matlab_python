"""Atraso de grupo e resposta ao impulso: método principal vs verificação independente."""

import numpy as np
import pytest
from scipy import signal

from dsp_av2.iir import analysis
from dsp_av2.iir import prototypes as P

TWO_PI = 2 * np.pi
PROTOS = [
    P.design_butterworth(500, 2000, 0.25, 50),
    P.design_butterworth(TWO_PI * 1e4, TWO_PI * 2e4, 0.5, 45),
    P.design_chebyshev1(10, 15, 1, 50),
    P.design_chebyshev1(TWO_PI * 4e3, TWO_PI * 2e4, 0.5, 45),
]


@pytest.mark.parametrize("proto", PROTOS)
def test_group_delay_matches_numerical_phase_derivative(proto):
    wmax = 3 * np.max(np.abs(proto.p))
    w = np.linspace(0, wmax, 200001)
    phase = np.unwrap(np.angle(analysis.frequency_response(proto, w)))
    tau_num = -np.gradient(phase, w)
    tau = analysis.group_delay(proto, w)
    inner = slice(10, -10)
    np.testing.assert_allclose(tau_num[inner], tau[inner], rtol=1e-4, atol=1e-6 * tau.max())


def test_butterworth_dc_group_delay_closed_form():
    """τ(0) = Σ −σ_k / |p_k|² (fórmula analítica avaliada em Ω = 0)."""
    proto = PROTOS[0]
    expected = np.sum(-proto.p.real / np.abs(proto.p) ** 2)
    assert analysis.group_delay(proto, np.array([0.0]))[0] == pytest.approx(expected)


@pytest.mark.parametrize("proto", PROTOS)
def test_impulse_response_matches_residues(proto):
    t = np.linspace(0, analysis.impulse_time_span(proto), 3001)
    h_sim = analysis.impulse_response(proto, t)
    h_res = analysis.impulse_response_residues(proto, t)
    assert not np.iscomplexobj(h_sim)
    np.testing.assert_allclose(h_sim, h_res, atol=1e-9 * np.max(np.abs(h_res)))


@pytest.mark.parametrize("proto", PROTOS)
def test_impulse_response_area_equals_dc_gain(proto):
    """∫h(t)dt = H(0)."""
    t = np.linspace(0, analysis.impulse_time_span(proto, decay_db=200), 400001)
    area = np.trapezoid(analysis.impulse_response_residues(proto, t), t)
    H0 = np.real(analysis.frequency_response(proto, np.array([0.0]))[0])
    assert area == pytest.approx(H0, rel=1e-5)


def test_magnitude_db_is_absolute():
    """Sem normalização: o Chebyshev-I de ordem par tem |H(0)| = −Rp dB."""
    proto = PROTOS[2]
    db0 = analysis.magnitude_db(analysis.frequency_response(proto, np.array([0.0])))[0]
    assert db0 == pytest.approx(-1.0, abs=1e-9)
