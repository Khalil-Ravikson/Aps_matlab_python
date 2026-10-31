"""Janelas e fórmulas de dimensionamento (DTA §3.1.C–D)."""

import numpy as np
import pytest
from scipy.signal import windows as sw

from dsp_av2.fir import windows as W

LENGTHS = [5, 16, 43, 67, 110]


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("name", ["hann", "hamming", "blackman"])
def test_fixed_windows_match_scipy_symmetric(name, M):
    ours = W.FIXED_WINDOWS[name].function(M)
    ref = getattr(sw, name)(M, sym=True)
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-12)


def test_hann_has_zero_endpoints():
    """Decisão D1: Hann com zeros nas extremidades (hann(M) do MATLAB)."""
    w = W.hann(43)
    assert w[0] == pytest.approx(0, abs=1e-15)
    assert w[-1] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize("M", LENGTHS)
@pytest.mark.parametrize("beta", [0.0, 3.3953, 4.0909, 5.6533])
def test_kaiser_matches_numpy(M, beta):
    np.testing.assert_allclose(W.kaiser(M, beta), np.kaiser(M, beta), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "as_db, beta",
    [
        (-20 * np.log10(0.005), 4.0909),  # P7.13 (DTA)
        (40.0, 3.3953),                   # P7.14 (DTA)
        (60.0, 5.65326),                  # P7.15 (DTA)
        (21.0, 0.0),
        (10.0, 0.0),
    ],
)
def test_kaiser_beta(as_db, beta):
    assert W.kaiser_beta(as_db) == pytest.approx(beta, abs=5e-5)


def test_kaiser_beta_is_continuous_at_50db():
    """Os dois ramos empíricos quase coincidem em 50 dB (4.5513 vs 4.5335)."""
    below = 0.5842 * (50 - 21) ** 0.4 + 0.07886 * (50 - 21)
    assert abs(W.kaiser_beta(50.0) - below) < 0.02


@pytest.mark.parametrize(
    "name, dw_pi, M",
    [
        ("hann", 0.15, 43),      # P7.9
        ("hamming", 0.10, 67),   # P7.10 / P7.11
        ("blackman", 0.20, 56),  # P7.12: 11/0.2 = 55 → 56; a paridade (PA) leva a 57
        ("hann", 0.10, 63),      # P7.16 (Tabela 7.1)
        ("blackman", 0.10, 111),
    ],
)
def test_fixed_window_length_table(name, dw_pi, M):
    assert W.fixed_window_length(name, dw_pi * np.pi) == M


@pytest.mark.parametrize("lo, hi", [(0.4, 0.5), (0.5, 0.6), (0.25, 0.35), (0.65, 0.75)])
def test_fixed_window_length_is_robust_to_float_noise(lo, hi):
    """(hi − lo)π gera 6.6π/Δω = 66.00000000000001; o resultado deve ser 67."""
    assert W.fixed_window_length("hamming", (hi - lo) * np.pi) == 67


def test_kaiser_length_estimate():
    """⌈(A_s − 7.95)/(2.285 Δω)⌉ + 1 (DTA §3.1.D), apenas a estimativa."""
    dw = 0.1 * np.pi
    assert W.kaiser_length_estimate(-20 * np.log10(0.005), dw) == 55
    assert W.kaiser_length_estimate(40.0, dw) == 46
    assert W.kaiser_length_estimate(60.0, dw) == 74
