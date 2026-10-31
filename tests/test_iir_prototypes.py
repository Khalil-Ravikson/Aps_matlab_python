"""Fórmulas do livro (DTA §3.2) contra as rotinas analógicas do SciPy.

Valores numéricos usados aqui foram confirmados de forma independente no
plano (§4.2) antes de servirem de referência.
"""

import numpy as np
import pytest
from scipy import signal

from dsp_av2.iir import forms
from dsp_av2.iir import prototypes as P

TWO_PI = 2 * np.pi
BUTTER_CASES = [  # (Ωp, Ωs, Rp, As, N, Ωc)
    (500.0, 2000.0, 0.25, 50.0, 6, 632.7668217),           # P8.12
    (TWO_PI * 1e4, TWO_PI * 2e4, 0.5, 45.0, 9, 70621.0233),  # P8.13
]
CHEB_CASES = [  # (Ωp, Ωs, Rp, As, N)
    (10.0, 15.0, 1.0, 50.0, 8),                    # P8.14
    (TWO_PI * 4e3, TWO_PI * 2e4, 0.5, 45.0, 4),    # P8.15
]


def _sorted(p):
    return np.array(sorted(p, key=lambda x: (round(x.real, 6), round(x.imag, 6))))


@pytest.mark.parametrize("wp, ws, rp, as_, N, wc", BUTTER_CASES)
def test_butterworth_order_and_cutoff(wp, ws, rp, as_, N, wc):
    n, _ = P.butter_order(wp, ws, rp, as_)
    assert n == N
    assert P.butter_cutoff(wp, rp, n) == pytest.approx(wc, rel=1e-9)
    n_sp, wn_sp = signal.buttord(wp, ws, rp, as_, analog=True)
    assert (n_sp, wn_sp) == (N, pytest.approx(wc, rel=1e-9))  # SciPy usa o mesmo critério


@pytest.mark.parametrize("wp, ws, rp, as_, N, wc", BUTTER_CASES)
def test_butterworth_poles_match_scipy(wp, ws, rp, as_, N, wc):
    proto = P.design_butterworth(wp, ws, rp, as_)
    z, p, k = signal.butter(N, proto.params["Omega_c"], analog=True, output="zpk")
    np.testing.assert_allclose(_sorted(proto.p), _sorted(p), rtol=1e-12)
    assert proto.k == pytest.approx(k, rel=1e-12)
    assert np.all(proto.p.real < 0)  # estabilidade: semiplano esquerdo


@pytest.mark.parametrize("wp, ws, rp, as_, N", CHEB_CASES)
def test_chebyshev1_matches_scipy(wp, ws, rp, as_, N):
    n, _ = P.cheb1_order(wp, ws, rp, as_)
    assert n == N
    assert signal.cheb1ord(wp, ws, rp, as_, analog=True)[0] == N
    proto = P.design_chebyshev1(wp, ws, rp, as_)
    z, p, k = signal.cheby1(N, rp, wp, analog=True, output="zpk")
    np.testing.assert_allclose(_sorted(proto.p), _sorted(p), rtol=1e-12)
    assert proto.k == pytest.approx(k, rel=1e-12)


def test_p8_12_rational_coefficients():
    """B(s) e A(s) do DTA §4 (Resolução P8.12), confirmados no plano §4.2."""
    b, a = forms.rational(P.design_butterworth(500, 2000, 0.25, 50))
    assert b == pytest.approx([6.41892345e16], rel=1e-8)
    expected_a = [1, 2.444823e3, 2.988580e6, 2.316084e9, 1.196609e12, 3.919424e14, 6.418923e16]
    np.testing.assert_allclose(a, expected_a, rtol=5e-7)
    assert not np.iscomplexobj(a)


def test_p8_13_cascade_angles_are_k_pi_over_9():
    """Errata do DTA: para N = 9 (ímpar), θ_k = kπ/9, não (2k−1)π/18."""
    proto = P.design_butterworth(TWO_PI * 1e4, TWO_PI * 2e4, 0.5, 45)
    c = forms.cascade(proto)
    wc = proto.params["Omega_c"]
    biquads = [s for s in c.sections if s.order == 2]
    first = [s for s in c.sections if s.order == 1]
    assert len(biquads) == 4 and len(first) == 1
    assert first[0].den[1] == pytest.approx(wc)  # polo real em −Ωc
    theta = np.sort(np.arccos([s.den[1] / (2 * wc) for s in biquads]))
    np.testing.assert_allclose(theta, np.arange(1, 5) * np.pi / 9, rtol=1e-10)
    assert c.gain == pytest.approx(1.0)


def test_p8_15_cascade_matches_dta():
    proto = P.design_chebyshev1(TWO_PI * 4e3, TWO_PI * 2e4, 0.5, 45)
    c = forms.cascade(proto)
    dens = sorted((s.den for s in c.sections), key=lambda d: d[1])
    np.testing.assert_allclose(dens[0], [1, 8814.21, 6.7178e8], rtol=1e-4)
    np.testing.assert_allclose(dens[1], [1, 21279.38, 2.2513e8], rtol=1e-4)
    assert c.gain == pytest.approx(0.94406, rel=1e-5)


@pytest.mark.parametrize("design, args", [
    (P.design_butterworth, (500, 2000, 0.25, 50)),
    (P.design_butterworth, (TWO_PI * 1e4, TWO_PI * 2e4, 0.5, 45)),
    (P.design_chebyshev1, (10, 15, 1, 50)),
    (P.design_chebyshev1, (TWO_PI * 4e3, TWO_PI * 2e4, 0.5, 45)),
])
def test_cascade_recomposes_to_same_frequency_response(design, args):
    proto = design(*args)
    c = forms.cascade(proto)
    w = np.geomspace(args[0] / 100, args[1] * 10, 500)
    _, H = signal.freqs_zpk(proto.z, proto.p, proto.k, worN=w)
    Hc = c.gain * np.ones_like(w, dtype=complex)
    for s in c.sections:
        Hc *= signal.freqs(s.num, s.den, worN=w)[1]
    np.testing.assert_allclose(Hc, H, rtol=1e-9, atol=1e-15)


def test_poly_to_latex():
    assert forms.poly_to_latex(np.array([1.0, -2.5e3, 4.0])) == r"s^{2} - 2.500000\times10^{3}s + 4.000000"
