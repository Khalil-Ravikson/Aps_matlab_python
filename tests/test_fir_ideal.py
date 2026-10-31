"""Respostas ideais h_d(n) (DTA §3.1.E)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import ideal

PI = np.pi


@pytest.mark.parametrize("M", [21, 22, 43])
def test_lowpass_matches_firwin_boxcar(M):
    """firwin com janela retangular e sem normalização = h_LP ideal truncada."""
    ref = firwin(M, 0.3, window="boxcar", scale=False)
    np.testing.assert_allclose(ideal.ideal_lowpass(0.3 * PI, M), ref, atol=1e-14)


def test_lowpass_center_sample():
    h = ideal.ideal_lowpass(0.275 * PI, 43)
    assert h[21] == pytest.approx(0.275)


@pytest.mark.parametrize("M", [20, 43])
def test_ideal_responses_are_symmetric(M):
    for h in (ideal.ideal_lowpass(0.3 * PI, M), ideal.ideal_bandpass(0.3 * PI, 0.6 * PI, M)):
        np.testing.assert_allclose(h, h[::-1], atol=1e-15)


def test_highpass_and_bandstop_require_odd_length():
    with pytest.raises(ValueError, match="M ímpar"):
        ideal.ideal_highpass(0.5 * PI, 42)
    with pytest.raises(ValueError, match="M ímpar"):
        ideal.ideal_bandstop(0.35 * PI, 0.65 * PI, 66)


def test_bandpass_definition():
    M = 43
    expected = ideal.ideal_lowpass(0.65 * PI, M) - ideal.ideal_lowpass(0.275 * PI, M)
    np.testing.assert_array_equal(ideal.ideal_bandpass(0.275 * PI, 0.65 * PI, M), expected)


@pytest.mark.parametrize(
    "cutoffs, gains, explicit",
    [
        ([0.3], [1, 0], lambda M: ideal.ideal_lowpass(0.3 * PI, M)),
        ([0.5], [0, 1], lambda M: ideal.ideal_highpass(0.5 * PI, M)),
        ([0.35, 0.65], [0, 1, 0], lambda M: ideal.ideal_bandpass(0.35 * PI, 0.65 * PI, M)),
        ([0.35, 0.65], [1, 0, 1], lambda M: ideal.ideal_bandstop(0.35 * PI, 0.65 * PI, M)),
        # Escada do P7.15: G1 h_LP1 + G2 (h_LP2 − h_LP1) = 0.5 h_LP1 + 0.5 h_LP2 (DTA)
        ([0.35, 0.75], [1, 0.5, 0],
         lambda M: 0.5 * ideal.ideal_lowpass(0.35 * PI, M) + 0.5 * ideal.ideal_lowpass(0.75 * PI, M)),
    ],
)
def test_multiband_reduces_to_dta_formulas(cutoffs, gains, explicit):
    M = 43
    got = ideal.ideal_multiband([c * PI for c in cutoffs], gains, M)
    np.testing.assert_allclose(got, explicit(M), atol=1e-15)
