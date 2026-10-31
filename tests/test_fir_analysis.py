"""Análise em frequência e medições (Seção 7.2 / freqz_m)."""

import numpy as np
import pytest

from dsp_av2.fir import analysis, ideal, windows
from dsp_av2.specs import Band

PI = np.pi


def test_frequency_grid_contains_edges_and_book_step():
    w = analysis.frequency_grid((0.2, 0.35, 0.55, 0.75), 501)
    for e in (0.2, 0.35, 0.55, 0.75):
        assert np.any(np.isclose(w, e * PI, atol=1e-15))
    assert w[0] == 0 and w[-1] == pytest.approx(PI)
    assert np.all(np.diff(w) > 0)


def test_grid_without_edges_overestimates_attenuation():
    """Documenta a origem das divergências do DTA em P7.10 (decisão D8)."""
    M = 67
    h = (ideal.ideal_bandstop(0.35 * PI, 0.65 * PI, M)) * windows.hamming(M)
    pb, sb = [Band(0, 0.3, 1), Band(0.7, 1, 1)], [Band(0.4, 0.6, 0)]
    _, as_edges = analysis.measure_rp_as(h, pb, sb, analysis.frequency_grid((0.3, 0.4, 0.6, 0.7), 501))
    _, as_2049 = analysis.measure_rp_as(h, pb, sb, np.linspace(0, PI, 2049))
    assert as_edges == pytest.approx(50.04, abs=0.01)
    assert as_2049 == pytest.approx(51.14, abs=0.01)  # valor que aparece no DTA (51.11)
    assert as_2049 > as_edges


@pytest.mark.parametrize("M, expected", [(43, 1), (46, 2)])
def test_linear_phase_type(M, expected):
    h = ideal.ideal_bandpass(0.3 * PI, 0.6 * PI, M) * windows.hann(M)
    assert analysis.linear_phase_type(h) == expected


def test_linear_phase_type_rejects_asymmetric():
    with pytest.raises(ValueError):
        analysis.linear_phase_type(np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("M", [43, 46])
def test_amplitude_response_is_signed_magnitude(M):
    h = ideal.ideal_bandpass(0.3 * PI, 0.6 * PI, M) * windows.hann(M)
    w = analysis.frequency_grid((), 1001)
    Hr = analysis.amplitude_response(h, w)
    np.testing.assert_allclose(np.abs(Hr), np.abs(analysis.frequency_response(h, w)), atol=1e-12)
    if M % 2 == 0:  # Tipo 2: zero obrigatório em ω = π
        assert Hr[-1] == pytest.approx(0, abs=1e-12)


def test_magnitude_db_peak_is_zero():
    h = ideal.ideal_lowpass(0.4 * PI, 31) * windows.hamming(31)
    db = analysis.magnitude_db(analysis.frequency_response(h, analysis.frequency_grid()))
    assert db.max() == pytest.approx(0.0)
