"""P8.12 — validação contra o enunciado (e valores do DTA confirmados no plano §4.2)."""

import numpy as np
import pytest

from dsp_av2.iir import analysis
from dsp_av2.specs import P8_12 as SPEC
from exercises import p8_12


@pytest.fixture(scope="module")
def proto():
    return p8_12.design()


def test_order_and_cutoff(proto):
    assert proto.N == 6
    assert proto.params["N_real"] == pytest.approx(5.17, abs=0.005)
    assert proto.params["Omega_c"] == pytest.approx(632.7668, abs=1e-4)


def test_meets_specification(proto):
    report = p8_12.validate(proto)
    assert report.passed, str(report)


def test_passband_edge_is_met_with_equality(proto):
    """Ω_c foi escolhido para |H(jΩp)| = −Rp dB exatamente (convenção do livro)."""
    db = analysis.magnitude_db(analysis.frequency_response(proto, np.array([SPEC.wp])))[0]
    assert db == pytest.approx(-SPEC.rp_db, abs=1e-12)


def test_rational_form_is_monic_with_real_coefficients(proto):
    b, a = p8_12.rational_form(proto)
    assert a[0] == 1.0 and len(a) == 7 and len(b) == 1
    assert b[0] == pytest.approx(a[-1])  # H(0) = B/a0 = 1
    assert np.all(a > 0)  # polinômio de Hurwitz: coeficientes positivos


def test_figure_has_requested_panels(proto):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p8_12.plot(proto).axes]
    assert titles == ["Magnitude", "Log-magnitude", "Fase", "Resposta ao impulso"]
