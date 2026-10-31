"""P7.12 — validação contra a especificação do enunciado (não contra o DTA)."""

import numpy as np
import pytest
from scipy.signal import firwin

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.specs import P7_12 as SPEC
from exercises import p7_12


@pytest.fixture(scope="module")
def design():
    return p7_12.design()


def test_window_selection():
    by_name = {c.window: c for c in p7_12.evaluate_candidates()}
    assert by_name["hann"].design is None            # descartada pela tabela (44 < 50 dB)
    assert by_name["hamming"].design.M == 35         # ⌈6.6/0.2⌉ + 1 = 34 → ímpar 35
    assert not by_name["hamming"].accepted           # R_p ≈ 0.04 dB > 0.004 dB
    assert by_name["blackman"].accepted


def test_design_parameters(design):
    assert design.window == "blackman"
    assert design.transition_width == pytest.approx(0.2)
    assert design.cutoffs == pytest.approx((0.5,))
    assert design.M_table == 56   # ⌈11π/0.2π⌉ + 1 = 56 (o DTA escreve 57 direto)
    assert design.M == 57         # paridade: passa-altas exige Tipo 1
    assert design.alpha == 28


def test_linear_phase_type_1(design):
    assert analysis.linear_phase_type(design.h) == 1


def test_meets_specification(design):
    report = p7_12.validate(design)
    assert report.passed, str(report)


@pytest.mark.parametrize("M", [35, 45, 65, 85, 95])
def test_hamming_ripple_does_not_improve_with_length(M):
    """R_p da Hamming fica em ~0.027–0.040 dB para qualquer M: aumentar M não resolve."""
    d = design_fixed_window(SPEC, "hamming", M=M)
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    rp, _ = analysis.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, w)
    assert 0.02 < rp < 0.045
    assert rp > SPEC.rp_db


def test_cross_check_with_firwin(design):
    ref = firwin(57, 0.5, window="blackman", pass_zero=False, scale=False)
    np.testing.assert_allclose(design.h, ref, atol=1e-14)


def test_figure_has_passband_zoom(design):
    import matplotlib
    matplotlib.use("Agg")
    titles = [ax.get_title(loc="left") for ax in p7_12.plot(design).axes]
    assert any("Zoom na passagem" in t for t in titles)
