r"""P8.15 — Protótipo analógico Chebyshev-I passa-baixas (em Hz), forma em cascata.

Enunciado e especificações: :data:`dsp_av2.specs.P8_15` (DTA §2.2, item 12).

Roteiro de solução (DTA §3.2.B e §4, Resolução P8.15):

1. :math:`\Omega_p = 2\pi\cdot4\cdot10^3`, :math:`\Omega_s = 2\pi\cdot2\cdot10^4` rad/s.
2. :math:`\epsilon = \sqrt{10^{0.05}-1} = 0.349311`; :math:`g = 509.08`; :math:`\Omega_r = 5`.
3. :math:`N = \lceil \cosh^{-1}g / \cosh^{-1}5 \rceil = \lceil 6.926/2.292 \rceil = \lceil 3.02 \rceil = 4`.
4. Polos: :math:`-4407.10 \pm j25541.22` e :math:`-10639.69 \pm j10579.52`.
5. Cascata (N par → dois biquads, ganho global :math:`1/\sqrt{1+\epsilon^2} = 0.94406`):

   .. math::
       H_a(s) = 0.94406\cdot\frac{6.7178\times10^8}{s^2 + 8814.21\,s + 6.7178\times10^8}
                \cdot\frac{2.2513\times10^8}{s^2 + 21279.38\,s + 2.2513\times10^8}

Execução: ``python -m exercises.p8_15`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib
import numpy as np

from dsp_av2 import plotting, reporting
from dsp_av2.iir import analysis, forms
from dsp_av2.iir.prototypes import AnalogPrototype, design_chebyshev1
from dsp_av2.specs import P8_15 as SPEC
from dsp_av2.validation import ValidationReport, validate_analog_lowpass

W_MAX = 2 * SPEC.ws  # gráficos de 0 a 2Fs = 40 kHz


def design() -> AnalogPrototype:
    """Passos 1–4 (fórmulas do livro)."""
    return design_chebyshev1(SPEC.wp, SPEC.ws, SPEC.rp_db, SPEC.as_db)


def cascade_form(proto: AnalogPrototype) -> forms.Cascade:
    """Passo 5: biquads com ganho DC unitário e ganho global G."""
    return forms.cascade(proto)


def validate(proto: AnalogPrototype) -> ValidationReport:
    return validate_analog_lowpass(SPEC, proto)


def plot(proto: AnalogPrototype):
    """Magnitude, log-magnitude, fase e resposta ao impulso (pedidos)."""
    w = np.linspace(0, W_MAX, 8001)
    H = analysis.frequency_response(proto, w)
    t = np.linspace(0, analysis.impulse_time_span(proto), 3001)
    return plotting.analog_overview_figure(
        w=w, H=H, panels=SPEC.plots, edges=(SPEC.wp, SPEC.ws), unit=SPEC.freq_unit,
        rp_db=SPEC.rp_db, as_db=SPEC.as_db, phase=np.angle(H),
        t=t, h=analysis.impulse_response(proto, t),
        suptitle=f"P8.15 — Chebyshev-I analógico passa-baixas, N = {proto.N}, "
                 f"ε = {proto.params['epsilon']:.6f}",
    )


def metrics(proto: AnalogPrototype, report: ValidationReport) -> dict:
    c = cascade_form(proto)
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "epsilon": proto.params["epsilon"],
        "N_real": proto.params["N_real"],
        "N": proto.N,
        "poles": {"real": proto.p.real, "imag": proto.p.imag},
        "cascade": {"gain": c.gain, "sections": [{"num": s.num, "den": s.den} for s in c.sections]},
        "validation": report.to_dict(),
    }


def main() -> None:
    matplotlib.use("Agg")
    proto = design()
    c = cascade_form(proto)
    print(f"ε = {proto.params['epsilon']:.6f}; N = {proto.N} (valor real {proto.params['N_real']:.4f})")
    print("Polos:", np.round(np.sort_complex(proto.p), 2))
    print(f"Ganho global G = {c.gain:.6f}")
    for s in c.sections:
        print("  num =", np.array2string(s.num, precision=6), " den =", np.array2string(s.den, precision=6))
    report = validate(proto)
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(proto), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(proto, report)))


if __name__ == "__main__":
    main()
