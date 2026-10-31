r"""P8.14 — Protótipo analógico Chebyshev-I passa-baixas, forma racional.

Enunciado e especificações: :data:`dsp_av2.specs.P8_14` (DTA §2.2, item 11).

Roteiro de solução (DTA §3.2.B e §4, Resolução P8.14):

1. :math:`\epsilon = \sqrt{10^{0.1}-1} = 0.508847`.
2. :math:`g = \sqrt{10^{5}-1}/\epsilon = 621.46`, :math:`\Omega_r = 15/10 = 1.5`.
3. :math:`N = \lceil \cosh^{-1}g / \cosh^{-1}\Omega_r \rceil = \lceil 7.125/0.9624 \rceil = \lceil 7.40 \rceil = 8`.
4. Polos sobre a elipse (a, b) com :math:`\Omega_p = 10`; N par →
   :math:`H_a(0) = 1/\sqrt{1+\epsilon^2}`, logo :math:`B = a_0/\sqrt{1+\epsilon^2}`.
5. Forma racional :math:`H_a(s) = B/A(s)`, com :math:`B \approx 1.535\times10^6`
   e :math:`a_0 \approx 1.723\times10^6`.

Execução: ``python -m exercises.p8_14`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib
import numpy as np

from dsp_av2 import plotting, reporting
from dsp_av2.iir import analysis, forms
from dsp_av2.iir.prototypes import AnalogPrototype, design_chebyshev1
from dsp_av2.specs import P8_14 as SPEC
from dsp_av2.validation import ValidationReport, validate_analog_lowpass

W_MAX = 2 * SPEC.ws  # gráficos de 0 a 30 rad/s


def design() -> AnalogPrototype:
    """Passos 1–4 (fórmulas do livro)."""
    return design_chebyshev1(SPEC.wp, SPEC.ws, SPEC.rp_db, SPEC.as_db)


def rational_form(proto: AnalogPrototype) -> tuple[np.ndarray, np.ndarray]:
    """Passo 5: coeficientes de B(s) e A(s)."""
    return forms.rational(proto)


def validate(proto: AnalogPrototype) -> ValidationReport:
    return validate_analog_lowpass(SPEC, proto)


def plot(proto: AnalogPrototype):
    """Magnitude, log-magnitude, atraso de grupo e resposta ao impulso (pedidos)."""
    w = np.linspace(0, W_MAX, 6001)
    H = analysis.frequency_response(proto, w)
    t = np.linspace(0, analysis.impulse_time_span(proto), 3001)
    return plotting.analog_overview_figure(
        w=w, H=H, panels=SPEC.plots, edges=(SPEC.wp, SPEC.ws), unit=SPEC.freq_unit,
        rp_db=SPEC.rp_db, as_db=SPEC.as_db, gd=analysis.group_delay(proto, w),
        t=t, h=analysis.impulse_response(proto, t),
        suptitle=f"P8.14 — Chebyshev-I analógico passa-baixas, N = {proto.N}, "
                 f"ε = {proto.params['epsilon']:.6f}",
    )


def metrics(proto: AnalogPrototype, report: ValidationReport) -> dict:
    b, a = rational_form(proto)
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "epsilon": proto.params["epsilon"],
        "N_real": proto.params["N_real"],
        "N": proto.N,
        "dc_gain": proto.params["dc_gain"],
        "poles": {"real": proto.p.real, "imag": proto.p.imag},
        "B": b,
        "A": a,
        "validation": report.to_dict(),
    }


def main() -> None:
    matplotlib.use("Agg")
    proto = design()
    b, a = rational_form(proto)
    print(f"ε = {proto.params['epsilon']:.6f}; N = {proto.N} (valor real {proto.params['N_real']:.4f}); "
          f"H(0) = {proto.params['dc_gain']:.6f}")
    print("B(s) =", np.array2string(b, precision=6))
    print("A(s) =", np.array2string(a, precision=6))
    report = validate(proto)
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(proto), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(proto, report)))


if __name__ == "__main__":
    main()
