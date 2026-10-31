r"""P8.12 — Protótipo analógico Butterworth passa-baixas, forma racional.

Enunciado e especificações: :data:`dsp_av2.specs.P8_12` (DTA §2.2, item 9).

Roteiro de solução (DTA §3.2.A e §4, Resolução P8.12):

1. Ordem:
   :math:`N = \left\lceil \frac{\log_{10}[(10^{0.025}-1)/(10^{5}-1)]}{2\log_{10}(500/2000)} \right\rceil
   = \lceil 5.17 \rceil = 6`.
2. Corte (atende R_p exatamente em Ω_p):
   :math:`\Omega_c = 500/(10^{0.025}-1)^{1/12} \approx 632.77` rad/s.
3. Polos: :math:`p_k = \Omega_c e^{j[\pi/2 + (2k+1)\pi/12]}`, k = 0…5.
4. Forma racional: :math:`H_a(s) = \Omega_c^6 / \prod_k (s - p_k) = B(s)/A(s)`.

Execução: ``python -m exercises.p8_12`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib
import numpy as np

from dsp_av2 import plotting, reporting
from dsp_av2.iir import analysis, forms
from dsp_av2.iir.prototypes import AnalogPrototype, design_butterworth
from dsp_av2.specs import P8_12 as SPEC
from dsp_av2.validation import ValidationReport, validate_analog_lowpass

W_MAX = 2 * SPEC.ws  # faixa dos gráficos: 0 a 2Ωs = 4000 rad/s


def design() -> AnalogPrototype:
    """Passos 1–3 do roteiro (fórmulas do livro)."""
    return design_butterworth(SPEC.wp, SPEC.ws, SPEC.rp_db, SPEC.as_db)


def rational_form(proto: AnalogPrototype) -> tuple[np.ndarray, np.ndarray]:
    """Passo 4: coeficientes de B(s) e A(s)."""
    return forms.rational(proto)


def validate(proto: AnalogPrototype) -> ValidationReport:
    return validate_analog_lowpass(SPEC, proto)


def plot(proto: AnalogPrototype):
    """Magnitude, log-magnitude, fase e resposta ao impulso (pedidos pelo enunciado)."""
    w = np.linspace(0, W_MAX, 4001)
    H = analysis.frequency_response(proto, w)
    t = np.linspace(0, analysis.impulse_time_span(proto), 2001)
    return plotting.analog_overview_figure(
        w=w, H=H, panels=SPEC.plots, edges=(SPEC.wp, SPEC.ws), unit=SPEC.freq_unit,
        rp_db=SPEC.rp_db, as_db=SPEC.as_db, phase=np.angle(H),
        t=t, h=analysis.impulse_response(proto, t),
        suptitle=f"P8.12 — Butterworth analógico passa-baixas, N = {proto.N}, "
                 f"Ωc = {proto.params['Omega_c']:.4f} rad/s",
    )


def metrics(proto: AnalogPrototype, report: ValidationReport) -> dict:
    b, a = rational_form(proto)
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "N_real": proto.params["N_real"],
        "N": proto.N,
        "Omega_c": proto.params["Omega_c"],
        "poles": {"real": proto.p.real, "imag": proto.p.imag},
        "B": b,
        "A": a,
        "validation": report.to_dict(),
    }


def main() -> None:
    matplotlib.use("Agg")
    proto = design()
    b, a = rational_form(proto)
    print(f"N = {proto.N} (valor real {proto.params['N_real']:.4f}); "
          f"Ωc = {proto.params['Omega_c']:.4f} rad/s")
    print("B(s) =", np.array2string(b, precision=6))
    print("A(s) =", np.array2string(a, precision=6))
    report = validate(proto)
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(proto), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(proto, report)))


if __name__ == "__main__":
    main()
