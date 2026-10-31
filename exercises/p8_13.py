r"""P8.13 — Protótipo analógico Butterworth passa-baixas (em Hz), forma em cascata.

Enunciado e especificações: :data:`dsp_av2.specs.P8_13` (DTA §2.2, item 10).

Roteiro de solução (DTA §3.2.A e §4, Resolução P8.13, com a errata do plano §4.2):

1. :math:`\Omega_p = 2\pi\cdot10^4`, :math:`\Omega_s = 2\pi\cdot2\cdot10^4` rad/s.
2. :math:`N = \lceil \log_{10}[(10^{0.05}-1)/(10^{4.5}-1)] / (2\log_{10}0.5) \rceil = \lceil 8.99 \rceil = 9`.
3. :math:`\Omega_c = \Omega_p/(10^{0.05}-1)^{1/18} \approx 70621` rad/s (≈ 11.24 kHz).
4. Cascata (N ímpar → 1 seção de 1ª ordem + 4 biquads):

   .. math::
       H_a(s) = \frac{\Omega_c}{s+\Omega_c}\prod_{k=1}^{4}
                \frac{\Omega_c^2}{s^2 + 2\Omega_c\cos\theta_k\,s + \Omega_c^2},
       \qquad \theta_k = \frac{k\pi}{9}

   **Errata:** o DTA usa :math:`\theta_k = (2k-1)\pi/18`, que vale para N par.
   Com N = 9 há um polo real em :math:`-\Omega_c` e os demais ficam a
   20°, 40°, 60° e 80° do eixo real negativo.

Execução: ``python -m exercises.p8_13`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib
import numpy as np

from dsp_av2 import plotting, reporting
from dsp_av2.iir import analysis, forms
from dsp_av2.iir.prototypes import AnalogPrototype, design_butterworth
from dsp_av2.specs import P8_13 as SPEC
from dsp_av2.validation import ValidationReport, validate_analog_lowpass

W_MAX = 2 * SPEC.ws  # gráficos de 0 a 2Fs = 40 kHz


def design() -> AnalogPrototype:
    """Passos 1–3 (fórmulas do livro)."""
    return design_butterworth(SPEC.wp, SPEC.ws, SPEC.rp_db, SPEC.as_db)


def cascade_form(proto: AnalogPrototype) -> forms.Cascade:
    """Passo 4: seções reais de 1ª/2ª ordem, cada uma com ganho DC unitário."""
    return forms.cascade(proto)


def section_angles(proto: AnalogPrototype, c: forms.Cascade) -> np.ndarray:
    r"""θ_k de cada biquad: :math:`\cos\theta_k = a_1/(2\Omega_c)`, em graus."""
    wc = proto.params["Omega_c"]
    return np.degrees([np.arccos(s.den[1] / (2 * wc)) for s in c.sections if s.order == 2])


def validate(proto: AnalogPrototype) -> ValidationReport:
    return validate_analog_lowpass(SPEC, proto)


def plot(proto: AnalogPrototype):
    """Magnitude, log-magnitude, atraso de grupo e resposta ao impulso (pedidos)."""
    w = np.linspace(0, W_MAX, 4001)
    H = analysis.frequency_response(proto, w)
    t = np.linspace(0, analysis.impulse_time_span(proto), 2001)
    return plotting.analog_overview_figure(
        w=w, H=H, panels=SPEC.plots, edges=(SPEC.wp, SPEC.ws), unit=SPEC.freq_unit,
        rp_db=SPEC.rp_db, as_db=SPEC.as_db, gd=analysis.group_delay(proto, w),
        t=t, h=analysis.impulse_response(proto, t),
        suptitle=f"P8.13 — Butterworth analógico passa-baixas, N = {proto.N}, "
                 f"Fc = {proto.params['Omega_c'] / (2 * np.pi):.2f} Hz",
    )


def metrics(proto: AnalogPrototype, report: ValidationReport) -> dict:
    c = cascade_form(proto)
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "N_real": proto.params["N_real"],
        "N": proto.N,
        "Omega_c": proto.params["Omega_c"],
        "Fc_Hz": proto.params["Omega_c"] / (2 * np.pi),
        "poles": {"real": proto.p.real, "imag": proto.p.imag},
        "cascade": {
            "gain": c.gain,
            "sections": [{"num": s.num, "den": s.den} for s in c.sections],
            "theta_deg": section_angles(proto, c),
        },
        "validation": report.to_dict(),
    }


def main() -> None:
    matplotlib.use("Agg")
    proto = design()
    c = cascade_form(proto)
    print(f"N = {proto.N} (valor real {proto.params['N_real']:.4f}); "
          f"Ωc = {proto.params['Omega_c']:.4f} rad/s ({proto.params['Omega_c'] / (2 * np.pi):.2f} Hz)")
    print(f"Ganho global G = {c.gain:.6g}")
    for s in c.sections:
        print("  num =", np.array2string(s.num, precision=6), " den =", np.array2string(s.den, precision=6))
    print("θ_k (graus) =", np.round(section_angles(proto, c), 6))
    report = validate(proto)
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(proto), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(proto, report)))


if __name__ == "__main__":
    main()
