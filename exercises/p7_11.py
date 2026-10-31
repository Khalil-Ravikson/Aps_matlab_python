r"""P7.11 — Passa-faixa com janela de Hamming.

Enunciado e especificações: :data:`dsp_av2.specs.P7_11` (DTA §2.1, item 3).

Roteiro de solução (DTA §4, Resolução P7.11):

1. Transições: :math:`\Delta\omega_1 = 0.4\pi - 0.3\pi = 0.1\pi`,
   :math:`\Delta\omega_2 = 0.6\pi - 0.5\pi = 0.1\pi`; logo :math:`\Delta\omega = 0.1\pi`.
2. Janela imposta: Hamming, com :math:`A_s = 53` dB :math:`\ge 50` dB pela
   Tabela 7.1.
3. :math:`M = \lceil 6.6\pi / 0.1\pi \rceil + 1 = 66 + 1 = 67`
   (Tipo 1, :math:`\alpha = 33`; passa-faixa admitiria M par, mas a tabela já dá ímpar).
4. Cortes ideais: :math:`\omega_{c1} = (0.3+0.4)\pi/2 = 0.35\pi`,
   :math:`\omega_{c2} = (0.5+0.6)\pi/2 = 0.55\pi`.
5. :math:`h(n) = [h_{LP}(n;\omega_{c2}) - h_{LP}(n;\omega_{c1})]\,w_{Hamming}(n)`.

Execução: ``python -m exercises.p7_11`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_11 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_db

WINDOW = "hamming"  # imposta pelo enunciado


def design() -> FIRDesign:
    """Cálculo do filtro (passos 1–5 do roteiro)."""
    if FIXED_WINDOWS[WINDOW].min_stopband_db < SPEC.as_db:
        raise ValueError("a janela imposta não atinge A_s pela Tabela 7.1")
    return design_fixed_window(SPEC, WINDOW)


def validate(d: FIRDesign) -> ValidationReport:
    """Confronta o filtro com R_p e A_s do enunciado."""
    return validate_fir_db(d)


def plot(d: FIRDesign):
    """h(n) e |H| em dB, como pede o enunciado, mais o zoom na passagem."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    db = analysis.magnitude_db(analysis.frequency_response(d.h, w))
    return plotting.fir_overview_figure(
        d.h, w, db, SPEC.edges, SPEC.rp_db, SPEC.as_db,
        suptitle=f"P7.11 — Passa-faixa, janela de Hamming, M = {d.M}",
        passbands=SPEC.passbands,  # painel complementar (não pedido) para mostrar Rp
    )


def metrics(d: FIRDesign, report: ValidationReport) -> dict:
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "window": d.window,
        "transition_width_pi": d.transition_width,
        "cutoffs_pi": d.cutoffs,
        "M_table": d.M_table,
        "M": d.M,
        "alpha": d.alpha,
        "validation": report.to_dict(),
        "h": d.h,
    }


def main() -> None:
    matplotlib.use("Agg")
    d = design()
    report = validate(d)
    print(f"Δω = {d.transition_width:.4g}π; ωc = {[f'{c:.4g}π' for c in d.cutoffs]}; "
          f"M = {d.M} (Tabela 7.1: {d.M_table}); α = {d.alpha:g}")
    print(report)
    fig = plot(d)
    print("Figura:", reporting.save_figure(SPEC.exercise, fig, "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(d, report)))


if __name__ == "__main__":
    main()
