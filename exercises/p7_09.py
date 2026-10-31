r"""P7.9 — Passa-faixa de fase linear com janela de Hann.

Enunciado e especificações: :data:`dsp_av2.specs.P7_9` (DTA §2.1, item 1).

Roteiro de solução (DTA §4, Resolução P7.9):

1. Transições: :math:`\Delta\omega_1 = 0.35\pi - 0.2\pi = 0.15\pi`,
   :math:`\Delta\omega_2 = 0.75\pi - 0.55\pi = 0.20\pi`;
   usa-se a menor, :math:`\Delta\omega = 0.15\pi`.
2. Janela imposta pelo enunciado: Hann. A Tabela 7.1 lhe atribui
   :math:`A_s = 44` dB :math:`\ge 40` dB exigidos, então ela é adequada.
3. :math:`M = \lceil 6.2\pi / 0.15\pi \rceil + 1 = \lceil 41.33 \rceil + 1 = 43`
   (ímpar → Tipo 1, :math:`\alpha = 21`).
4. Cortes ideais: :math:`\omega_{c1} = (0.2+0.35)\pi/2 = 0.275\pi`,
   :math:`\omega_{c2} = (0.55+0.75)\pi/2 = 0.65\pi`.
5. :math:`h(n) = [h_{LP}(n;\omega_{c2}) - h_{LP}(n;\omega_{c1})]\,w_{Hann}(n)`.

Execução: ``python -m exercises.p7_09`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_9 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_db

WINDOW = "hann"  # imposta pelo enunciado


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
        suptitle=f"P7.9 — Passa-faixa, janela de Hann, M = {d.M}",
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
