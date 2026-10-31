r"""P7.12 — Passa-altas com uma janela fixa (escolha justificada).

Enunciado e especificações: :data:`dsp_av2.specs.P7_12` (DTA §2.1, item 4).

Roteiro de solução (DTA §4, Resolução P7.12, com a correção aritmética do plano §9):

1. :math:`\Delta\omega = 0.6\pi - 0.4\pi = 0.2\pi`; corte :math:`\omega_c = 0.5\pi`.
2. Escolha da janela. O enunciado deixa a janela livre, então cada candidata
   da Tabela 7.1 é projetada e **medida**:

   - Hann: :math:`A_s = 44` dB < 50 dB pela tabela → descartada sem projetar;
   - Hamming: :math:`M = \lceil 6.6/0.2 \rceil + 1 = 34 \to 35` (ímpar), atende
     :math:`A_s`, mas :math:`R_p \approx 0.03` dB > 0.004 dB → descartada;
   - Blackman: :math:`M = \lceil 11/0.2 \rceil + 1 = 56 \to 57` (ímpar),
     :math:`R_p \approx 0.003` dB → **escolhida**.

   O R_p de uma janela é praticamente independente de M (é fixado pelos
   lóbulos laterais). Por isso, aumentar M na Hamming não resolve: é preciso
   uma janela de lóbulos menores.
3. Passa-altas exige M ímpar (Tipo 1): :math:`H(\pi) \ne 0`. Daí 56 → 57.
4. :math:`h(n) = [\delta(n-\alpha) - h_{LP}(n;\omega_c)]\,w_{Blackman}(n)`, :math:`\alpha = 28`.

Execução: ``python -m exercises.p7_12`` (a partir da raiz do projeto).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_12 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_db

CANDIDATES = ("hann", "hamming", "blackman")  # ordem crescente de comprimento


@dataclass(frozen=True)
class Candidate:
    window: str
    table_as_db: float
    design: FIRDesign | None   # None se descartada pela tabela
    report: ValidationReport | None

    @property
    def accepted(self) -> bool:
        return self.report is not None and self.report.passed


def evaluate_candidates() -> list[Candidate]:
    """Passo 2: projeta e mede cada janela fixa admissível pela tabela."""
    out = []
    for name in CANDIDATES:
        table_as = FIXED_WINDOWS[name].min_stopband_db
        if table_as < SPEC.as_db:
            out.append(Candidate(name, table_as, None, None))
            continue
        d = design_fixed_window(SPEC, name)
        out.append(Candidate(name, table_as, d, validate_fir_db(d)))
    return out


def design() -> FIRDesign:
    """Primeira janela (a de menor comprimento) que atende R_p e A_s medidos."""
    for c in evaluate_candidates():
        if c.accepted:
            return c.design
    raise RuntimeError("nenhuma janela fixa atende à especificação")


def validate(d: FIRDesign) -> ValidationReport:
    return validate_fir_db(d)


def plot(d: FIRDesign):
    """Zoom na passagem (pedido pelo enunciado), mais h(n) e |H| em dB completos."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    db = analysis.magnitude_db(analysis.frequency_response(d.h, w))
    return plotting.fir_overview_figure(
        d.h, w, db, SPEC.edges, SPEC.rp_db, SPEC.as_db,
        suptitle=f"P7.12 — Passa-altas, janela de Blackman, M = {d.M}",
        passbands=SPEC.passbands,
    )


def metrics(d: FIRDesign, report: ValidationReport) -> dict:
    cands = []
    for c in evaluate_candidates():
        entry = {"window": c.window, "table_As_dB": c.table_as_db, "accepted": c.accepted}
        if c.design is not None:
            entry |= {"M": c.design.M, "validation": c.report.to_dict()}
        cands.append(entry)
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "candidates": cands,
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
    for c in evaluate_candidates():
        if c.design is None:
            print(f"{c.window:9s}: A_s tabela = {c.table_as_db:g} dB < {SPEC.as_db:g} dB → descartada")
        else:
            rp, as_ = c.report.checks[2].measured, c.report.checks[3].measured
            print(f"{c.window:9s}: M = {c.design.M}, Rp = {rp:.5f} dB, As = {as_:.2f} dB → "
                  f"{'ATENDE' if c.accepted else 'não atende'}")
    d = design()
    report = validate(d)
    print(f"\nEscolhida: {d.window}; M = {d.M} (Tabela 7.1: {d.M_table}); α = {d.alpha:g}")
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(d), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(d, report)))


if __name__ == "__main__":
    main()
