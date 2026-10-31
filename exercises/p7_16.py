r"""P7.16 — Passa-faixa com a janela fixa de menor comprimento.

Enunciado e especificações: :data:`dsp_av2.specs.P7_16` (DTA §2.1, item 8).

Roteiro de solução (DTA §4, Resolução P7.16, com as decisões D2/D4/D5 do plano):

1. :math:`\Delta\omega = 0.1\pi`; cortes :math:`\omega_{c1} = 0.35\pi`, :math:`\omega_{c2} = 0.55\pi`.
2. Janelas fixas com :math:`A_s` nominal ≥ 40 dB: Hann (44), Hamming (53), Blackman (74).
3. Comprimento pela Tabela 7.1 (critério do DTA): Hann 63, Hamming 67, Blackman 111.
4. **Comprimento mínimo medido (D2/D5):** para cada janela, varrem-se todos os M
   (passa-faixa admite Tipo 2, D4) e toma-se o menor que atende R_p e A_s:
   Hamming 61, Hann 62, Blackman 82.
5. Resposta final: **Hamming, M = 61**. A tabela é conservadora porque seu Δω
   corresponde à atenuação plena da janela (53 dB na Hamming), e aqui bastam 40 dB.

Execução: ``python -m exercises.p7_16`` (a partir da raiz do projeto).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, LengthSearch, admissible_lengths, design_fixed_window, search_min_length
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_16 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_db


@dataclass(frozen=True)
class WindowResult:
    window: str
    M_table: int
    table_design_passes: bool
    search: LengthSearch

    @property
    def M_min(self) -> int:
        return self.search.M_min


def _accept(d: FIRDesign) -> bool:
    return validate_fir_db(d).passed


def evaluate_windows() -> list[WindowResult]:
    """Passos 2–4 para cada janela fixa admissível."""
    out = []
    for name, info in FIXED_WINDOWS.items():
        if info.min_stopband_db < SPEC.as_db:
            continue
        table = design_fixed_window(SPEC, name)
        s = search_min_length(
            build=lambda M, name=name: design_fixed_window(SPEC, name, M=M),
            accept=_accept,
            lengths=admissible_lengths(SPEC, table.M // 2, table.M + 10),
        )
        out.append(WindowResult(name, table.M, _accept(table), s))
    return out


def design() -> FIRDesign:
    """Passo 5: a janela com o menor comprimento mínimo medido."""
    best = min(evaluate_windows(), key=lambda r: r.M_min)
    return best.search.best


def validate(d: FIRDesign) -> ValidationReport:
    return validate_fir_db(d)


def plot(d: FIRDesign):
    """Log-magnitude em dB e stem de h(n) (pedidos), mais zoom na passagem."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    db = analysis.magnitude_db(analysis.frequency_response(d.h, w))
    return plotting.fir_overview_figure(
        d.h, w, db, SPEC.edges, SPEC.rp_db, SPEC.as_db,
        suptitle=f"P7.16 — Passa-faixa, janela fixa de menor comprimento: {d.window.capitalize()}, M = {d.M}",
        passbands=SPEC.passbands,
    )


def metrics(d: FIRDesign, report: ValidationReport, results: list[WindowResult]) -> dict:
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "windows": [
            {"window": r.window, "M_table": r.M_table, "table_design_passes": r.table_design_passes,
             "M_min_measured": r.M_min, "search": r.search.tried}
            for r in results
        ],
        "window": d.window,
        "cutoffs_pi": d.cutoffs,
        "M": d.M,
        "alpha": d.alpha,
        "validation": report.to_dict(),
        "h": d.h,
    }


def main() -> None:
    matplotlib.use("Agg")
    results = evaluate_windows()
    for r in results:
        print(f"{r.window:9s}: M tabela = {r.M_table:3d} ({'atende' if r.table_design_passes else 'não atende'}); "
              f"M mínimo medido = {r.M_min}")
    d = design()
    report = validate(d)
    print(f"\nEscolhida: {d.window}, M = {d.M} (Tipo {report.linear_phase_type})")
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(d), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(d, report, results)))


if __name__ == "__main__":
    main()
