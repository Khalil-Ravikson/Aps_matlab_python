r"""P7.13 — Rejeita-faixa de comprimento mínimo com janela de Kaiser.

Enunciado e especificações: :data:`dsp_av2.specs.P7_13` (DTA §2.1, item 5).

Roteiro de solução (DTA §3.1.D e §4, Resolução P7.13, com a correção do plano §4):

1. :math:`\delta = \min(0.025, 0.005) = 0.005`,
   :math:`A_s = -20\log_{10} 0.005 = 46.02` dB.
2. :math:`21 < A_s < 50` → :math:`\beta = 0.5842(A_s-21)^{0.4} + 0.07886(A_s-21) = 4.0909`.
3. :math:`\Delta\omega = 0.35\pi - 0.25\pi = 0.1\pi`; estimativa
   :math:`M = \lceil 38.07/(2.285 \cdot 0.1\pi) \rceil + 1 = \lceil 53.03 \rceil + 1 = 55`.
4. Cortes: :math:`\omega_{c1} = 0.3\pi`, :math:`\omega_{c2} = 0.7\pi`.
5. **Verificação (decisão D2):** com M = 55, :math:`\max|H|` na rejeição é
   ≈ 0.0059 > 0.005, então a estimativa **não atende**. A busca nos comprimentos
   ímpares (rejeita-faixa exige Tipo 1), com β fixo, encontra o menor M que
   atende: **M = 57**.

Execução: ``python -m exercises.p7_13`` (a partir da raiz do projeto).
"""

from __future__ import annotations

import matplotlib
import numpy as np

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, LengthSearch, admissible_lengths, design_kaiser, search_min_length
from dsp_av2.specs import P7_13 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_tolerance


def estimate() -> FIRDesign:
    """Passos 1–4: projeto com o M estimado pela fórmula de Kaiser."""
    return design_kaiser(SPEC)


def search() -> LengthSearch:
    """Passo 5: menor M admissível que atende às tolerâncias (β fixo)."""
    M_est = estimate().M
    return search_min_length(
        build=lambda M: design_kaiser(SPEC, M=M),
        accept=lambda d: validate_fir_tolerance(d).passed,
        lengths=admissible_lengths(SPEC, M_est // 2, M_est + 30),
    )


def design() -> FIRDesign:
    return search().best


def validate(d: FIRDesign) -> ValidationReport:
    return validate_fir_tolerance(d)


def plot(d: FIRDesign):
    """Subplots de Hr(ω) e |H| em dB (pedidos), mais h(n)."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    H = analysis.frequency_response(d.h, w)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.abs(H))
    return plotting.fir_tolerance_figure(
        d.h, w, analysis.amplitude_response(d.h, w), db, SPEC.bands, d.params["As_design_dB"],
        suptitle=f"P7.13 — Rejeita-faixa, janela de Kaiser (β = {d.params['beta']:.4f}), M = {d.M}",
    )


def metrics(d: FIRDesign, report: ValidationReport, s: LengthSearch) -> dict:
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "As_design_dB": d.params["As_design_dB"],
        "beta": d.params["beta"],
        "transition_width_pi": d.transition_width,
        "cutoffs_pi": d.cutoffs,
        "M_estimate": estimate().M,
        "M_min_verified": d.M,
        "search": s.tried,
        "validation": report.to_dict(),
        "h": d.h,
    }


def main() -> None:
    matplotlib.use("Agg")
    e = estimate()
    print(f"A_s = {e.params['As_design_dB']:.4f} dB; β = {e.params['beta']:.4f}; "
          f"Δω = {e.transition_width:g}π; ωc = {e.cutoffs}")
    print(f"Estimativa de Kaiser: M = {e.M}")
    print(validate(e))
    s = search()
    print("\nBusca (M, atende?):", s.tried[-4:])
    d = s.best
    report = validate(d)
    print(report)
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(d), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(d, report, s)))


if __name__ == "__main__":
    main()
