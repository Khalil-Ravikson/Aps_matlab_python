r"""P7.15 — Filtro em escada (staircase) com janela de Kaiser; comparação com o Exemplo 7.26.

Enunciado e especificações: :data:`dsp_av2.specs.P7_15` (DTA §2.1, item 7).

Roteiro de solução (DTA §4, Resolução P7.15, com as decisões D2/D4/D6/D7 do plano):

1. :math:`\delta = \min(0.01, 0.005, 0.001) = 0.001` → :math:`A_s = 60` dB
   (critério do DTA; ver D7 sobre ser conservador).
2. :math:`A_s \ge 50` → :math:`\beta = 0.1102(60 - 8.7) = 5.6533`.
3. :math:`\Delta\omega = 0.1\pi` (as duas transições); estimativa
   :math:`M = \lceil 52.05/(2.285 \cdot 0.1\pi) \rceil + 1 = \lceil 72.51 \rceil + 1 = 74`.
   O ganho em ω = π é 0, então M par é admissível (o DTA força 75).
4. Resposta ideal em degraus, cortes em 0.35π e 0.75π:
   :math:`h_d = 1\cdot h_{LP,1} + 0.5(h_{LP,2} - h_{LP,1}) = 0.5\,h_{LP}(0.35\pi) + 0.5\,h_{LP}(0.75\pi)`.
5. Busca do menor M (β fixo, todos os comprimentos): **M = 72**.
6. Comparação: o Exemplo 7.26 (Parks-McClellan) obtém M = 49. Aqui o valor é
   reproduzido de forma independente com ``scipy.signal.remez``, com pesos
   :math:`W_k = \delta_{\min}/\delta_k`.

Execução: ``python -m exercises.p7_15`` (a partir da raiz do projeto).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
from scipy import signal

from dsp_av2 import plotting, reporting
from dsp_av2.fir import analysis
from dsp_av2.fir.design import FIRDesign, LengthSearch, admissible_lengths, design_kaiser, search_min_length
from dsp_av2.specs import EXAMPLE_7_26_M, P7_15 as SPEC
from dsp_av2.validation import ValidationReport, validate_fir_tolerance


def estimate() -> FIRDesign:
    """Passos 1–4: projeto com o M estimado pela fórmula de Kaiser."""
    return design_kaiser(SPEC)


def search() -> LengthSearch:
    """Passo 5: menor M que atende às três tolerâncias (β fixo)."""
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


# --- Comparação com o Exemplo 7.26 (Parks-McClellan) ------------------------

def remez_weights() -> list[float]:
    r"""Pesos do equiripple: :math:`W_k = \delta_{\min}/\delta_k`, igualando os erros ponderados."""
    dmin = min(b.delta for b in SPEC.bands)
    return [dmin / b.delta for b in SPEC.bands]


def remez_design(M: int) -> np.ndarray:
    """Filtro equiripple de comprimento M (equivalente ao ``firpm`` do Exemplo 7.26)."""
    edges = [x for b in SPEC.bands for x in (b.lo, b.hi)]
    return signal.remez(M, edges, [b.gain for b in SPEC.bands], weight=remez_weights(), fs=2)


@dataclass(frozen=True)
class RemezResult:
    M_min: int
    h: np.ndarray
    tried: tuple[tuple[int, bool, float], ...]  # (M, atende?, A_s da banda 3 em dB)


def remez_search(lo: int = 40, hi: int = 60) -> RemezResult:
    """Menor M equiripple que atende às três tolerâncias."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    tried = []
    for M in range(lo, hi + 1):
        h = remez_design(M)
        devs = analysis.band_deviations(h, SPEC.bands, w)
        ok = all(dv <= b.delta for dv, b in zip(devs, SPEC.bands))
        tried.append((M, ok, -20 * np.log10(devs[2])))
        if ok:
            return RemezResult(M, h, tuple(tried))
    raise RuntimeError("nenhum comprimento equiripple atende na faixa")


def plot(d: FIRDesign):
    """Magnitude em dB (pedida), com h(n) e H_r(ω) como contexto."""
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    H = analysis.frequency_response(d.h, w)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.abs(H))
    return plotting.fir_tolerance_figure(
        d.h, w, analysis.amplitude_response(d.h, w), db, SPEC.bands, d.params["As_design_dB"],
        suptitle=f"P7.15 — Filtro em escada, janela de Kaiser (β = {d.params['beta']:.4f}), M = {d.M}",
    )


def metrics(d: FIRDesign, report: ValidationReport, s: LengthSearch, r: RemezResult) -> dict:
    return {
        "exercise": SPEC.exercise,
        "source": SPEC.source,
        "As_design_dB": d.params["As_design_dB"],
        "beta": d.params["beta"],
        "cutoffs_pi": d.cutoffs,
        "M_estimate": estimate().M,
        "M_min_verified": d.M,
        "search": s.tried,
        "example_7_26_M_book": EXAMPLE_7_26_M,
        "remez_M_min": r.M_min,
        "remez_search": r.tried,
        "length_ratio_kaiser_over_pm": d.M / EXAMPLE_7_26_M,
        "validation": report.to_dict(),
        "h": d.h,
    }


def main() -> None:
    matplotlib.use("Agg")
    e = estimate()
    print(f"A_s = {e.params['As_design_dB']:.2f} dB; β = {e.params['beta']:.5f}; "
          f"ωc = {e.cutoffs}; estimativa M = {e.M}")
    s = search()
    d = s.best
    report = validate(d)
    print("Busca Kaiser (M, atende?):", s.tried[-4:])
    print(report)
    r = remez_search()
    print(f"\nEquiripple (remez): M mínimo = {r.M_min}; últimos: "
          f"{[(M, ok, round(a, 2)) for M, ok, a in r.tried[-3:]]}")
    print(f"Exemplo 7.26 (livro): M = {EXAMPLE_7_26_M}")
    print(f"Kaiser/Parks-McClellan = {d.M}/{EXAMPLE_7_26_M} = {d.M / EXAMPLE_7_26_M:.3f}")
    print("Figura:", reporting.save_figure(SPEC.exercise, plot(d), "overview"))
    print("Métricas:", reporting.save_metrics(SPEC.exercise, metrics(d, report, s, r)))


if __name__ == "__main__":
    main()
