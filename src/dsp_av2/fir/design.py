r"""Projeto FIR pelo método da janela: :math:`h(n) = h_d(n)\,w(n)`.

Referência: DTA §3.1 (passos A–E); Ingle & Proakis, Seção 7.4.

Procedimento (janelas fixas):

1. :math:`\Delta\omega = \min(|\omega_p - \omega_s|)` sobre todas as transições;
2. :math:`\omega_c = (\omega_p + \omega_s)/2` em cada transição;
3. :math:`M = \lceil k\pi/\Delta\omega \rceil + 1` (Tabela 7.1);
4. se o filtro exige ganho não nulo em ω = π (passa-altas, rejeita-faixa),
   M precisa ser ímpar (Tipo 1); caso M saia par, usa-se M + 1;
5. :math:`h(n) = h_d(n)\,w(n)`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from ..specs import FIRSpec, FIRToleranceSpec
from . import ideal
from .windows import FIXED_WINDOWS, fixed_window_length, kaiser, kaiser_beta, kaiser_length_estimate


@dataclass(frozen=True)
class Transition:
    """Faixa de transição entre duas bandas (unidades de π)."""

    lo: float
    hi: float

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def cutoff(self) -> float:
        return (self.lo + self.hi) / 2


@dataclass(frozen=True)
class FIRDesign:
    """Resultado de um projeto FIR, com todos os valores intermediários."""

    spec: FIRSpec | FIRToleranceSpec
    window: str
    transitions: tuple[Transition, ...]
    transition_width: float   # Δω em unidades de π
    cutoffs: tuple[float, ...]  # ω_c em unidades de π
    M_table: int              # M pela fórmula (Tabela 7.1 ou Kaiser), antes da paridade
    M: int                    # comprimento final
    hd: np.ndarray            # resposta ideal h_d(n)
    w: np.ndarray             # janela w(n)
    h: np.ndarray             # h(n) = h_d(n) w(n)
    params: dict = field(default_factory=dict)  # ex.: A_s de projeto e β (Kaiser)

    @property
    def alpha(self) -> float:
        return (self.M - 1) / 2


def transitions(spec: FIRSpec | FIRToleranceSpec) -> tuple[Transition, ...]:
    """Transições entre bandas consecutivas da especificação."""
    bands = spec.bands
    return tuple(Transition(a.hi, b.lo) for a, b in zip(bands, bands[1:]))


def requires_odd_length(spec: FIRSpec | FIRToleranceSpec) -> bool:
    """True se H(π) ≠ 0 é exigido: o Tipo 2 (M par) fica proibido (DTA §3.1.B)."""
    return spec.bands[-1].gain != 0.0


def ideal_response(spec: FIRSpec | FIRToleranceSpec, cutoffs_pi: tuple[float, ...], M: int) -> np.ndarray:
    """h_d(n) pela fórmula do DTA §3.1.E correspondente ao tipo do filtro."""
    wc = [c * np.pi for c in cutoffs_pi]
    if spec.filter_type == "lowpass":
        return ideal.ideal_lowpass(wc[0], M)
    if spec.filter_type == "highpass":
        return ideal.ideal_highpass(wc[0], M)
    if spec.filter_type == "bandpass":
        return ideal.ideal_bandpass(wc[0], wc[1], M)
    if spec.filter_type == "bandstop":
        return ideal.ideal_bandstop(wc[0], wc[1], M)
    if spec.filter_type == "multiband":
        return ideal.ideal_multiband(wc, [b.gain for b in spec.bands], M)
    raise ValueError(f"tipo de filtro não suportado: {spec.filter_type!r}")


def design_fixed_window(spec: FIRSpec, window: str, M: int | None = None) -> FIRDesign:
    """Projeta um FIR com janela fixa (Hann, Hamming ou Blackman).

    Args:
        spec: especificação do enunciado.
        window: nome da janela em :data:`FIXED_WINDOWS`.
        M: comprimento forçado. Se ``None``, usa a Tabela 7.1 (passo 3).
    """
    trans = transitions(spec)
    dw = min(t.width for t in trans)
    cutoffs = tuple(t.cutoff for t in trans)
    M_table = fixed_window_length(window, dw * np.pi)
    if M is None:
        M = M_table
        if requires_odd_length(spec) and M % 2 == 0:
            M += 1
    hd = ideal_response(spec, cutoffs, M)
    w = FIXED_WINDOWS[window].function(M)
    return FIRDesign(
        spec=spec,
        window=window,
        transitions=trans,
        transition_width=dw,
        cutoffs=cutoffs,
        M_table=M_table,
        M=M,
        hd=hd,
        w=w,
        h=hd * w,
    )


def design_kaiser(spec: FIRToleranceSpec, M: int | None = None) -> FIRDesign:
    r"""Projeto com janela de Kaiser (DTA §3.1.D).

    1. :math:`\delta = \min_k \delta_k`, :math:`A_s = -20\log_{10}\delta`;
    2. :math:`\beta` pela fórmula empírica de Kaiser;
    3. :math:`M \approx \lceil (A_s - 7.95)/(2.285\,\Delta\omega) \rceil + 1`
       (**estimativa**; o comprimento final vem de :func:`search_min_length`);
    4. M ímpar se o filtro exige ganho em ω = π.

    Args:
        M: comprimento forçado (usado na busca). ``None`` usa a estimativa.
    """
    trans = transitions(spec)
    dw = min(t.width for t in trans)
    cutoffs = tuple(t.cutoff for t in trans)
    delta = min(b.delta for b in spec.bands)
    as_db = -20 * np.log10(delta)
    beta = kaiser_beta(as_db)
    M_est = kaiser_length_estimate(as_db, dw * np.pi)
    if M is None:
        M = M_est
        if requires_odd_length(spec) and M % 2 == 0:
            M += 1
    hd = ideal_response(spec, cutoffs, M)
    w = kaiser(M, beta)
    return FIRDesign(
        spec=spec,
        window="kaiser",
        transitions=trans,
        transition_width=dw,
        cutoffs=cutoffs,
        M_table=M_est,
        M=M,
        hd=hd,
        w=w,
        h=hd * w,
        params={"delta_min": delta, "As_design_dB": as_db, "beta": beta},
    )


@dataclass(frozen=True)
class LengthSearch:
    """Resultado da busca do menor M verificado (decisão D2 do plano)."""

    best: FIRDesign
    tried: tuple[tuple[int, bool], ...]  # (M, atende?) em ordem crescente

    @property
    def M_min(self) -> int:
        return self.best.M


def search_min_length(build: Callable[[int], FIRDesign],
                      accept: Callable[[FIRDesign], bool],
                      lengths: Iterable[int]) -> LengthSearch:
    """Menor M de ``lengths`` (em ordem crescente) cujo projeto é aceito.

    Todos os comprimentos menores da lista são testados e registrados, de modo
    que a minimalidade fica documentada dentro da faixa varrida.
    """
    tried = []
    for M in sorted(lengths):
        d = build(M)
        ok = bool(accept(d))
        tried.append((M, ok))
        if ok:
            return LengthSearch(d, tuple(tried))
    raise RuntimeError("nenhum comprimento da faixa atende à especificação")


def admissible_lengths(spec, lo: int, hi: int) -> range:
    """Comprimentos admissíveis em [lo, hi]: só ímpares se H(π) ≠ 0 é exigido."""
    if requires_odd_length(spec):
        return range(lo | 1, hi + 1, 2)
    return range(lo, hi + 1)
