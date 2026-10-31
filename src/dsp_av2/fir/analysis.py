r"""Análise de filtros FIR: resposta em frequência, amplitude e medições.

Referência: Ingle & Proakis, Seção 7.2 (tipos de fase linear, :math:`H_r`) e
a função ``freqz_m`` do livro (resposta em dB normalizada pelo máximo, em uma
grade de 501 pontos em [0, π]).

Regra de grade (decisão D8 do plano): toda grade usada para verificar
especificações **contém as frequências de borda exatas**. Grades que pulam as
bordas (ex.: 2048 pontos) superestimam A_s.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BOOK_GRID_POINTS = 501
DENSE_GRID_POINTS = 20001


def frequency_grid(edges_pi=(), n_points: int = BOOK_GRID_POINTS) -> np.ndarray:
    """Grade uniforme em [0, π] acrescida das bordas (dadas em unidades de π).

    Com ``n_points = 501`` o passo é π/500, o mesmo da ``freqz_m`` do livro.
    """
    w = np.linspace(0.0, np.pi, n_points)
    w_edges = np.asarray(edges_pi, dtype=float) * np.pi
    return np.unique(np.concatenate([w, w_edges]))


def frequency_response(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    r""":math:`H(e^{j\omega}) = \sum_n h(n) e^{-j\omega n}` avaliada em ``w``."""
    _, H = signal.freqz(h, 1, worN=w)
    return H


def magnitude_db(H: np.ndarray) -> np.ndarray:
    r"""Magnitude em dB relativa ao máximo (convenção ``freqz_m`` do livro).

    .. math:: |H|_{dB} = 20\log_{10}\frac{|H(e^{j\omega})|}{\max|H|}
    """
    mag = np.abs(H)
    with np.errstate(divide="ignore"):
        return 20 * np.log10(mag / mag.max())


def linear_phase_type(h: np.ndarray, atol: float = 1e-12) -> int:
    """Classifica h(n) nos Tipos 1–4 de fase linear (DTA §3.1.B).

    Tipo 1: simétrico, M ímpar · Tipo 2: simétrico, M par ·
    Tipo 3: antissimétrico, M ímpar · Tipo 4: antissimétrico, M par.
    Lança ``ValueError`` se h(n) não tiver simetria.
    """
    odd = len(h) % 2 == 1
    if np.allclose(h, h[::-1], rtol=0, atol=atol):
        return 1 if odd else 2
    if np.allclose(h, -h[::-1], rtol=0, atol=atol):
        return 3 if odd else 4
    raise ValueError("h(n) não é simétrica nem antissimétrica: fase não linear")


def amplitude_response(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    r"""Resposta de amplitude real :math:`H_r(\omega)` de um FIR simétrico.

    Para os Tipos 1 e 2, :math:`H(e^{j\omega}) = H_r(\omega)\,e^{-j\omega\alpha}`
    com :math:`\alpha=(M-1)/2`, logo
    :math:`H_r(\omega) = \mathrm{Re}\{H(e^{j\omega})\,e^{j\omega\alpha}\}`.
    """
    ptype = linear_phase_type(h)
    if ptype not in (1, 2):
        raise ValueError("H_r implementada apenas para Tipos 1 e 2 (h simétrica)")
    alpha = (len(h) - 1) / 2
    rotated = frequency_response(h, w) * np.exp(1j * w * alpha)
    # A parte imaginária é ruído numérico; verificação explícita:
    if np.max(np.abs(rotated.imag)) > 1e-9 * max(1.0, np.max(np.abs(rotated))):
        raise ArithmeticError("parte imaginária não desprezível em H_r")
    return rotated.real


def _band_mask(w: np.ndarray, lo_pi: float, hi_pi: float) -> np.ndarray:
    tol = 1e-12
    return (w >= lo_pi * np.pi - tol) & (w <= hi_pi * np.pi + tol)


def measure_rp_as(h: np.ndarray, passbands, stopbands, w: np.ndarray) -> tuple[float, float]:
    r"""Mede R_p e A_s como no livro (Exemplos da Seção 7.4).

    Com :math:`|H|_{dB}` normalizada pelo máximo:

    - :math:`R_p = -\min_{\omega \in \text{passagem}} |H|_{dB}`
    - :math:`A_s = -\max_{\omega \in \text{rejeição}} |H|_{dB}`

    Args:
        passbands, stopbands: sequências de objetos com ``lo``/``hi`` em
            unidades de π (ex.: :class:`dsp_av2.specs.Band`).
        w: grade em rad/amostra (deve conter as bordas).
    """
    db = magnitude_db(frequency_response(h, w))
    pmask = np.zeros_like(w, dtype=bool)
    for b in passbands:
        pmask |= _band_mask(w, b.lo, b.hi)
    smask = np.zeros_like(w, dtype=bool)
    for b in stopbands:
        smask |= _band_mask(w, b.lo, b.hi)
    return float(-db[pmask].min()), float(-db[smask].max())


def band_deviations(h: np.ndarray, bands, w: np.ndarray) -> list[float]:
    r"""Desvio máximo :math:`\max_{\omega \in B_k} \big||H(e^{j\omega})| - G_k\big|` por faixa.

    Usa a magnitude **absoluta** (sem normalizar), pois as tolerâncias dos
    enunciados P7.13–P7.15 são absolutas.
    """
    mag = np.abs(frequency_response(h, w))
    return [float(np.max(np.abs(mag[_band_mask(w, b.lo, b.hi)] - b.gain))) for b in bands]
