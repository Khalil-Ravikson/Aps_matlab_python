r"""Projeto de protótipos analógicos passa-baixas pelas fórmulas do livro.

Referência: DTA §3.2; Ingle & Proakis, Seções 8.2.1 (Butterworth, funções
``afd_butt``/``u_buttap``) e 8.2.2 (Chebyshev-I, ``afd_chb1``/``u_chb1ap``).

Os polos são calculados explicitamente pelas equações do texto. As rotinas do
SciPy (``buttord``, ``butter``, ``cheb1ord``, ``cheby1`` com ``analog=True``)
servem de verificação cruzada nos testes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_CEIL_DECIMALS = 9  # mesmo cuidado de windows.py contra ruído de ponto flutuante


def _ceil(x: float) -> int:
    return math.ceil(round(x, _CEIL_DECIMALS))


@dataclass(frozen=True)
class AnalogPrototype:
    """Protótipo analógico em forma zpk: H(s) = k ∏(s − z) / ∏(s − p)."""

    approximation: str
    N: int
    z: np.ndarray
    p: np.ndarray
    k: float
    params: dict  # parâmetros intermediários (Ω_c, ε, ...) para o relatório


# ---------------------------------------------------------------------------
# Butterworth
# ---------------------------------------------------------------------------

def butter_order(wp: float, ws: float, rp_db: float, as_db: float) -> tuple[int, float]:
    r"""Ordem mínima de Butterworth.

    .. math::
        N = \left\lceil \frac{\log_{10}\left[(10^{R_p/10}-1)/(10^{A_s/10}-1)\right]}
                              {2\log_{10}(\Omega_p/\Omega_s)} \right\rceil

    Returns:
        ``(N, N_real)``, a ordem inteira e o valor antes do teto.
    """
    num = math.log10((10 ** (rp_db / 10) - 1) / (10 ** (as_db / 10) - 1))
    n_real = num / (2 * math.log10(wp / ws))
    return _ceil(n_real), n_real


def butter_cutoff(wp: float, rp_db: float, N: int) -> float:
    r""":math:`\Omega_c = \Omega_p / (10^{R_p/10}-1)^{1/(2N)}` (atende R_p exatamente em Ω_p)."""
    return wp / (10 ** (rp_db / 10) - 1) ** (1 / (2 * N))


def butter_poles(N: int, wc: float) -> np.ndarray:
    r"""Polos no semiplano esquerdo.

    .. math:: p_k = \Omega_c\, e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{2N}\right]},\quad k = 0,\dots,N-1
    """
    k = np.arange(N)
    return wc * np.exp(1j * (np.pi / 2 + (2 * k + 1) * np.pi / (2 * N)))


def butterworth_of_order(N: int, wp: float, rp_db: float) -> AnalogPrototype:
    r"""Butterworth de ordem N com R_p exato em Ω_p: :math:`H(s) = \Omega_c^N / \prod_k (s - p_k)`."""
    wc = butter_cutoff(wp, rp_db, N)
    return AnalogPrototype(
        approximation="butterworth",
        N=N,
        z=np.array([]),
        p=butter_poles(N, wc),
        k=wc**N,
        params={"Omega_c": wc},
    )


def design_butterworth(wp: float, ws: float, rp_db: float, as_db: float) -> AnalogPrototype:
    """Butterworth passa-baixas de ordem mínima (``afd_butt`` do livro)."""
    N, n_real = butter_order(wp, ws, rp_db, as_db)
    proto = butterworth_of_order(N, wp, rp_db)
    proto.params["N_real"] = n_real
    return proto


# ---------------------------------------------------------------------------
# Chebyshev tipo I
# ---------------------------------------------------------------------------

def cheb1_order(wp: float, ws: float, rp_db: float, as_db: float) -> tuple[int, float]:
    r"""Ordem mínima de Chebyshev-I.

    .. math::
        \epsilon = \sqrt{10^{R_p/10}-1},\quad
        g = \frac{\sqrt{10^{A_s/10}-1}}{\epsilon},\quad
        N = \left\lceil \frac{\cosh^{-1} g}{\cosh^{-1}(\Omega_s/\Omega_p)} \right\rceil
    """
    eps = math.sqrt(10 ** (rp_db / 10) - 1)
    g = math.sqrt(10 ** (as_db / 10) - 1) / eps
    n_real = math.acosh(g) / math.acosh(ws / wp)
    return _ceil(n_real), n_real


def cheb1_poles(N: int, wp: float, rp_db: float) -> np.ndarray:
    r"""Polos de Chebyshev-I sobre a elipse (DTA §3.2.B).

    .. math::
        \alpha = \tfrac{1}{\epsilon} + \sqrt{1 + \tfrac{1}{\epsilon^2}},\quad
        a = \tfrac12(\alpha^{1/N} - \alpha^{-1/N}),\quad
        b = \tfrac12(\alpha^{1/N} + \alpha^{-1/N})

    .. math::
        \sigma_k = -a\,\Omega_p \sin\frac{(2k+1)\pi}{2N},\qquad
        \Omega_k = b\,\Omega_p \cos\frac{(2k+1)\pi}{2N}
    """
    eps = math.sqrt(10 ** (rp_db / 10) - 1)
    alpha = 1 / eps + math.sqrt(1 + 1 / eps**2)
    a = 0.5 * (alpha ** (1 / N) - alpha ** (-1 / N))
    b = 0.5 * (alpha ** (1 / N) + alpha ** (-1 / N))
    theta = (2 * np.arange(N) + 1) * np.pi / (2 * N)
    return -a * wp * np.sin(theta) + 1j * b * wp * np.cos(theta)


def chebyshev1_of_order(N: int, wp: float, rp_db: float) -> AnalogPrototype:
    r"""Chebyshev-I de ordem N.

    O ganho é escolhido para que :math:`|H(j0)| = 1` (N ímpar) ou
    :math:`|H(j0)| = 1/\sqrt{1+\epsilon^2}` (N par):

    .. math:: k = \prod_k (-p_k) \cdot \begin{cases} 1, & N \text{ ímpar} \\ 1/\sqrt{1+\epsilon^2}, & N \text{ par} \end{cases}
    """
    eps = math.sqrt(10 ** (rp_db / 10) - 1)
    p = cheb1_poles(N, wp, rp_db)
    dc = 1.0 if N % 2 else 1 / math.sqrt(1 + eps**2)
    k = float(np.real(np.prod(-p))) * dc
    return AnalogPrototype(
        approximation="chebyshev1",
        N=N,
        z=np.array([]),
        p=p,
        k=k,
        params={"epsilon": eps, "dc_gain": dc},
    )


def design_chebyshev1(wp: float, ws: float, rp_db: float, as_db: float) -> AnalogPrototype:
    """Chebyshev-I passa-baixas de ordem mínima (``afd_chb1`` do livro)."""
    N, n_real = cheb1_order(wp, ws, rp_db, as_db)
    proto = chebyshev1_of_order(N, wp, rp_db)
    proto.params["N_real"] = n_real
    return proto


OF_ORDER = {"butterworth": butterworth_of_order, "chebyshev1": chebyshev1_of_order}
DESIGN = {"butterworth": design_butterworth, "chebyshev1": design_chebyshev1}
