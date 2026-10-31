r"""Janelas e fórmulas de dimensionamento (DTA §3.1.C e §3.1.D).

Referência: Ingle & Proakis, Tabela 7.1 (p. 325, conforme DTA) e Seção 7.5
(Kaiser, pp. 338–342, conforme DTA).

Convenção (decisão D1 do plano): janelas **simétricas** de comprimento M com
zeros nas extremidades no caso Hann, i.e. :math:`w(n)` com argumento
:math:`2\pi n/(M-1)`, :math:`0 \le n \le M-1` — a mesma definição do ``hann(M)``
do MATLAB usado pelo livro.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import i0

# Casas decimais usadas antes do teto (ceil). Sem isso, erros de ponto
# flutuante transformam razões exatas em valores ligeiramente maiores:
# Δω = (0.5 − 0.4)π → 6.6π/Δω = 66.00000000000001 → ceil = 67 → M = 68 (o
# correto é 66 + 1 = 67). Arredondar a 9 casas elimina esse ruído.
_CEIL_DECIMALS = 9


def _ceil(x: float) -> int:
    return math.ceil(round(x, _CEIL_DECIMALS))


def _n(M: int) -> np.ndarray:
    if M < 2:
        raise ValueError(f"janelas exigem M ≥ 2 (recebido {M})")
    return np.arange(M)


def hann(M: int) -> np.ndarray:
    r""":math:`w(n) = 0.5 - 0.5\cos\left(\frac{2\pi n}{M-1}\right)`."""
    n = _n(M)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (M - 1))


def hamming(M: int) -> np.ndarray:
    r""":math:`w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{M-1}\right)`."""
    n = _n(M)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))


def blackman(M: int) -> np.ndarray:
    r""":math:`w(n) = 0.42 - 0.5\cos\left(\frac{2\pi n}{M-1}\right) + 0.08\cos\left(\frac{4\pi n}{M-1}\right)`."""
    n = _n(M)
    return (
        0.42
        - 0.5 * np.cos(2 * np.pi * n / (M - 1))
        + 0.08 * np.cos(4 * np.pi * n / (M - 1))
    )


def kaiser(M: int, beta: float) -> np.ndarray:
    r"""Janela de Kaiser.

    .. math::
        w(n) = \frac{I_0\left(\beta\sqrt{1-\left(\frac{2n}{M-1}-1\right)^2}\right)}{I_0(\beta)}
    """
    n = _n(M)
    arg = 1.0 - (2.0 * n / (M - 1) - 1.0) ** 2
    # Arredondamentos podem gerar -1e-16 nas extremidades.
    return i0(beta * np.sqrt(np.clip(arg, 0.0, None))) / i0(beta)


@dataclass(frozen=True)
class FixedWindow:
    """Linha da Tabela 7.1: largura de transição :math:`k\\pi/M` e A_s."""

    name: str
    transition_factor: float  # k em Δω = kπ/M
    min_stopband_db: float    # A_s nominal da tabela (dB); o valor medido pode ser menor
    function: Callable[[int], np.ndarray]


# Tabela 7.1 (DTA §3.1.C). Retangular e Bartlett omitidas: não aparecem no DTA
# e nenhuma atende às atenuações exigidas (≥ 40 dB) nos exercícios.
FIXED_WINDOWS: dict[str, FixedWindow] = {
    "hann": FixedWindow("hann", 6.2, 44.0, hann),
    "hamming": FixedWindow("hamming", 6.6, 53.0, hamming),
    "blackman": FixedWindow("blackman", 11.0, 74.0, blackman),
}


def fixed_window_length(name: str, transition_width: float) -> int:
    r"""Comprimento pela Tabela 7.1: :math:`M = \lceil k\pi/\Delta\omega \rceil + 1`.

    Args:
        name: ``"hann"``, ``"hamming"`` ou ``"blackman"``.
        transition_width: Δω em rad/amostra.
    """
    k = FIXED_WINDOWS[name].transition_factor
    return _ceil(k * np.pi / transition_width) + 1


def kaiser_beta(as_db: float) -> float:
    r"""Parâmetro β de Kaiser (DTA §3.1.D).

    .. math::
        \beta = \begin{cases}
            0.1102(A_s - 8.7), & A_s \ge 50 \\
            0.5842(A_s - 21)^{0.4} + 0.07886(A_s - 21), & 21 < A_s < 50 \\
            0, & A_s \le 21
        \end{cases}
    """
    if as_db >= 50:
        return 0.1102 * (as_db - 8.7)
    if as_db > 21:
        return 0.5842 * (as_db - 21) ** 0.4 + 0.07886 * (as_db - 21)
    return 0.0


def kaiser_length_estimate(as_db: float, transition_width: float) -> int:
    r"""Estimativa empírica de Kaiser (DTA §3.1.D).

    .. math::
        M \approx \left\lceil \frac{A_s - 7.95}{2.285\,\Delta\omega} \right\rceil + 1

    É apenas uma **estimativa**: o comprimento final deve ser confirmado
    medindo-se a resposta em frequência (decisão D2 do plano).
    """
    return _ceil((as_db - 7.95) / (2.285 * transition_width)) + 1
