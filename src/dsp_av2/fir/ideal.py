r"""Respostas ao impulso ideais :math:`h_d(n)` com atraso :math:`\alpha=(M-1)/2`.

Referência: DTA §3.1.E; Ingle & Proakis, Seção 7.3 (função ``ideal_lp``).

Todas as funções recebem frequências de corte em **rad/amostra** e retornam um
vetor de comprimento ``M`` (índices ``n = 0, …, M-1``). As respostas são
construídas explicitamente com NumPy, conforme exigido pelos enunciados
("Do not use fir1").
"""

from __future__ import annotations

import numpy as np


def _check_length(M: int) -> None:
    if M < 1:
        raise ValueError(f"M deve ser positivo (recebido {M})")


def ideal_lowpass(wc: float, M: int) -> np.ndarray:
    r"""Passa-baixas ideal deslocado.

    .. math::
        h_d(n) = \frac{\sin[\omega_c (n-\alpha)]}{\pi (n-\alpha)},\quad
        n \ne \alpha; \qquad h_d(\alpha) = \frac{\omega_c}{\pi}
    """
    _check_length(M)
    alpha = (M - 1) / 2
    m = np.arange(M) - alpha
    # Evita divisão por zero no ponto central (só existe quando M é ímpar).
    m_safe = np.where(m == 0, 1.0, m)
    return np.where(m == 0, wc / np.pi, np.sin(wc * m_safe) / (np.pi * m_safe))


def delayed_impulse(M: int) -> np.ndarray:
    r""":math:`\delta(n-\alpha)`, o passa-tudo ideal com atraso :math:`\alpha`.

    Só é representável por uma sequência quando :math:`\alpha` é inteiro, isto
    é, quando M é ímpar. Para M par, :math:`\delta(n-\alpha)` não existe e os
    filtros que dependem dele (passa-altas, rejeita-faixa) não podem ser do
    Tipo 2 — coerente com o zero obrigatório do Tipo 2 em ω = π (DTA §3.1.B).
    """
    _check_length(M)
    if M % 2 == 0:
        raise ValueError(
            "δ(n−α) exige M ímpar: com M par (Tipo 2) H(π) = 0, o que é "
            "incompatível com passa-altas e rejeita-faixa (DTA §3.1.B)."
        )
    d = np.zeros(M)
    d[(M - 1) // 2] = 1.0
    return d


def ideal_highpass(wc: float, M: int) -> np.ndarray:
    r""":math:`h_d(n) = \delta(n-\alpha) - h_{LP}(n;\omega_c)`. Exige M ímpar."""
    return delayed_impulse(M) - ideal_lowpass(wc, M)


def ideal_bandpass(wc1: float, wc2: float, M: int) -> np.ndarray:
    r""":math:`h_d(n) = h_{LP}(n;\omega_{c2}) - h_{LP}(n;\omega_{c1})`."""
    if not 0 <= wc1 < wc2 <= np.pi:
        raise ValueError("é preciso 0 ≤ ωc1 < ωc2 ≤ π")
    return ideal_lowpass(wc2, M) - ideal_lowpass(wc1, M)


def ideal_bandstop(wc1: float, wc2: float, M: int) -> np.ndarray:
    r""":math:`h_d(n) = \delta(n-\alpha) - h_{BP}(n)`. Exige M ímpar."""
    return delayed_impulse(M) - ideal_bandpass(wc1, wc2, M)


def ideal_multiband(cutoffs: list[float], gains: list[float], M: int) -> np.ndarray:
    r"""Resposta ideal em degraus (multibanda), DTA §3.1.E.

    Para ganhos :math:`G_1, \dots, G_K` separados pelos cortes
    :math:`\omega_{c,1} < \dots < \omega_{c,K-1}`:

    .. math::
        h_d(n) = G_K\,\delta(n-\alpha)
                 + \sum_{k=1}^{K-1} (G_k - G_{k+1})\, h_{LP}(n;\omega_{c,k})

    que é a forma telescópica de
    :math:`G_1 h_{LP,1} + G_2 (h_{LP,2} - h_{LP,1}) + \dots` do DTA. O termo
    :math:`\delta(n-\alpha)` só aparece se :math:`G_K \ne 0` (ganho em ω = π).
    """
    if len(gains) != len(cutoffs) + 1:
        raise ValueError("são necessários len(cutoffs) + 1 ganhos")
    if any(b <= a for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError("os cortes devem ser estritamente crescentes")
    hd = np.zeros(M)
    for wc, g_left, g_right in zip(cutoffs, gains[:-1], gains[1:]):
        hd += (g_left - g_right) * ideal_lowpass(wc, M)
    if gains[-1] != 0.0:
        hd += gains[-1] * delayed_impulse(M)
    return hd
