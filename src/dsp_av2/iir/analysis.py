r"""Análise de protótipos analógicos: resposta em frequência, fase, atraso de
grupo e resposta ao impulso.

Tudo é calculado a partir da forma zpk, que é numericamente robusta. A forma
tf (B, A) fica mal condicionada em ordens altas: no P8.13, :math:`a_0 \approx
4.4\times10^{43}`.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from .prototypes import AnalogPrototype


def frequency_response(proto: AnalogPrototype, w: np.ndarray) -> np.ndarray:
    r""":math:`H(j\Omega)` em ``w`` (rad/s)."""
    _, H = signal.freqs_zpk(proto.z, proto.p, proto.k, worN=w)
    return H


def magnitude_db(H: np.ndarray) -> np.ndarray:
    r""":math:`20\log_{10}|H(j\Omega)|`, **sem** normalizar pelo máximo.

    Nos protótipos analógicos a referência é o ganho absoluto: o Butterworth
    tem :math:`|H(j0)| = 1` e o Chebyshev-I de ordem par tem
    :math:`|H(j0)| = 1/\sqrt{1+\epsilon^2}`, com máximo 1 dentro da passagem.
    """
    with np.errstate(divide="ignore"):
        return 20 * np.log10(np.abs(H))


def group_delay(proto: AnalogPrototype, w: np.ndarray) -> np.ndarray:
    r"""Atraso de grupo analítico :math:`\tau(\Omega) = -d\phi/d\Omega`.

    Para :math:`H(s) = k\prod(s-z_i)/\prod(s-p_k)`, cada polo
    :math:`p = \sigma + j\omega_0` contribui com
    :math:`-\sigma / [\sigma^2 + (\Omega-\omega_0)^2]` e cada zero com o
    termo de sinal oposto:

    .. math::
        \tau(\Omega) = \sum_k \frac{-\sigma_k}{\sigma_k^2 + (\Omega - \omega_k)^2}
                     - \sum_i \frac{-\sigma_{z,i}}{\sigma_{z,i}^2 + (\Omega - \omega_{z,i})^2}
    """
    w = np.asarray(w, dtype=float)[:, None]

    def contrib(roots):
        if len(roots) == 0:
            return np.zeros(w.shape[0])
        s, o = roots.real[None, :], roots.imag[None, :]
        return np.sum(-s / (s**2 + (w - o) ** 2), axis=1)

    return contrib(np.asarray(proto.p)) - contrib(np.asarray(proto.z))


def impulse_time_span(proto: AnalogPrototype, decay_db: float = 60.0) -> float:
    r"""Duração para a envoltória mais lenta, :math:`e^{\sigma_{\max} t}`, cair ``decay_db``.

    :math:`t_{fim} = \frac{\ln 10^{\,\text{decay}/20}}{\min_k |\mathrm{Re}\,p_k|}`
    """
    slowest = np.min(np.abs(np.asarray(proto.p).real))
    return np.log(10 ** (decay_db / 20)) / slowest


def impulse_response(proto: AnalogPrototype, t: np.ndarray) -> np.ndarray:
    r"""h(t) por simulação (``scipy.signal.impulse``) com escala de frequência.

    Simular direto em zpk gera matrizes de estado complexas, e a forma tf
    original é mal condicionada (coeficientes até :math:`10^{43}`). Por isso o
    filtro é normalizado por :math:`\Omega_0 = \max|p_k|`:

    .. math::
        H(s) = H_n(s/\Omega_0) \;\Longrightarrow\; h(t) = \Omega_0\, h_n(\Omega_0 t)

    e :math:`H_n` tem polos de módulo ≤ 1, com coeficientes de ordem unitária.
    """
    p, z = np.asarray(proto.p), np.asarray(proto.z)
    w0 = np.max(np.abs(p))
    kn = proto.k * w0 ** (len(z) - len(p))
    b, a = signal.zpk2tf(z / w0, p / w0, kn)
    _, hn = signal.impulse((np.real(b), np.real(a)), T=w0 * np.asarray(t))
    return w0 * hn


def impulse_response_residues(proto: AnalogPrototype, t: np.ndarray) -> np.ndarray:
    r"""h(t) pela transformada inversa de Laplace (polos simples, sem zeros).

    .. math::
        h(t) = \sum_k r_k\, e^{p_k t}\,u(t),\qquad
        r_k = \frac{k}{\prod_{j\ne k}(p_k - p_j)}

    Usado como verificação independente de :func:`impulse_response`.
    """
    if len(proto.z):
        raise NotImplementedError("apenas protótipos só-polos")
    p = np.asarray(proto.p)
    r = np.array([proto.k / np.prod(pk - np.delete(p, i)) for i, pk in enumerate(p)])
    return np.real(np.exp(np.outer(t, p)) @ r)
