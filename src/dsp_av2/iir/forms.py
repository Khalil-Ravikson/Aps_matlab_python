r"""Formas da função de sistema analógica: racional e cascata.

- **Racional**: :math:`H(s) = B(s)/A(s)`, polinômios em potências decrescentes
  de s (convenção de ``scipy.signal`` e do MATLAB).
- **Cascata**: :math:`H(s) = G \prod_k H_k(s)`, com seções de 2ª ordem para
  pares conjugados e de 1ª ordem para polos reais. Cada seção é normalizada
  para ganho DC unitário, :math:`H_k(0) = 1`, como nas resoluções do DTA
  (ex.: :math:`\Omega_c^2/(s^2 + 2\Omega_c\cos\theta_k\,s + \Omega_c^2)`).

``scipy.signal.zpk2sos`` não é usado porque foi feito para sistemas
discretos (polinômios em :math:`z^{-1}`); em s o agrupamento é explícito.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .prototypes import AnalogPrototype


def _real(c: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Descarta a parte imaginária residual (polos conjugados → coeficientes reais)."""
    c = np.atleast_1d(c)
    if np.iscomplexobj(c):
        if np.max(np.abs(c.imag)) > rtol * np.max(np.abs(c.real)):
            raise ValueError("coeficientes com parte imaginária não desprezível")
        c = c.real
    return c


def rational(proto: AnalogPrototype) -> tuple[np.ndarray, np.ndarray]:
    """Coeficientes (B, A) da forma racional, com A mônico."""
    b, a = signal.zpk2tf(proto.z, proto.p, proto.k)
    return _real(b), _real(a)


@dataclass(frozen=True)
class Section:
    """Seção da cascata: H_k(s) = num(s) / den(s), coeficientes reais."""

    num: np.ndarray
    den: np.ndarray

    @property
    def order(self) -> int:
        return len(self.den) - 1


@dataclass(frozen=True)
class Cascade:
    gain: float
    sections: tuple[Section, ...]


def cascade(proto: AnalogPrototype, imag_tol: float = 1e-9) -> Cascade:
    r"""Agrupa os polos em seções reais de 1ª/2ª ordem (protótipos só-polos).

    Par conjugado :math:`p, p^*`:
    :math:`H_k(s) = \frac{|p|^2}{s^2 - 2\,\mathrm{Re}(p)\,s + |p|^2}`;
    polo real :math:`p`: :math:`H_k(s) = \frac{-p}{s - p}`.
    O ganho global é :math:`G = k / \prod_k \text{num}_k`.
    Seções ordenadas do polo de menor para o de maior fator de qualidade.
    """
    if len(proto.z):
        raise NotImplementedError("cascata implementada apenas para protótipos só-polos")
    p = np.asarray(proto.p)
    scale = max(1.0, np.max(np.abs(p)))
    upper = p[p.imag > imag_tol * scale]
    real = p[np.abs(p.imag) <= imag_tol * scale].real
    if 2 * len(upper) + len(real) != len(p):
        raise ValueError("polos complexos sem conjugado correspondente")

    sections = []
    for r in sorted(real):
        sections.append(Section(np.array([-r]), np.array([1.0, -r])))
    # Q = |p| / (−2 Re p): seções mais amortecidas primeiro
    for q in sorted(upper, key=lambda q: abs(q) / (-2 * q.real)):
        m2 = abs(q) ** 2
        sections.append(Section(np.array([m2]), np.array([1.0, -2 * q.real, m2])))

    gain = proto.k / np.prod([s.num[0] for s in sections])
    return Cascade(float(np.real(gain)), tuple(sections))


def cascade_to_rational(c: Cascade) -> tuple[np.ndarray, np.ndarray]:
    """Recompõe B(s)/A(s) multiplicando as seções (usado para validar a cascata)."""
    b, a = np.array([c.gain]), np.array([1.0])
    for s in c.sections:
        b = np.polymul(b, s.num)
        a = np.polymul(a, s.den)
    return b, a


def poly_to_latex(c: np.ndarray, var: str = "s", digits: int = 7) -> str:
    """Polinômio em potências decrescentes → LaTeX (ex.: ``s^2 + 2.5\\times10^{3}s + …``)."""
    def num(x: float) -> str:
        m, e = f"{x:.{digits - 1}e}".split("e")
        e = int(e)
        return m if e == 0 else rf"{m}\times10^{{{e}}}"

    n = len(c) - 1
    terms = []
    for i, x in enumerate(np.real(c)):
        if x == 0:
            continue
        p = n - i
        coef = "" if (x == 1 and p > 0) else num(abs(x))
        power = "" if p == 0 else (var if p == 1 else f"{var}^{{{p}}}")
        sign = "-" if x < 0 else "+"
        terms.append((sign, f"{coef}{power}"))
    first_sign, first = terms[0]
    out = ("-" if first_sign == "-" else "") + first
    for sign, t in terms[1:]:
        out += f" {sign} {t}"
    return out


def cascade_to_latex(c: Cascade, var: str = "s", digits: int = 7) -> str:
    """Cascata → LaTeX: ``G \\cdot \\frac{num_1}{den_1} \\cdot \\frac{num_2}{den_2} …``."""
    parts = [] if np.isclose(c.gain, 1.0, rtol=1e-12) else [f"{c.gain:.{digits - 2}f}"]
    for s in c.sections:
        parts.append(rf"\dfrac{{{poly_to_latex(s.num, var, digits)}}}{{{poly_to_latex(s.den, var, digits)}}}")
    return r" \cdot ".join(parts)
