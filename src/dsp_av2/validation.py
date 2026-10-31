"""Validação dos projetos contra a **especificação do enunciado**.

Os números de resultado do DTA não são usados como oráculo (plano, §0 e §4):
cada verificação compara um valor medido com o limite exigido pelo enunciado.
Este módulo não projeta nem desenha nada.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .fir import analysis
from .fir.design import FIRDesign
from .iir import analysis as ianalysis
from .iir.prototypes import OF_ORDER, AnalogPrototype
from .specs import AnalogLowpassSpec

# Tolerância (dB) para limites atingidos com igualdade por construção, como
# R_p em Ω_p no Butterworth (Ω_c é escolhido para isso).
DB_TOL = 1e-9


@dataclass(frozen=True)
class Check:
    name: str
    measured: float
    limit: float
    relation: str  # "<=" ou ">="
    passed: bool

    @property
    def margin(self) -> float:
        """Folga positiva quando a verificação passa (mesma unidade do limite)."""
        return self.limit - self.measured if self.relation == "<=" else self.measured - self.limit


def _check(name: str, measured: float, relation: str, limit: float, tol: float = 0.0) -> Check:
    """``tol`` absorve ruído de ponto flutuante quando o projeto atende o limite com igualdade."""
    ok = measured <= limit + tol if relation == "<=" else measured >= limit - tol
    return Check(name, float(measured), float(limit), relation, bool(ok))


@dataclass(frozen=True)
class ValidationReport:
    exercise: str
    checks: tuple[Check, ...]
    linear_phase_type: int | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "passed": self.passed,
            **({"linear_phase_type": self.linear_phase_type} if self.linear_phase_type else {}),
            "checks": [asdict(c) | {"margin": c.margin} for c in self.checks],
        }

    def __str__(self) -> str:
        head = f"Validação {self.exercise}"
        if self.linear_phase_type:
            head += f" (Tipo {self.linear_phase_type} de fase linear)"
        lines = [head]
        for c in self.checks:
            mark = "OK  " if c.passed else "FALHA"
            lines.append(
                f"  [{mark}] {c.name}: {c.measured:.4f} {c.relation} {c.limit:.4f}"
                f"  (folga {round(c.margin, 9) + 0.0:+.4f})"  # + 0.0 evita "−0.0000"
            )
        lines.append("  Resultado: " + ("ATENDE à especificação" if self.passed else "NÃO ATENDE"))
        return "\n".join(lines)


def validate_fir_db(design: FIRDesign) -> ValidationReport:
    """Verifica R_p e A_s (dB) em duas grades que contêm as bordas exatas.

    - grade do livro (501 pontos, passo π/500), para comparação com o texto;
    - grade densa (20001 pontos), que é a que decide a conformidade.
    """
    spec = design.spec
    ptype = analysis.linear_phase_type(design.h)
    checks = []
    for label, n in (("livro", analysis.BOOK_GRID_POINTS), ("densa", analysis.DENSE_GRID_POINTS)):
        w = analysis.frequency_grid(spec.edges, n)
        rp, as_ = analysis.measure_rp_as(design.h, spec.passbands, spec.stopbands, w)
        checks.append(_check(f"Rp [dB] (grade {label})", rp, "<=", spec.rp_db))
        checks.append(_check(f"As [dB] (grade {label})", as_, ">=", spec.as_db))
    return ValidationReport(spec.exercise, tuple(checks), ptype)


def validate_analog_lowpass(spec: AnalogLowpassSpec, proto: AnalogPrototype) -> ValidationReport:
    r"""Verifica um protótipo analógico passa-baixas contra o enunciado.

    - perda máxima em :math:`0 \le \Omega \le \Omega_p` ≤ R_p (inclui Ω_p);
    - atenuação mínima em :math:`\Omega_s \le \Omega \le 100\,\Omega_s` ≥ A_s;
    - minimalidade: o mesmo projeto com ordem N − 1 **não** atinge A_s em Ω_s.

    Perda/atenuação são medidas em relação a :math:`\max|H| = 1`.
    """
    wpass = np.unique(np.concatenate([np.linspace(0, spec.wp, 20001), [spec.wp]]))
    wstop = np.unique(np.concatenate([np.geomspace(spec.ws, 100 * spec.ws, 20001), [spec.ws]]))
    loss_pass = -ianalysis.magnitude_db(ianalysis.frequency_response(proto, wpass))
    att_stop = -ianalysis.magnitude_db(ianalysis.frequency_response(proto, wstop))
    checks = [
        _check("Perda máx. na passagem [dB]", loss_pass.max(), "<=", spec.rp_db, DB_TOL),
        _check("Atenuação mín. na rejeição [dB]", att_stop.min(), ">=", spec.as_db, DB_TOL),
    ]
    if proto.N > 1:
        lower = OF_ORDER[proto.approximation](proto.N - 1, spec.wp, spec.rp_db)
        att_lower = -ianalysis.magnitude_db(ianalysis.frequency_response(lower, np.array([spec.ws])))[0]
        checks.append(_check(f"Atenuação em Ωs com N−1 = {proto.N - 1} [dB] (deve falhar)",
                             att_lower, "<=", spec.as_db))
    return ValidationReport(spec.exercise, tuple(checks))


def validate_fir_tolerance(design: FIRDesign, n_points: int | None = None) -> ValidationReport:
    """Verifica |H| contra G_k ± δ_k em cada faixa (grade densa com as bordas)."""
    spec = design.spec
    n = n_points or analysis.DENSE_GRID_POINTS
    w = analysis.frequency_grid(spec.edges, n)
    devs = analysis.band_deviations(design.h, spec.bands, w)
    checks = tuple(
        _check(f"||H| − {b.gain:g}| em [{b.lo:g}π, {b.hi:g}π]", dev, "<=", b.delta)
        for b, dev in zip(spec.bands, devs)
    )
    return ValidationReport(spec.exercise, checks, analysis.linear_phase_type(design.h))
