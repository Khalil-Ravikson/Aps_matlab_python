"""Especificações literais dos exercícios.

Este módulo contém **apenas dados** transcritos dos enunciados (DTA §2), sem
nenhum cálculo derivado. Frequências digitais são armazenadas em múltiplos de
π (ex.: ``0.35`` significa 0.35π rad/amostra), exatamente como aparecem nos
enunciados; a conversão para rad/amostra é feita por quem consome a spec.

Novos exercícios são adicionados aqui à medida que são implementados
(ver plano de implementação, §7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    """Faixa de frequência [lo, hi] (em unidades de π) com ganho ideal."""

    lo: float
    hi: float
    gain: float

    @property
    def is_stopband(self) -> bool:
        return self.gain == 0.0


@dataclass(frozen=True)
class FIRSpec:
    """Especificação de um filtro FIR dada em decibéis (Rp, As).

    Attributes:
        exercise: identificador do exercício (ex.: ``"P7.9"``).
        filter_type: ``"lowpass"``, ``"highpass"``, ``"bandpass"``,
            ``"bandstop"``.
        bands: faixas de passagem e de rejeição, em ordem crescente de
            frequência.
        rp_db: ondulação máxima admitida na passagem, R_p (dB).
        as_db: atenuação mínima exigida na rejeição, A_s (dB).
        method: método de projeto exigido pelo enunciado.
        statement: enunciado original (para rastreabilidade).
        source: localização no DTA e no livro.
    """

    exercise: str
    filter_type: str
    bands: tuple[Band, ...]
    rp_db: float
    as_db: float
    method: str
    statement: str
    source: str

    @property
    def passbands(self) -> tuple[Band, ...]:
        return tuple(b for b in self.bands if not b.is_stopband)

    @property
    def stopbands(self) -> tuple[Band, ...]:
        return tuple(b for b in self.bands if b.is_stopband)

    @property
    def edges(self) -> tuple[float, ...]:
        """Todas as frequências de borda (unidades de π), sem 0 e 1."""
        e = sorted({x for b in self.bands for x in (b.lo, b.hi)} - {0.0, 1.0})
        return tuple(e)


@dataclass(frozen=True)
class ToleranceBand:
    r"""Faixa [lo, hi] (unidades de π) com ganho ideal e tolerância absoluta δ.

    A exigência é :math:`\big||H(e^{j\omega})| - G\big| \le \delta` na faixa.
    """

    lo: float
    hi: float
    gain: float
    delta: float

    @property
    def is_stopband(self) -> bool:
        return self.gain == 0.0


@dataclass(frozen=True)
class FIRToleranceSpec:
    """Especificação FIR por tolerâncias absolutas (P7.13–P7.15).

    Attributes:
        filter_type: ``"bandpass"``, ``"bandstop"`` ou ``"multiband"``.
        bands: faixas em ordem crescente, cada uma com ganho e δ.
    """

    exercise: str
    filter_type: str
    bands: tuple[ToleranceBand, ...]
    method: str
    statement: str
    source: str

    @property
    def passbands(self) -> tuple[ToleranceBand, ...]:
        return tuple(b for b in self.bands if not b.is_stopband)

    @property
    def stopbands(self) -> tuple[ToleranceBand, ...]:
        return tuple(b for b in self.bands if b.is_stopband)

    @property
    def edges(self) -> tuple[float, ...]:
        e = sorted({x for b in self.bands for x in (b.lo, b.hi)} - {0.0, 1.0})
        return tuple(e)


# ---------------------------------------------------------------------------
# Capítulo 7 — FIR
# ---------------------------------------------------------------------------

P7_9 = FIRSpec(
    exercise="P7.9",
    filter_type="bandpass",
    bands=(
        Band(0.0, 0.20, 0.0),   # rejeição inferior: ω ≤ 0.2π
        Band(0.35, 0.55, 1.0),  # passagem: 0.35π ≤ ω ≤ 0.55π
        Band(0.75, 1.0, 0.0),   # rejeição superior: ω ≥ 0.75π
    ),
    rp_db=0.25,
    as_db=40.0,
    method="window:hann",
    statement=(
        "Design a linear-phase bandpass filter using the Hann window design "
        "technique. The specifications are: lower stopband edge: 0.2π, "
        "As = 40 dB; upper stopband edge: 0.75π; lower passband edge: 0.35π, "
        "Rp = 0.25 dB; upper passband edge: 0.55π. Plot the impulse response "
        "and the magnitude response (in dB) of the designed filter. Do not "
        "use the fir1 function."
    ),
    source="DTA §2.1 item 1 e §4 (Resolução P7.9); Ingle & Proakis, p. 378",
)


# ---------------------------------------------------------------------------
# Capítulo 8 — protótipos analógicos IIR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalogLowpassSpec:
    """Especificação de um protótipo analógico passa-baixas.

    Attributes:
        approximation: ``"butterworth"`` ou ``"chebyshev1"``.
        wp, ws: bordas de passagem e de rejeição em **rad/s**. Enunciados em
            Hz são convertidos por Ω = 2πF na própria spec, com comentário.
        rp_db: ondulação máxima na passagem (dB).
        as_db: atenuação mínima na rejeição (dB).
        form: forma pedida da função de sistema, ``"rational"`` ou ``"cascade"``.
        plots: gráficos pedidos pelo enunciado.
        freq_unit: unidade usada nos gráficos (``"rad/s"`` ou ``"Hz"``, decisão D9).
    """

    exercise: str
    approximation: str
    wp: float
    ws: float
    rp_db: float
    as_db: float
    form: str
    plots: tuple[str, ...]
    freq_unit: str
    statement: str
    source: str


P8_12 = AnalogLowpassSpec(
    exercise="P8.12",
    approximation="butterworth",
    wp=500.0,    # rad/s
    ws=2000.0,   # rad/s
    rp_db=0.25,
    as_db=50.0,
    form="rational",
    plots=("magnitude", "magnitude_db", "phase", "impulse"),
    freq_unit="rad/s",
    statement=(
        "Design an analog Butterworth lowpass filter that has a 0.25 dB or "
        "better ripple at 500 rad/sec and at least 50 dB of attenuation at "
        "2000 rad/sec. Determine the system function in a rational function "
        "form. Plot the magnitude response, the log-magnitude response in dB, "
        "the phase response, and the impulse response of the filter."
    ),
    source="DTA §2.2 item 9 e §4 (Resolução P8.12); Ingle & Proakis, p. 465",
)


P7_10 = FIRSpec(
    exercise="P7.10",
    filter_type="bandstop",
    bands=(
        Band(0.0, 0.30, 1.0),   # passagem inferior: ω ≤ 0.3π
        Band(0.40, 0.60, 0.0),  # rejeição: 0.4π ≤ ω ≤ 0.6π
        Band(0.70, 1.0, 1.0),   # passagem superior: ω ≥ 0.7π
    ),
    rp_db=0.2,
    as_db=50.0,
    method="window:hamming",
    statement=(
        "Design a bandstop filter using the Hamming window design technique. "
        "The specifications are: lower stopband edge: 0.4π, As = 50 dB; upper "
        "stopband edge: 0.6π; lower passband edge: 0.3π, Rp = 0.2 dB; upper "
        "passband edge: 0.7π. Plot the impulse response and the magnitude "
        "response (in dB) of the designed filter. Do not use the fir1 function."
    ),
    source="DTA §2.1 item 2 e §4 (Resolução P7.10); Ingle & Proakis, pp. 378–379",
)


P7_11 = FIRSpec(
    exercise="P7.11",
    filter_type="bandpass",
    bands=(
        Band(0.0, 0.30, 0.0),   # rejeição inferior: ω ≤ 0.3π
        Band(0.40, 0.50, 1.0),  # passagem: 0.4π ≤ ω ≤ 0.5π
        Band(0.60, 1.0, 0.0),   # rejeição superior: ω ≥ 0.6π
    ),
    rp_db=0.5,
    as_db=50.0,
    method="window:hamming",
    statement=(
        "Design a bandpass filter using the Hamming window design technique. "
        "The specifications are: lower stopband edge: 0.3π, As = 50 dB; upper "
        "stopband edge: 0.6π; lower passband edge: 0.4π, Rp = 0.5 dB; upper "
        "passband edge: 0.5π. Plot the impulse response and the magnitude "
        "response (in dB) of the designed filter. Do not use the fir1 function."
    ),
    source="DTA §2.1 item 3 e §4 (Resolução P7.11); Ingle & Proakis, p. 379",
)


P7_12 = FIRSpec(
    exercise="P7.12",
    filter_type="highpass",
    bands=(
        Band(0.0, 0.40, 0.0),  # rejeição: ω ≤ 0.4π
        Band(0.60, 1.0, 1.0),  # passagem: ω ≥ 0.6π
    ),
    rp_db=0.004,
    as_db=50.0,
    method="window:fixed (escolha justificada)",
    statement=(
        "Design a highpass filter using one of the fixed window functions. The "
        "specifications are: stopband edge: 0.4π, As = 50 dB; passband edge: "
        "0.6π, Rp = 0.004 dB. Plot the zoomed magnitude response (in dB) of the "
        "designed filter in the passband to verify the passband ripple Rp. Do "
        "not use the fir1 function."
    ),
    source="DTA §2.1 item 4 e §4 (Resolução P7.12); Ingle & Proakis, p. 379",
)


P7_13 = FIRToleranceSpec(
    exercise="P7.13",
    filter_type="bandstop",
    bands=(
        ToleranceBand(0.0, 0.25, 1.0, 0.025),   # 0.975 ≤ |H| ≤ 1.025
        ToleranceBand(0.35, 0.65, 0.0, 0.005),  # 0 ≤ |H| ≤ 0.005
        ToleranceBand(0.75, 1.0, 1.0, 0.025),   # 0.975 ≤ |H| ≤ 1.025
    ),
    method="window:kaiser (comprimento mínimo)",
    statement=(
        "Using the Kaiser window method, design a linear-phase FIR digital filter "
        "that meets the following specifications: 0.975 ≤ |H| ≤ 1.025, "
        "0 ≤ ω ≤ 0.25π; 0 ≤ |H| ≤ 0.005, 0.35π ≤ ω ≤ 0.65π; 0.975 ≤ |H| ≤ 1.025, "
        "0.75π ≤ ω ≤ π. Determine the minimum length impulse response h(n) of "
        "such a filter. Provide a plot containing subplots of the amplitude "
        "response and the magnitude response in dB. Do not use the fir1 function."
    ),
    source="DTA §2.1 item 5 e §4 (Resolução P7.13); Ingle & Proakis, p. 379",
)


P7_14 = FIRToleranceSpec(
    exercise="P7.14",
    filter_type="bandpass",
    bands=(
        ToleranceBand(0.0, 0.25, 0.0, 0.01),    # 0 ≤ |H| ≤ 0.01
        ToleranceBand(0.35, 0.65, 1.0, 0.05),   # 0.95 ≤ |H| ≤ 1.05
        ToleranceBand(0.75, 1.0, 0.0, 0.01),    # 0 ≤ |H| ≤ 0.01
    ),
    method="window:kaiser (comprimento mínimo)",
    statement=(
        "We wish to use the Kaiser window method to design a linear-phase FIR "
        "digital filter that meets the following specifications: "
        "0 ≤ |H| ≤ 0.01, 0 ≤ ω ≤ 0.25π; 0.95 ≤ |H| ≤ 1.05, 0.35π ≤ ω ≤ 0.65π; "
        "0 ≤ |H| ≤ 0.01, 0.75π ≤ ω ≤ π. Determine the minimum length impulse "
        "response h(n) of such a filter. Provide a plot containing subplots of "
        "the amplitude response and the magnitude response in dB. Do not use "
        "the fir1 function."
    ),
    source="DTA §2.1 item 6 e §4 (Resolução P7.14); Ingle & Proakis, pp. 379–380",
)


P7_15 = FIRToleranceSpec(
    exercise="P7.15",
    filter_type="multiband",
    bands=(
        ToleranceBand(0.0, 0.30, 1.0, 0.01),    # Banda 1: ganho 1,   δ1 = 0.01
        ToleranceBand(0.40, 0.70, 0.5, 0.005),  # Banda 2: ganho 0.5, δ2 = 0.005
        ToleranceBand(0.80, 1.0, 0.0, 0.001),   # Banda 3: ganho 0,   δ3 = 0.001
    ),
    method="window:kaiser; comparar com Exemplo 7.26 (Parks-McClellan)",
    statement=(
        "Design the staircase filter of Example 7.26 using the Kaiser window "
        "approach. The specifications are: Band-1: 0 ≤ ω ≤ 0.3π, Ideal gain = 1, "
        "δ1 = 0.01; Band-2: 0.4π ≤ ω ≤ 0.7π, Ideal gain = 0.5, δ2 = 0.005; "
        "Band-3: 0.8π ≤ ω ≤ π, Ideal gain = 0, δ3 = 0.001. Compare the filter "
        "length of this design with that of Example 7.26. Provide a plot of the "
        "magnitude response in dB. Do not use the fir1 function."
    ),
    source="DTA §2.1 item 7 e §4 (Resolução P7.15); Ingle & Proakis, p. 380; "
           "Exemplo 7.26 em pp. 367–368 (M = 49, informado pelo usuário, decisão D6)",
)

# Exemplo 7.26 (Parks-McClellan), conforme informado em 2026-10-30 (decisão D6):
# estimativa inicial M = 51; firpm: M = 51 → 62.07 dB, M = 50 → 60.03 dB,
# M = 49 → 60.61 dB; comprimento final M = 49.
EXAMPLE_7_26_M = 49


P7_16 = FIRSpec(
    exercise="P7.16",
    filter_type="bandpass",
    bands=(
        Band(0.0, 0.30, 0.0),   # rejeição inferior: ω ≤ 0.3π
        Band(0.40, 0.50, 1.0),  # passagem: 0.4π ≤ ω ≤ 0.5π
        Band(0.60, 1.0, 0.0),   # rejeição superior: ω ≥ 0.6π
    ),
    rp_db=0.5,
    as_db=40.0,
    method="window:fixed de menor comprimento",
    statement=(
        "Design a bandpass filter using a fixed window design technique that has "
        "the minimum length and that satisfies the following specifications: "
        "lower stopband edge = 0.3π, As = 40 dB, upper stopband edge = 0.6π; "
        "lower passband edge = 0.4π, Rp = 0.5 dB, upper passband edge = 0.5π. "
        "Provide a plot of the log-magnitude response in dB and stem plot of the "
        "impulse response."
    ),
    source="DTA §2.1 item 8 e §4 (Resolução P7.16); Ingle & Proakis, p. 380",
)


P8_13 = AnalogLowpassSpec(
    exercise="P8.13",
    approximation="butterworth",
    wp=2 * math.pi * 10e3,   # Ωp = 2π·10 kHz
    ws=2 * math.pi * 20e3,   # Ωs = 2π·20 kHz
    rp_db=0.5,
    as_db=45.0,
    form="cascade",
    plots=("magnitude", "magnitude_db", "group_delay", "impulse"),
    freq_unit="Hz",
    statement=(
        "Design an analog Butterworth lowpass filter that has a 0.5 dB or better "
        "ripple at 10 kHz and at least 45 dB of attenuation at 20 kHz. Determine "
        "the system function in a cascade form. Plot the magnitude response, the "
        "log-magnitude response in dB, the group-delay, and the impulse response "
        "of the filter."
    ),
    source="DTA §2.2 item 10 e §4 (Resolução P8.13, com errata em θ_k); Ingle & Proakis, p. 465",
)


P8_14 = AnalogLowpassSpec(
    exercise="P8.14",
    approximation="chebyshev1",
    wp=10.0,   # rad/s
    ws=15.0,   # rad/s
    rp_db=1.0,
    as_db=50.0,
    form="rational",
    plots=("magnitude", "magnitude_db", "group_delay", "impulse"),
    freq_unit="rad/s",
    statement=(
        "Design a lowpass analog Chebyshev-I filter with an acceptable ripple of "
        "1 dB for |Ω| ≤ 10 and an attenuation of 50 dB or greater beyond "
        "|Ω| = 15 rad/sec. Determine the system function in a rational function "
        "form. Plot the magnitude response, the log-magnitude response in dB, the "
        "group-delay, and the impulse response of the filter."
    ),
    source="DTA §2.2 item 11 e §4 (Resolução P8.14); Ingle & Proakis, p. 465",
)


P8_15 = AnalogLowpassSpec(
    exercise="P8.15",
    approximation="chebyshev1",
    wp=2 * math.pi * 4e3,    # Ωp = 2π·4 kHz
    ws=2 * math.pi * 20e3,   # Ωs = 2π·20 kHz
    rp_db=0.5,
    as_db=45.0,
    form="cascade",
    plots=("magnitude", "magnitude_db", "phase", "impulse"),
    freq_unit="Hz",
    statement=(
        "Design a lowpass analog Chebyshev-I filter with the following "
        "characteristics: A passband ripple of 0.5 dB, passband cutoff frequency "
        "of 4 kHz, and stopband attenuation of 45 dB or greater beyond 20 kHz. "
        "Determine the system function in a cascade form. Plot the magnitude "
        "response, the log-magnitude response in dB, the phase response, and the "
        "impulse response of the filter."
    ),
    source="DTA §2.2 item 12 e §4 (Resolução P8.15); Ingle & Proakis, pp. 465–466",
)
