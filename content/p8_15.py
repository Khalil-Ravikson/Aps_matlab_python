r"""Conteúdo didático — P8.15 (Chebyshev-I analógico passa-baixas, forma em cascata).

Fonte técnica: DTA §2.2 item 12, §3.2.B e §4; Ingle & Proakis, Seção 8.2.2
(conforme DTA). Todos os valores do DTA para este exercício foram confirmados.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from dsp_av2.iir import analysis
from dsp_av2.iir.prototypes import chebyshev1_of_order
from dsp_av2.specs import P8_15 as SPEC
from exercises import p8_15

from .helpers import TWO_PI, iir_edges_context, md_table, pole_table
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable, tex_sci


def _ellipse(N, rp):
    eps = math.sqrt(10 ** (rp / 10) - 1)
    alpha = 1 / eps + math.sqrt(1 + 1 / eps ** 2)
    return alpha, 0.5 * (alpha ** (1 / N) - alpha ** (-1 / N)), 0.5 * (alpha ** (1 / N) + alpha ** (-1 / N))


@lru_cache(maxsize=1)
def objects() -> dict:
    proto = p8_15.design()
    _, a, b = _ellipse(proto.N, SPEC.rp_db)
    return {"spec": SPEC, "proto": proto, "cascade": p8_15.cascade_form(proto), "report": p8_15.validate(proto),
            "lower": chebyshev1_of_order(proto.N - 1, SPEC.wp, SPEC.rp_db),
            "pole_curve": ("ellipse", a * SPEC.wp, b * SPEC.wp)}


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    proto, c = o["proto"], o["cascade"]
    ctx = iir_edges_context(SPEC, proto)
    eps = proto.params["epsilon"]
    g = math.sqrt(10 ** (SPEC.as_db / 10) - 1) / eps
    alpha, a, b = _ellipse(proto.N, SPEC.rp_db)
    secs = sorted(c.sections, key=lambda s: s.den[1])
    rows = []
    for i, s in enumerate(secs, 1):
        sigma = -s.den[1] / 2
        wd = math.sqrt(s.den[2] - sigma ** 2)
        rows.append((f"Biquad {i}", f"${sigma:,.2f} \\pm j{wd:,.2f}$".replace(",", "\\,"), f"${tex_sci(s.den[1], 7)}$",
                     f"${tex_sci(s.den[2], 5)}$", f"{math.sqrt(s.den[2]) / TWO_PI:.1f} Hz"))
    att_lower = -analysis.magnitude_db(analysis.frequency_response(o["lower"], np.array([SPEC.ws])))[0]
    low = secs[0]
    ctx |= {
        "eps": eps, "g": g, "acosh_g": math.acosh(g), "acosh_r": math.acosh(SPEC.ws / SPEC.wp),
        "alpha_e": alpha, "a_e": a, "b_e": b, "a_wp": a * SPEC.wp, "b_wp": b * SPEC.wp,
        "pole_table": pole_table(proto, unit_khz=True),
        "sec_table": md_table(["Seção", "par de polos", "coef. de $s$", "termo constante", "$\\sqrt{\\text{const}}/2\\pi$"], rows),
        "gain": c.gain, "sqrt1e2": math.sqrt(1 + eps ** 2), "n_sec": len(secs),
        "s1_a1": secs[0].den[1], "s1_a0": secs[0].den[2], "s2_a1": secs[1].den[1], "s2_a0": secs[1].den[2],
        "low_f": math.sqrt(low.den[2]) / TWO_PI / 1e3, "Np1": proto.N + 1,
        "att_lower": att_lower, "lower_gap": SPEC.as_db - att_lower,
    }
    return ctx


CONTENT = ExerciseContent(
    id="p8_15", number="P8.15", chapter=8,
    title="Chebyshev-I analógico passa-baixas em forma de cascata",
    tagline="IIR · protótipo analógico · Chebyshev tipo I · cascata",
    highlights=(
        Value("N", "[[N]]", "resultado", "ordem mínima"),
        Value(r"G", "[[gain:.5f]]", "resultado", "ganho global: $1/\\sqrt{1+\\epsilon^2}$"),
        Value(r"\text{seções}", "[[n_sec]] biquads", "resultado", "N par: sem seção de 1ª ordem"),
        Value(r"A(F_s)", "[[att_ws:.2f]] dB", "resultado", "atenuação em [[Fs_k:g]] kHz"),
    ),
    errata=(),
    result_keys=("eps", "att_ws", "att_lower", "N_real", "gain"),

    asked=r"""
Projetar um **filtro analógico passa-baixas Chebyshev tipo I** com:

- ondulação de **[[Rp:g]] dB** na passagem, até $F_p = [[Fp_k:g]]$ kHz;
- atenuação de **[[As:g]] dB ou mais** a partir de $F_s = [[Fs_k:g]]$ kHz.

O resultado deve ser dado em **forma de cascata** (seções de 2ª ordem). São pedidos quatro gráficos: magnitude, magnitude em dB, **fase** e resposta ao impulso.
""",
    data_note=r"""
Frequências em hertz: a conversão $\Omega = 2\pi F$ vem antes das fórmulas. A transição é larga ($F_s/F_p = [[Wr:g]]$), e por isso a ordem vai sair baixa.
""",
    data_rows=(
        Value(r"F_p", "[[Fp_k:g]] kHz", "dado", "borda de passagem"),
        Value(r"R_p", "[[Rp:g]] dB", "dado", "ondulação na passagem"),
        Value(r"F_s", "[[Fs_k:g]] kHz", "dado", "borda de rejeição"),
        Value(r"A_s", "[[As:g]] dB", "dado", "atenuação mínima"),
        Value(r"\text{Forma}", "cascata", "dado", "produto de seções de 2ª ordem"),
    ),

    theory=r"""
As fórmulas são as do P8.14 (Chebyshev-I: ondulação constante na passagem, polos sobre uma elipse). A diferença está na forma pedida.

**Cascata.** Cada par conjugado $p, p^*$ forma um biquad real $s^2 - 2\,\mathrm{Re}(p)\,s + |p|^2$. Normalizando cada seção para ganho DC unitário (numerador $= |p|^2$), o ganho global fica com toda a informação de nível. Com $N$ par, esse ganho é $1/\sqrt{1+\epsilon^2}$, porque o DC está num vale da ondulação.

**Vantagem prática.** Em hardware ou software, seções de 2ª ordem em série são muito menos sensíveis a erros de coeficiente que um único polinômio de ordem alta.
""",
    deepening=(
        Deepening("Por que arredondar para cima, mesmo com N_real logo acima de um inteiro?", r"""
A fórmula dá $N_{real} = [[N_real:.3f]]$, só um pouco acima de 3. Como a ordem precisa ser inteira e **não pode** ficar abaixo do necessário, usa-se [[N]]. A verificação confirma: com $N = [[Nm1]]$, a atenuação em $F_s$ seria [[att_lower:.2f]] dB, [[lower_gap:.2f]] dB abaixo do exigido. Em compensação, $N = [[N]]$ entrega [[margin_ws:.1f]] dB de folga.
"""),
    ),

    equations=(
        Equation(r"\epsilon = \sqrt{10^{R_p/10}-1}, \qquad N = \left\lceil \frac{\cosh^{-1} g}{\cosh^{-1}(\Omega_s/\Omega_p)} \right\rceil", "Ondulação e ordem", "", (), "DTA §3.2.B"),
        Equation(r"p_k = -a\,\Omega_p\sin\frac{(2k+1)\pi}{2N} + j\,b\,\Omega_p\cos\frac{(2k+1)\pi}{2N}", "Polos sobre a elipse", "", (), "DTA §3.2.B"),
        Equation(r"H_a(s) = G\prod_{k=1}^{N/2}\frac{|p_k|^2}{s^2 - 2\,\mathrm{Re}(p_k)\,s + |p_k|^2}, \qquad G = \frac{1}{\sqrt{1+\epsilon^2}}", "Cascata (N par)",
                 "Cada biquad tem ganho DC unitário; $G$ ajusta o nível.", (Variable(r"G", "ganho global"),), "DTA §4"),
    ),

    steps=(
        Step("Converter e calcular ε", "Frequências em rad/s; ondulação a partir de $R_p$.",
             r"\Omega_p = 2\pi\cdot[[Fp:.0f]] = [[Wp:.2f]],\quad \Omega_s = [[Ws:.2f]]\ \text{rad/s}, \qquad \epsilon = \sqrt{10^{[[Rp10:g]]}-1} = [[eps:.6f]]",
             (Value(r"\epsilon", "[[eps:.6f]]", "calculado", "fator de ondulação"),)),
        Step("Ordem mínima", "",
             r"g = [[g:.2f]], \quad N = \left\lceil \frac{[[acosh_g:.4f]]}{[[acosh_r:.4f]]} \right\rceil = \lceil [[N_real:.3f]] \rceil = [[N]]",
             (Value("N", "[[N]]", "resultado", "ordem mínima"),)),
        Step("Elipse e polos", "Semieixos e os [[N]] polos (dois pares conjugados):\n\n[[pole_table]]",
             r"a\Omega_p = [[a_wp:.2f]], \qquad b\Omega_p = [[b_wp:.2f]]\ \text{rad/s}",
             (Value(r"\text{polos}", "[[N]]", "calculado", "dois pares conjugados"),)),
        Step("Formar os biquads", "Cada par conjugado vira uma seção com ganho DC unitário:\n\n[[sec_table]]",
             r"s^2 + [[s1_a1:.2f]]\,s + [[s1_a0:tex5]], \qquad s^2 + [[s2_a1:.2f]]\,s + [[s2_a0:tex5]]",
             (Value(r"\text{seções}", "[[n_sec]]", "resultado", "biquads com numerador = termo constante"),)),
        Step("Ganho global", "Com $N$ par, o DC deve valer $1/\\sqrt{1+\\epsilon^2}$ (vale da ondulação).",
             r"G = \frac{1}{\sqrt{1 + [[eps:.6f]]^2}} = \frac{1}{[[sqrt1e2:.5f]]} = [[gain:.5f]]",
             (Value("G", "[[gain:.5f]]", "resultado", "ganho global"),)),
        Step("Verificar e confirmar a minimalidade", "",
             r"\text{perda}(F_p) = [[loss_wp:.4f]]\ \text{dB}, \quad \text{aten.}(F_s) = [[att_ws:.2f]]\ \text{dB}, \quad N=[[Nm1]]:\ [[att_lower:.2f]]\ \text{dB}",
             (Value(r"\text{atenuação em } F_s", "[[att_ws:.2f]] dB", "resultado", "folga de [[margin_ws:.1f]] dB"),
              Value(r"N - 1 = [[Nm1]]", "[[att_lower:.2f]] dB", "criterio", "< [[As:g]] dB: [[N]] é o mínimo"))),
    ),

    implementation=r"""
1. **Ordem e polos** (`cheb1_order`, `cheb1_poles`), conferidos com o SciPy nos testes.
2. **Cascata** (`forms.cascade`): agrupa os pares conjugados em biquads com ganho DC unitário e calcula $G = k/\prod \text{num}_k$. Um teste confirma que o produto das seções reproduz a resposta do filtro.
3. **Análise**: magnitude e fase a partir dos polos; $h(t)$ conferida com a inversa de Laplace por resíduos.
""",
    code_roles=(
        CodeRole("dsp_av2.iir.prototypes.cheb1_order", "Passos 1–2"),
        CodeRole("dsp_av2.iir.prototypes.cheb1_poles", "Passo 3"),
        CodeRole("dsp_av2.iir.forms.cascade", "Passos 4–5: biquads e ganho"),
        CodeRole("dsp_av2.validation.validate_analog_lowpass", "Passo 6"),
    ),

    figures=(
        Figure("poles", "Polos no plano $s$", r"""
Dois pares conjugados sobre a elipse do Chebyshev. O par mais alto (mais perto do eixo $j\Omega$) responde pela oscilação perto de $F_p$, e o par mais baixo, pela parte de baixa frequência.
"""),
        Figure("magnitude", "Magnitude", r"""
$N + 1 = [[Np1]]$ extremos em $[0, F_p]$, começando no vale $[[gain:.4f]]$ em DC (N par). Depois de $F_p$, a queda é monotônica.
"""),
        Figure("db", "Magnitude em dB", r"""
Ondulação de [[Rp:g]] dB até [[Fp_k:g]] kHz e [[att_ws:.2f]] dB em [[Fs_k:g]] kHz, [[margin_ws:.1f]] dB além do exigido, por causa do arredondamento $N_{real} = [[N_real:.2f]] \to [[N]]$.
"""),
        Figure("phase", "Fase", r"""
A fase total vai de 0 a $-N\pi/2 = -[[N]]\pi/2$. O traço principal salta de $-\pi$ para $+\pi$ perto de $F_p$, onde a fase varia mais rápido, e depois tende ao valor final, equivalente a $-[[N]]\pi/2$ módulo $2\pi$.
"""),
        Figure("impulse", "Resposta ao impulso", r"""
Oscilação amortecida dominada pelo par de polos de menor amortecimento, com frequência próxima de $F_p$ e duração da ordem de 1 ms.
"""),
    ),

    verification=r"""
Perda máxima em $[0, \Omega_p]$ ≤ $R_p$, atenuação mínima em $[\Omega_s, 100\,\Omega_s]$ ≥ $A_s$, e minimalidade ($N - 1$ não atende).
""",
    conclusion=r"""
- $\epsilon = [[eps:.6f]]$ e **$N = [[N]]$** ($\lceil [[N_real:.3f]] \rceil$).
- Cascata: $H_a(s) = [[gain:.5f]] \cdot \frac{[[s1_a0:tex5]]}{s^2 + [[s1_a1:.2f]]s + [[s1_a0:tex5]]} \cdot \frac{[[s2_a0:tex5]]}{s^2 + [[s2_a1:.2f]]s + [[s2_a0:tex5]]}$, idêntica ao DTA.
- Perda de [[Rp:g]] dB em [[Fp_k:g]] kHz e [[att_ws:.2f]] dB em [[Fs_k:g]] kHz. Com $N = [[Nm1]]$ faltariam [[lower_gap:.2f]] dB.
""",
)
