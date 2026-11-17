r"""Conteúdo didático — P8.14 (Chebyshev-I analógico passa-baixas, forma racional).

Fonte técnica: DTA §2.2 item 11, §3.2.B e §4; Ingle & Proakis, Seção 8.2.2
(conforme DTA). Todos os valores do DTA para este exercício foram confirmados.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from dsp_av2.iir import analysis
from dsp_av2.iir.prototypes import butter_order, chebyshev1_of_order
from dsp_av2.specs import P8_14 as SPEC
from exercises import p8_14

from .helpers import group_delay_stats, iir_edges_context, md_table, pole_table
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable, tex_sci


def _ellipse(N, rp):
    eps = math.sqrt(10 ** (rp / 10) - 1)
    alpha = 1 / eps + math.sqrt(1 + 1 / eps ** 2)
    a = 0.5 * (alpha ** (1 / N) - alpha ** (-1 / N))
    b = 0.5 * (alpha ** (1 / N) + alpha ** (-1 / N))
    return alpha, a, b


@lru_cache(maxsize=1)
def objects() -> dict:
    proto = p8_14.design()
    _, a, b = _ellipse(proto.N, SPEC.rp_db)
    return {"spec": SPEC, "proto": proto, "report": p8_14.validate(proto),
            "lower": chebyshev1_of_order(proto.N - 1, SPEC.wp, SPEC.rp_db),
            "pole_curve": ("ellipse", a * SPEC.wp, b * SPEC.wp)}


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    proto = o["proto"]
    ctx = iir_edges_context(SPEC, proto)
    eps = proto.params["epsilon"]
    g = math.sqrt(10 ** (SPEC.as_db / 10) - 1) / eps
    alpha, a, b = _ellipse(proto.N, SPEC.rp_db)
    B, A = p8_14.rational_form(proto)
    coef_rows = [(f"$s^{{{len(A) - 1 - i}}}$", f"${tex_sci(c, 7)}$") for i, c in enumerate(A)]
    w = np.linspace(0, SPEC.wp, 200001)
    mag = np.abs(analysis.frequency_response(proto, w))
    d = np.diff(mag)
    extrema = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1]))) + 2
    att_lower = -analysis.magnitude_db(analysis.frequency_response(o["lower"], np.array([SPEC.ws])))[0]
    gd = group_delay_stats(proto, 2 * SPEC.ws)
    Nb, Nb_real = butter_order(SPEC.wp, SPEC.ws, SPEC.rp_db, SPEC.as_db)
    ctx |= {
        "eps": eps, "eps2": eps ** 2, "g": g, "acosh_g": math.acosh(g), "acosh_r": math.acosh(SPEC.ws / SPEC.wp),
        "alpha_e": alpha, "a_e": a, "b_e": b, "a_wp": a * SPEC.wp, "b_wp": b * SPEC.wp, "ab_ratio": a / b,
        "pole_table": pole_table(proto), "coef_table": md_table(["potência", "coeficiente de $A(s)$"], coef_rows),
        "B": B[0], "a0": A[-1], "dc": proto.params["dc_gain"], "dc_db": 20 * math.log10(proto.params["dc_gain"]),
        "sqrt1e2": math.sqrt(1 + eps ** 2),
        "extrema": extrema, "Np1": proto.N + 1, "min_mag": mag.min(), "rip_lin": 10 ** (-SPEC.rp_db / 20),
        "att_lower": att_lower, "gd0": gd["gd0"], "gd_peak": gd["gd_peak"], "gd_peak_w": gd["gd_peak_w"],
        "Nb": Nb, "Nb_real": Nb_real, "half": proto.N / Nb,
        "period": 2 * math.pi / SPEC.wp,
    }
    return ctx


CONTENT = ExerciseContent(
    id="p8_14", number="P8.14", chapter=8,
    title="Chebyshev-I analógico passa-baixas em forma racional",
    tagline="IIR · protótipo analógico · Chebyshev tipo I · forma racional",
    highlights=(
        Value("N", "[[N]]", "resultado", "ordem mínima"),
        Value(r"\epsilon", "[[eps:.6f]]", "calculado", "fator de ondulação"),
        Value(r"A(\Omega_s)", "[[att_ws:.2f]] dB", "resultado", "atenuação em [[Ws:g]] rad/s"),
        Value(r"N_{\text{Butterworth}}", "[[Nb]]", "calculado", "ordem que um Butterworth exigiria"),
    ),
    errata=(),
    result_keys=("eps", "att_ws", "att_lower", "N_real", "B", "a0", "gd_peak"),

    asked=r"""
Projetar um **filtro analógico passa-baixas Chebyshev tipo I** que:

- tenha ondulação de no máximo **[[Rp:g]] dB** para $|\Omega| \le [[Wp:g]]$ rad/s;
- atenue **[[As:g]] dB ou mais** a partir de $|\Omega| = [[Ws:g]]$ rad/s.

O resultado deve ser dado em **forma racional**. São pedidos quatro gráficos: magnitude, magnitude em dB, **atraso de grupo** e resposta ao impulso. A transição é estreita: $\Omega_s/\Omega_p = [[Wr:g]]$.
""",
    data_note=r"""
Uma transição tão estreita favorece o Chebyshev: ele aceita ondulação na passagem em troca de uma queda muito mais abrupta (passo 6).
""",
    data_rows=(
        Value(r"\Omega_p", "[[Wp:g]] rad/s", "dado", "borda de passagem"),
        Value(r"R_p", "[[Rp:g]] dB", "dado", "ondulação máxima na passagem"),
        Value(r"\Omega_s", "[[Ws:g]] rad/s", "dado", "borda de rejeição"),
        Value(r"A_s", "[[As:g]] dB", "dado", "atenuação mínima"),
        Value(r"\text{Forma}", "racional", "dado", "$B(s)/A(s)$"),
    ),

    theory=r"""
**Chebyshev tipo I.** A magnitude é

$$|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 T_N^2(\Omega/\Omega_p)},$$

em que $T_N$ é o polinômio de Chebyshev de ordem $N$. Para $|x| \le 1$, $T_N(x) = \cos(N\cos^{-1}x)$ oscila entre ±1, o que dá **ondulação de amplitude constante** (equiripple) na passagem. Para $x > 1$, $T_N(x) = \cosh(N\cosh^{-1}x)$ cresce rapidamente, o que dá uma queda abrupta.

**Polos sobre uma elipse.** Os polos ficam numa elipse de semieixos $a\Omega_p$ (real) e $b\Omega_p$ (imaginário), com $a \ll b$. Polos mais próximos do eixo $j\Omega$ produzem transição mais abrupta, mas também atraso de grupo maior e mais irregular.

**Ganho DC.** Com $N$ par, $T_N(0) = \pm 1$ e o DC fica num **vale** da ondulação: $|H_a(0)| = 1/\sqrt{1+\epsilon^2}$. Com $N$ ímpar, $|H_a(0)| = 1$.
""",
    deepening=(
        Deepening("De onde vem a fórmula da ordem?", r"""
Impondo $A_s$ em $\Omega_s$: $1 + \epsilon^2 T_N^2(\Omega_r) = 10^{A_s/10}$, com $\Omega_r = \Omega_s/\Omega_p$. Logo $T_N(\Omega_r) = \sqrt{10^{A_s/10}-1}/\epsilon = g$. Como $\Omega_r > 1$, $T_N(\Omega_r) = \cosh(N\cosh^{-1}\Omega_r)$, e então $N = \cosh^{-1}g / \cosh^{-1}\Omega_r$, arredondado para cima.
"""),
        Deepening("Por que o atraso de grupo é tão irregular?", r"""
Cada polo $p_k = \sigma_k + j\omega_k$ contribui com um "pico" $\frac{-\sigma_k}{\sigma_k^2 + (\Omega - \omega_k)^2}$ no atraso de grupo. Na elipse do Chebyshev, $|\sigma_k|$ é pequeno (razão $a/b = [[ab_ratio:.3f]]$), e os picos são altos e estreitos. Por isso o atraso tem ondulações na passagem e um pico pronunciado perto de $\Omega_p$.
"""),
    ),

    equations=(
        Equation(r"\epsilon = \sqrt{10^{R_p/10} - 1}", "Fator de ondulação", "", (Variable(r"\epsilon", "define a amplitude da ondulação na passagem"),), "DTA §3.2.B"),
        Equation(r"N = \left\lceil \frac{\cosh^{-1} g}{\cosh^{-1}\Omega_r} \right\rceil, \quad g = \frac{\sqrt{10^{A_s/10}-1}}{\epsilon}, \quad \Omega_r = \frac{\Omega_s}{\Omega_p}", "Ordem mínima", "", (), "DTA §3.2.B"),
        Equation(r"\alpha = \tfrac{1}{\epsilon} + \sqrt{1 + \tfrac{1}{\epsilon^2}},\quad a = \tfrac12(\alpha^{1/N} - \alpha^{-1/N}),\quad b = \tfrac12(\alpha^{1/N} + \alpha^{-1/N})", "Semieixos da elipse", "", (), "DTA §3.2.B"),
        Equation(r"p_k = -a\,\Omega_p\sin\frac{(2k+1)\pi}{2N} + j\,b\,\Omega_p\cos\frac{(2k+1)\pi}{2N}", "Polos", "", (), "DTA §3.2.B"),
        Equation(r"H_a(s) = \frac{B}{\prod_k (s - p_k)}, \qquad B = \frac{a_0}{\sqrt{1+\epsilon^2}}\ \ (N\ \text{par})", "Forma racional e ganho",
                 "O ganho é escolhido para que o máximo da passagem seja 1.", (), "DTA §4"),
    ),

    steps=(
        Step("Fator de ondulação", "Da ondulação admitida na passagem.",
             r"\epsilon = \sqrt{10^{[[Rp10:g]]} - 1} = [[eps:.6f]]",
             (Value(r"\epsilon", "[[eps:.6f]]", "calculado", "fator de ondulação"),)),
        Step("Ordem mínima", "Com $g$ e $\\Omega_r$ da fórmula:",
             r"g = \frac{\sqrt{10^{[[As10:g]]}-1}}{[[eps:.6f]]} = [[g:.2f]], \quad N = \left\lceil \frac{\cosh^{-1}[[g:.2f]]}{\cosh^{-1}[[Wr:g]]} \right\rceil = \left\lceil \frac{[[acosh_g:.4f]]}{[[acosh_r:.4f]]} \right\rceil = \lceil [[N_real:.3f]] \rceil = [[N]]",
             (Value("N", "[[N]]", "resultado", "ordem mínima"),)),
        Step("Elipse dos polos", "Os semieixos definem onde ficam os polos.",
             r"\alpha = [[alpha_e:.4f]], \quad a = [[a_e:.5f]], \quad b = [[b_e:.5f]] \;\Rightarrow\; a\Omega_p = [[a_wp:.4f]],\ b\Omega_p = [[b_wp:.4f]]",
             (Value(r"a/b", "[[ab_ratio:.3f]]", "calculado", "elipse muito achatada: polos perto do eixo jΩ"),)),
        Step("Polos", "[[N]] polos ($N$ par: só pares conjugados).\n\n[[pole_table]]",
             r"p_k = -[[a_wp:.4f]]\sin\frac{(2k+1)\pi}{[[N2]]} + j\,[[b_wp:.4f]]\cos\frac{(2k+1)\pi}{[[N2]]}",
             (Value(r"\text{polos}", "[[N]]", "calculado", "sobre a elipse"),)),
        Step("Forma racional e ganho",
             "Expandindo $\\prod (s - p_k)$:\n\n[[coef_table]]\n\nComo $N$ é par, o DC fica num vale da ondulação, e o ganho é $B = a_0/\\sqrt{1+\\epsilon^2}$.",
             r"B = \frac{[[a0:tex7]]}{[[sqrt1e2:.5f]]} = [[B:tex7]], \qquad H_a(0) = \frac{B}{a_0} = [[dc:.6f]]\ ([[dc_db:.2f]]\ \text{dB})",
             (Value("B", "$[[B:tex7]]$", "resultado", "numerador"),
              Value(r"a_0", "$[[a0:tex7]]$", "resultado", "termo constante de $A(s)$"))),
        Step("Verificar e comparar com o Butterworth", "Nas bordas, com $N - 1$, e com a ordem que um Butterworth exigiria.",
             r"\text{aten.}(\Omega_s) = [[att_ws:.2f]]\ \text{dB}, \quad N=[[Nm1]]:\ [[att_lower:.2f]]\ \text{dB}, \quad N_{\text{Butterworth}} = \lceil [[Nb_real:.2f]] \rceil = [[Nb]]",
             (Value(r"\text{atenuação em } \Omega_s", "[[att_ws:.2f]] dB", "resultado", "≥ [[As:g]] dB"),
              Value(r"N-1 = [[Nm1]]", "[[att_lower:.2f]] dB", "criterio", "< [[As:g]] dB: [[N]] é o mínimo"),
              Value(r"N_{\text{Butterworth}}", "[[Nb]]", "calculado", "o Chebyshev usa [[half:.0%]] dessa ordem"))),
    ),

    implementation=r"""
1. **Ordem e polos** (`cheb1_order`, `cheb1_poles`): fórmulas do livro, conferidas com `scipy.signal.cheb1ord`/`cheby1(analog=True)` nos testes.
2. **Ganho** (`chebyshev1_of_order`): $k = \prod(-p_k)$ vezes $1/\sqrt{1+\epsilon^2}$ (N par).
3. **Forma racional** (`forms.rational`) e **atraso de grupo** analítico (`group_delay`), conferido com $-d\phi/d\Omega$ numérico.
""",
    code_roles=(
        CodeRole("dsp_av2.iir.prototypes.cheb1_order", "Passos 1–2: ε e N"),
        CodeRole("dsp_av2.iir.prototypes.cheb1_poles", "Passos 3–4: elipse e polos"),
        CodeRole("dsp_av2.iir.prototypes.chebyshev1_of_order", "Passo 5: ganho"),
        CodeRole("dsp_av2.iir.forms.rational", "Passo 5: B(s)/A(s)"),
        CodeRole("dsp_av2.iir.analysis.group_delay", "Atraso de grupo analítico"),
    ),

    figures=(
        Figure("poles", "Polos no plano $s$", r"""
Os [[N]] polos ficam sobre uma elipse muito achatada (semieixo real [[a_wp:.2f]], imaginário [[b_wp:.2f]] rad/s), bem mais perto do eixo $j\Omega$ que os de um Butterworth de mesma faixa de passagem.
"""),
        Figure("magnitude", "Magnitude", r"""
**[[extrema]] extremos** em $[0, \Omega_p]$ ($N + 1 = [[Np1]]$), alternando entre 1 e $10^{-R_p/20} = [[rip_lin:.4f]]$. Em DC a magnitude vale [[dc:.4f]] (vale), porque $N$ é par.
"""),
        Figure("db", "Magnitude em dB", r"""
Ondulação de [[Rp:g]] dB na passagem e queda abrupta: [[att_ws:.2f]] dB já em $\Omega_s = [[Ws:g]]$ rad/s.
"""),
        Figure("group_delay", "Atraso de grupo", r"""
Começa em [[gd0:.3f]] s, ondula na passagem e chega a [[gd_peak:.2f]] s perto de $\Omega = [[gd_peak_w:.2f]]$ rad/s. Esse pico alto e estreito é o preço da transição abrupta (aprofundamento).
"""),
        Figure("impulse", "Resposta ao impulso", r"""
Oscilação longa e pouco amortecida, com período próximo de $2\pi/\Omega_p \approx [[period:.2f]]$ s. Ela vem dos polos de baixo amortecimento perto do eixo $j\Omega$.
"""),
    ),

    verification=r"""
Perda máxima em $[0, \Omega_p]$ ≤ $R_p$ (medida em relação ao máximo, que é 1), atenuação mínima em $[\Omega_s, 100\,\Omega_s]$ ≥ $A_s$, e minimalidade ($N - 1$ não atende).
""",
    conclusion=r"""
- $\epsilon = [[eps:.6f]]$ e **$N = [[N]]$** ($\lceil [[N_real:.2f]] \rceil$). $B = [[B:tex7]]$ e $a_0 = [[a0:tex7]]$, valores que coincidem com o DTA.
- Perda de exatamente [[Rp:g]] dB em $\Omega_p$ (equiripple) e [[att_ws:.2f]] dB em $\Omega_s$. Com $N = [[Nm1]]$ seriam [[att_lower:.2f]] dB.
- Com as mesmas especificações, um Butterworth precisaria de $N = [[Nb]]$. A ondulação na passagem **reduz a ordem pela metade**, ao custo de um atraso de grupo muito irregular.
""",
)
