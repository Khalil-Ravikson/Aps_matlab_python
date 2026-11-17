r"""Conteúdo didático — P8.12 (Butterworth analógico passa-baixas, forma racional).

Fonte técnica: DTA §2.2 item 9, §3.2.A e §4; Ingle & Proakis, Seção 8.2.1
(conforme DTA). Todos os valores do DTA para este exercício foram confirmados.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from dsp_av2.iir import analysis
from dsp_av2.iir.prototypes import butterworth_of_order
from dsp_av2.specs import P8_12 as SPEC
from exercises import p8_12

from .helpers import iir_edges_context, md_table, pole_table
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable, tex_sci


@lru_cache(maxsize=1)
def objects() -> dict:
    proto = p8_12.design()
    return {"spec": SPEC, "proto": proto, "report": p8_12.validate(proto),
            "lower": butterworth_of_order(proto.N - 1, SPEC.wp, SPEC.rp_db),
            "pole_curve": ("circle", proto.params["Omega_c"])}


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    proto = o["proto"]
    ctx = iir_edges_context(SPEC, proto)
    wc = proto.params["Omega_c"]
    num = 10 ** (SPEC.rp_db / 10) - 1
    den = 10 ** (SPEC.as_db / 10) - 1
    b, a = p8_12.rational_form(proto)
    coef_rows = [(f"$s^{{{len(a) - 1 - i}}}$" if i < len(a) - 1 else "$s^0$", f"$a_{{{len(a) - 1 - i}}}$", f"${tex_sci(c, 7)}$")
                 for i, c in enumerate(a)]
    att_lower = -analysis.magnitude_db(analysis.frequency_response(o["lower"], np.array([SPEC.ws])))[0]
    ctx |= {
        "num": num, "den": den, "log_ratio": math.log10(num / den), "log_w": 2 * math.log10(SPEC.wp / SPEC.ws),
        "Omega_c": wc, "num_root": num ** (1 / (2 * proto.N)), "pole_step": 180 / proto.N,
        "pole_table": pole_table(proto), "coef_table": md_table(["potência", "coeficiente", "valor"], coef_rows),
        "B": b[0], "a0": a[-1], "wcN": wc ** proto.N, "att_lower": att_lower, "margin_ws": ctx["att_ws"] - SPEC.as_db,
        "closest_deg": 90 + 180 / (2 * proto.N),
    }
    return ctx


CONTENT = ExerciseContent(
    id="p8_12", number="P8.12", chapter=8,
    title="Butterworth analógico passa-baixas em forma racional",
    tagline="IIR · protótipo analógico · Butterworth · forma racional",
    highlights=(
        Value("N", "[[N]]", "resultado", "ordem mínima"),
        Value(r"\Omega_c", "[[Omega_c:.4f]] rad/s", "calculado", "corte de −3 dB"),
        Value(r"A(\Omega_s)", "[[att_ws:.2f]] dB", "resultado", "atenuação em [[Ws:g]] rad/s"),
        Value(r"H_a(0)", "[[H0:g]]", "resultado", "ganho DC"),
    ),
    errata=(),
    result_keys=("Omega_c", "att_ws", "att_lower", "N_real"),

    asked=r"""
Projetar um **filtro analógico passa-baixas de Butterworth** que:

- perca **no máximo [[Rp:g]] dB** até $\Omega_p = [[Wp:g]]$ rad/s;
- atenue **pelo menos [[As:g]] dB** a partir de $\Omega_s = [[Ws:g]]$ rad/s.

O resultado deve ser dado como **função racional** $H_a(s) = B(s)/A(s)$. São pedidos quatro gráficos: magnitude, magnitude em dB, **fase** e resposta ao impulso.
""",
    data_note=r"""
As frequências já estão em rad/s, então as fórmulas se aplicam sem conversão.
""",
    data_rows=(
        Value(r"\Omega_p", "[[Wp:g]] rad/s", "dado", "borda de passagem"),
        Value(r"R_p", "[[Rp:g]] dB", "dado", "perda máxima até $\\Omega_p$"),
        Value(r"\Omega_s", "[[Ws:g]] rad/s", "dado", "borda de rejeição"),
        Value(r"A_s", "[[As:g]] dB", "dado", "atenuação mínima a partir de $\\Omega_s$"),
        Value(r"\text{Forma}", "racional", "dado", "$B(s)/A(s)$"),
    ),

    theory=r"""
**Butterworth (maximamente plano):** $|H_a(j\Omega)|^2 = 1/[1 + (\Omega/\Omega_c)^{2N}]$. A magnitude é monotônica (sem ondulações), vale $-3$ dB em $\Omega_c$ e decai $20N$ dB por década.

**Ordem e corte.** Impondo $R_p$ em $\Omega_p$ e $A_s$ em $\Omega_s$ e eliminando $\Omega_c$, obtém-se a ordem mínima. Como $N$ é arredondado para cima, sobra folga. O livro escolhe $\Omega_c$ para atender $R_p$ **exatamente**, e a folga vai toda para a rejeição.

**Polos.** $N$ pontos igualmente espaçados de $180°/N$ no semicírculo esquerdo de raio $\Omega_c$. Isso garante um filtro estável.

**Forma racional.** $H_a(s) = \Omega_c^N/\prod_k(s - p_k)$. Multiplicando os fatores, obtém-se um denominador mônico de grau $N$ com coeficientes reais e positivos (polinômio de Hurwitz).
""",
    deepening=(
        Deepening("De onde vem a fórmula da ordem?", r"""
De $1 + (\Omega_p/\Omega_c)^{2N} = 10^{R_p/10}$ e $1 + (\Omega_s/\Omega_c)^{2N} = 10^{A_s/10}$. Dividindo uma pela outra, $(\Omega_p/\Omega_s)^{2N} = (10^{R_p/10}-1)/(10^{A_s/10}-1)$. Tomando o logaritmo e isolando $N$, chega-se à fórmula. Arredonda-se para cima porque $N$ é inteiro.
"""),
        Deepening("Como ler a fase?", r"""
A fase total vai de 0 a $-N\pi/2 = -[[N]]\pi/2$ quando $\Omega \to \infty$. O gráfico mostra a **fase principal**, em $(-\pi, \pi]$, como a função `freqs_m` do livro. Os saltos de $2\pi$ são só a passagem por $\pm\pi$, não descontinuidades físicas.
"""),
    ),

    equations=(
        Equation(r"N = \left\lceil \frac{\log_{10}\left[(10^{R_p/10}-1)/(10^{A_s/10}-1)\right]}{2\log_{10}(\Omega_p/\Omega_s)} \right\rceil", "Ordem mínima", "", (), "DTA §3.2.A"),
        Equation(r"\Omega_c = \frac{\Omega_p}{(10^{R_p/10}-1)^{1/(2N)}}", "Frequência de corte", "Atende $R_p$ exatamente em $\\Omega_p$.", (), "DTA §3.2.A"),
        Equation(r"p_k = \Omega_c\, e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{2N}\right]}, \quad k = 0,\dots,N-1", "Polos", "", (), "DTA §3.2.A"),
        Equation(r"H_a(s) = \frac{\Omega_c^N}{\prod_{k=0}^{N-1}(s - p_k)} = \frac{B}{s^N + a_{N-1}s^{N-1} + \dots + a_0}", "Forma racional",
                 "Numerador constante; denominador mônico.", (Variable(r"a_0 = \Omega_c^N", "garante $H_a(0) = 1$"),), "DTA §4"),
    ),

    steps=(
        Step("Calcular os termos da fórmula da ordem", "Razão entre as exigências e razão entre as bordas.",
             r"\frac{10^{[[Rp10:g]]}-1}{10^{[[As10:g]]}-1} = \frac{[[num:.5f]]}{[[den:.0f]]}, \quad \log_{10}(\cdot) = [[log_ratio:.4f]], \quad 2\log_{10}\frac{[[Wp:g]]}{[[Ws:g]]} = [[log_w:.5f]]",
             (Value(r"10^{R_p/10}-1", "[[num:.5f]]", "calculado", "termo de passagem"),)),
        Step("Ordem mínima", "Arredondando para cima.",
             r"N = \left\lceil \frac{[[log_ratio:.4f]]}{[[log_w:.5f]]} \right\rceil = \lceil [[N_real:.4f]] \rceil = [[N]]",
             (Value("N", "[[N]]", "resultado", "ordem mínima"),)),
        Step("Frequência de corte", "Convenção do livro: $R_p$ exato em $\\Omega_p$.",
             r"\Omega_c = \frac{[[Wp:g]]}{([[num:.5f]])^{1/[[N2]]}} = \frac{[[Wp:g]]}{[[num_root:.5f]]} = [[Omega_c:.4f]]\ \text{rad/s}",
             (Value(r"\Omega_c", "[[Omega_c:.4f]] rad/s", "calculado", "corte de −3 dB"),
              Value(r"\text{critério}", "$R_p$ exato em $\\Omega_p$", "escolhido", "convenção do livro (`afd_butt`)"))),
        Step("Localizar os polos", "Espaçados de $180°/[[N]] = [[pole_step:g]]°$ no semicírculo esquerdo. Com $N$ par, não há polo real.\n\n[[pole_table]]",
             r"p_k = [[Omega_c:.2f]]\; e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{[[N2]]}\right]}",
             (Value(r"\text{polos}", "[[N]] (todos complexos)", "calculado", "N par: só pares conjugados"),)),
        Step("Multiplicar os fatores: forma racional",
             "Expandindo $\\prod_k (s - p_k)$, as partes imaginárias se cancelam em pares conjugados. Coeficientes de $A(s)$:\n\n[[coef_table]]",
             r"B = \Omega_c^{[[N]]} = [[B:tex7]] = a_0 \;\Rightarrow\; H_a(0) = \frac{B}{a_0} = [[H0:g]]",
             (Value("B", "$[[B:tex7]]$", "resultado", "numerador"),
              Value(r"H_a(0)", "[[H0:g]]", "resultado", "ganho DC unitário"))),
        Step("Verificar as exigências e a minimalidade", "Avaliando nas bordas, e com uma ordem a menos.",
             r"\text{perda}(\Omega_p) = [[loss_wp:.4f]]\ \text{dB}, \quad \text{aten.}(\Omega_s) = [[att_ws:.2f]]\ \text{dB}, \quad N=[[Nm1]]:\ [[att_lower:.2f]]\ \text{dB}",
             (Value(r"\text{atenuação em } \Omega_s", "[[att_ws:.2f]] dB", "resultado", "folga de [[margin_ws:.2f]] dB"),
              Value(r"N-1 = [[Nm1]]", "[[att_lower:.2f]] dB", "criterio", "< [[As:g]] dB: [[N]] é o mínimo"))),
    ),

    implementation=r"""
1. **Ordem, corte e polos** (`butter_order`, `butter_cutoff`, `butter_poles`): fórmulas do livro. Conferidos com `scipy.signal.buttord`/`butter(analog=True)` nos testes.
2. **Forma racional** (`forms.rational`): `zpk2tf` com descarte da parte imaginária residual, só se for desprezível (verificado).
3. **Análise**: magnitude e fase a partir dos polos (`freqs_zpk`); $h(t)$ simulada e conferida com a inversa de Laplace por resíduos.
""",
    code_roles=(
        CodeRole("dsp_av2.iir.prototypes.butter_order", "Passos 1–2: ordem mínima"),
        CodeRole("dsp_av2.iir.prototypes.butter_cutoff", "Passo 3: Ω_c"),
        CodeRole("dsp_av2.iir.prototypes.butter_poles", "Passo 4: polos"),
        CodeRole("dsp_av2.iir.forms.rational", "Passo 5: B(s)/A(s)"),
        CodeRole("dsp_av2.validation.validate_analog_lowpass", "Passo 6: bordas e minimalidade"),
    ),

    figures=(
        Figure("poles", "Polos no plano $s$", r"""
[[N]] polos sobre o círculo de raio $\Omega_c = [[Omega_c:.1f]]$ rad/s, espaçados de [[pole_step:g]]°. Os mais próximos do eixo $j\Omega$ (a $\pm[[closest_deg:g]]°$) definem a oscilação da resposta ao impulso.
"""),
        Figure("magnitude", "Magnitude", r"""
Plana até perto de $\Omega_p$ e decrescendo sem ondulações. Em $\Omega_p$ a magnitude vale exatamente $10^{-R_p/20}$.
"""),
        Figure("db", "Magnitude em dB", r"""
Perda de [[loss_wp:.2f]] dB em $\Omega_p$ e atenuação de [[att_ws:.2f]] dB em $\Omega_s$: [[margin_ws:.1f]] dB de folga, gerada pelo arredondamento de $N = [[N_real:.2f]]$ para [[N]].
"""),
        Figure("phase", "Fase", r"""
Fase principal em $(-\pi, \pi]$. A fase total decresce até $-[[N]]\pi/2$. O salto é só a volta por $-\pi$.
"""),
        Figure("impulse", "Resposta ao impulso", r"""
Oscilação amortecida: sobe, ultrapassa o eixo e decai. O amortecimento vem da parte real dos polos, e a oscilação, da parte imaginária do par mais próximo do eixo $j\Omega$.
"""),
    ),

    verification=r"""
Três critérios: perda máxima em $[0, \Omega_p]$ ≤ $R_p$; atenuação mínima em $[\Omega_s, 100\,\Omega_s]$ ≥ $A_s$; e o mesmo projeto com $N - 1$ não alcança $A_s$ (minimalidade). Perdas medidas em relação a $|H_a(j0)| = 1$.
""",
    conclusion=r"""
- **$N = [[N]]$** ($\lceil [[N_real:.2f]] \rceil$) e **$\Omega_c = [[Omega_c:.4f]]$ rad/s**.
- $H_a(s) = B/A(s)$ com $B = a_0 = \Omega_c^{[[N]]} = [[B:tex7]]$ e todos os coeficientes de $A(s)$ positivos. Os coeficientes coincidem com os do DTA.
- Perda de [[loss_wp:.2f]] dB em $\Omega_p$ (igual ao limite, por construção) e [[att_ws:.2f]] dB em $\Omega_s$. Com $N = [[Nm1]]$ seriam [[att_lower:.2f]] dB, insuficiente.
""",
)
