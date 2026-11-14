r"""Conteúdo didático — P8.13 (Butterworth analógico passa-baixas, forma em cascata).

Fonte técnica: DTA §2.2 item 10, §3.2.A e §4; Ingle & Proakis, Seção 8.2.1
(conforme DTA). Correção E6 de docs/errata_dta.md (ângulos θ_k). Todos os
números de resultado vêm de :func:`context`, que executa ``exercises.p8_13``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from dsp_av2.iir import analysis
from dsp_av2.iir.prototypes import butterworth_of_order
from dsp_av2.specs import P8_13 as SPEC
from exercises import p8_13

from .model import (CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value,
                    Variable, tex_sci)

TWO_PI = 2 * math.pi


@lru_cache(maxsize=1)
def objects() -> dict:
    proto = p8_13.design()
    return {
        "spec": SPEC,
        "proto": proto,
        "cascade": p8_13.cascade_form(proto),
        "report": p8_13.validate(proto),
        "lower": butterworth_of_order(proto.N - 1, SPEC.wp, SPEC.rp_db),
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    proto, c, rep = o["proto"], o["cascade"], o["report"]
    wc = proto.params["Omega_c"]
    num = 10 ** (SPEC.rp_db / 10) - 1
    den = 10 ** (SPEC.as_db / 10) - 1
    biquads = [s for s in c.sections if s.order == 2]
    thetas = np.degrees([np.arccos(s.den[1] / (2 * wc)) for s in biquads])

    # Tabela dos polos: ângulo absoluto e ângulo a partir do eixo real negativo
    rows = [r"| $k$ | ângulo de $p_k$ | a partir do eixo real negativo | $p_k$ (rad/s) |", "|---|---|---|---|"]
    for k, p in enumerate(proto.p):
        ang = np.degrees(np.angle(p)) % 360
        rows.append(f"| {k} | {ang:.0f}° | {abs(180 - ang):.0f}° | ${p.real:,.1f} {'+' if p.imag >= 0 else '-'} j{abs(p.imag):,.1f}$ |".replace(",", "\\,"))

    # Tabela das seções
    srows = [r"| Seção | $\theta$ | Coef. de $s$: $2\Omega_c\cos\theta$ | Termo constante | Numerador |", "|---|---|---|---|---|"]
    first = [s for s in c.sections if s.order == 1][0]
    srows.append(f"| 1ª ordem | 0° | — | ${tex_sci(first.den[1], 7)}$ | ${tex_sci(first.num[0], 7)}$ |")
    for i, (s, th) in enumerate(sorted(zip(biquads, thetas), key=lambda z: z[1]), start=1):
        srows.append(f"| Biquad {i} | {th:.0f}° | ${tex_sci(s.den[1], 7)}$ | ${tex_sci(s.den[2], 7)}$ | ${tex_sci(s.num[0], 7)}$ |")

    gd0 = analysis.group_delay(proto, np.array([0.0]))[0]
    wg = np.linspace(0, 2 * SPEC.ws, 40001)
    gd = analysis.group_delay(proto, wg)
    att_lower = -analysis.magnitude_db(analysis.frequency_response(o["lower"], np.array([SPEC.ws])))[0]
    att_ws = -analysis.magnitude_db(analysis.frequency_response(proto, np.array([SPEC.ws])))[0]
    loss_wp = -analysis.magnitude_db(analysis.frequency_response(proto, np.array([SPEC.wp])))[0]

    return {
        "Fp": SPEC.wp / TWO_PI, "Fs": SPEC.ws / TWO_PI, "Fp_k": SPEC.wp / TWO_PI / 1e3, "Fs_k": SPEC.ws / TWO_PI / 1e3,
        "Rp": SPEC.rp_db, "As": SPEC.as_db, "Rp10": SPEC.rp_db / 10, "As10": SPEC.as_db / 10,
        "Wp": SPEC.wp, "Ws": SPEC.ws, "ratio": SPEC.wp / SPEC.ws,
        "num": num, "den": den, "log_ratio": math.log10(num / den),
        "log_w": 2 * math.log10(SPEC.wp / SPEC.ws),
        "N_real": proto.params["N_real"], "N": proto.N, "N2": 2 * proto.N, "Nm1": proto.N - 1,
        "n_pairs": len(biquads), "pole_step": 180 / proto.N,
        "Omega_c": wc, "Fc": wc / TWO_PI, "Fc_k": wc / TWO_PI / 1e3, "num_root": num ** (1 / (2 * proto.N)),
        "pole_table": "\n".join(rows), "section_table": "\n".join(srows),
        "theta_list": ", ".join(f"{t:.0f}°" for t in sorted(thetas)),
        "gain": c.gain, "a0_tf": wc ** proto.N,
        "loss_wp": loss_wp, "att_ws": att_ws, "margin_ws": att_ws - SPEC.as_db, "att_lower": att_lower,
        "gd0_ms": 1e3 * gd0, "gd_peak_ms": 1e3 * gd.max(), "gd_peak_kHz": wg[np.argmax(gd)] / TWO_PI / 1e3,
        "passed": rep.passed,
    }


CONTENT = ExerciseContent(
    id="p8_13",
    number="P8.13",
    chapter=8,
    title="Butterworth analógico passa-baixas em forma de cascata",
    tagline="IIR · protótipo analógico · Butterworth · cascata",
    highlights=(
        Value(r"N", "[[N]]", "resultado", "ordem mínima"),
        Value(r"F_c", "[[Fc:.2f]] Hz", "calculado", "frequência de corte (−3 dB)"),
        Value(r"\theta_k", "[[theta_list]]", "calculado", "ângulos dos biquads"),
        Value(r"A(F_s)", "[[att_ws:.2f]] dB", "resultado", "atenuação em [[Fs_k:g]] kHz"),
    ),
    errata=("E6",),
    result_keys=("Omega_c", "Fc", "N_real", "att_ws", "att_lower", "a0_tf"),

    # 2 ──────────────────────────────────────────────────────────────────────
    asked=r"""
Projetar um **filtro analógico passa-baixas de Butterworth**, o protótipo contínuo que os métodos do Capítulo 8 depois convertem em filtro digital, que:

- perca **no máximo [[Rp:g]] dB** até [[Fp_k:g]] kHz (faixa de passagem);
- atenue **pelo menos [[As:g]] dB** a partir de [[Fs_k:g]] kHz (faixa de rejeição).

O resultado deve ser dado como **função de sistema em cascata**, ou seja, um produto de seções de 1ª e 2ª ordem. São pedidos quatro gráficos: magnitude, magnitude em dB, **atraso de grupo** e resposta ao impulso.

Na prática, é preciso encontrar a **menor ordem** $N$ que atende às duas exigências, a frequência de corte $\Omega_c$, os polos e a fatoração em seções.
""",

    # 3 ──────────────────────────────────────────────────────────────────────
    data_note=r"""
O enunciado dá as frequências em **hertz**. As fórmulas do Butterworth usam frequência angular, então a conversão $\Omega = 2\pi F$ é o primeiro cálculo da resolução.
""",
    data_rows=(
        Value(r"F_p", "[[Fp_k:g]] kHz", "dado", "borda da faixa de passagem"),
        Value(r"R_p", "[[Rp:g]] dB", "dado", "perda máxima admitida até $F_p$"),
        Value(r"F_s", "[[Fs_k:g]] kHz", "dado", "borda da faixa de rejeição"),
        Value(r"A_s", "[[As:g]] dB", "dado", "atenuação mínima exigida a partir de $F_s$"),
        Value(r"\text{Aproximação}", "Butterworth", "dado", "imposta pelo enunciado"),
        Value(r"\text{Forma}", "cascata", "dado", "forma pedida para a função de sistema"),
    ),

    # 4 ──────────────────────────────────────────────────────────────────────
    theory=r"""
**Resposta maximamente plana.** O Butterworth de ordem $N$ é definido pela magnitude ao quadrado

$$|H_a(j\Omega)|^2 = \frac{1}{1 + (\Omega/\Omega_c)^{2N}}.$$

Em $\Omega = 0$, as primeiras $2N-1$ derivadas são nulas. A curva é a mais plana possível na origem e decresce **monotonicamente**, sem ondulações. Em $\Omega = \Omega_c$ a magnitude vale $1/\sqrt{2}$ ($-3$ dB). Quanto maior $N$, mais abrupta a queda: $20N$ dB por década na rejeição.

**Onde ficam os polos.** Os polos de $H_a(s)H_a(-s)$ são as raízes de $1 + (s/j\Omega_c)^{2N} = 0$: $2N$ pontos igualmente espaçados num círculo de raio $\Omega_c$. Para o filtro ser **estável**, $H_a(s)$ fica só com os $N$ polos do semiplano esquerdo, espaçados de $180°/N$.

**Por que cascata.** Polos complexos aparecem em pares conjugados $p, p^*$, e cada par forma um polinômio real de 2ª ordem (um **biquad**). Com $N$ ímpar sobra um polo real, que forma uma seção de 1ª ordem. A forma em cascata é a usada na prática, porque a forma racional de ordem alta tem coeficientes com enorme faixa dinâmica (aqui, o termo constante seria $\Omega_c^{[[N]]} \approx [[a0_tf:tex3]]$) e é numericamente frágil.
""",
    deepening=(
        Deepening("De onde vem a fórmula da ordem $N$?", r"""
Impondo as duas exigências com igualdade:

$$1 + \left(\frac{\Omega_p}{\Omega_c}\right)^{2N} = 10^{R_p/10}, \qquad 1 + \left(\frac{\Omega_s}{\Omega_c}\right)^{2N} = 10^{A_s/10}.$$

Dividindo uma pela outra, $\Omega_c$ desaparece e resta $(\Omega_p/\Omega_s)^{2N} = (10^{R_p/10}-1)/(10^{A_s/10}-1)$. Tomando $\log_{10}$ e isolando $N$, obtém-se a fórmula usada. Como $N$ precisa ser inteiro, arredonda-se **para cima**, e por isso sobra folga em pelo menos uma das bordas.
"""),
        Deepening("Por que escolher $\\Omega_c$ pela borda de passagem?", r"""
Com $N$ arredondado para cima, há um intervalo de valores de $\Omega_c$ que atendem às duas bordas. O livro (função `afd_butt`, conforme o DTA) escolhe $\Omega_c$ para atender $R_p$ **exatamente** em $\Omega_p$. Toda a folga do arredondamento fica na rejeição. É uma convenção, não uma imposição do enunciado, e a rotina `scipy.signal.buttord` adota o mesmo critério.
"""),
        Deepening("O que é o atraso de grupo?", r"""
$\tau(\Omega) = -d\phi/d\Omega$ é o atraso, em segundos, sofrido pela **envoltória** de um sinal de banda estreita centrado em $\Omega$. Se $\tau$ fosse constante, todas as componentes chegariam juntas, sem distorção de fase. No Butterworth, $\tau$ cresce perto do corte, de modo que componentes próximas a $F_c$ chegam mais atrasadas.

Para um filtro só com polos $p_k = \sigma_k + j\omega_k$, cada polo contribui com $\frac{-\sigma_k}{\sigma_k^2 + (\Omega - \omega_k)^2}$, um "pico" centrado em $\omega_k$ que fica mais alto e estreito quanto mais o polo se aproxima do eixo $j\Omega$.
"""),
    ),

    # 5 ──────────────────────────────────────────────────────────────────────
    equations=(
        Equation(r"\Omega = 2\pi F", "Conversão de frequência", "Hertz para radianos por segundo.",
                 (Variable(r"F", "frequência (Hz)"), Variable(r"\Omega", "frequência angular (rad/s)"))),
        Equation(r"N = \left\lceil \frac{\log_{10}\left[(10^{R_p/10}-1)/(10^{A_s/10}-1)\right]}{2\log_{10}(\Omega_p/\Omega_s)} \right\rceil",
                 "Ordem mínima", "Menor ordem inteira que atende simultaneamente $R_p$ em $\\Omega_p$ e $A_s$ em $\\Omega_s$.",
                 (Variable(r"R_p, A_s", "perda máxima e atenuação mínima (dB)"),
                  Variable(r"\Omega_p, \Omega_s", "bordas de passagem e de rejeição (rad/s)")), "DTA §3.2.A"),
        Equation(r"\Omega_c = \frac{\Omega_p}{(10^{R_p/10}-1)^{1/(2N)}}", "Frequência de corte (−3 dB)",
                 "Escolhida para que a perda em $\\Omega_p$ seja exatamente $R_p$.",
                 (Variable(r"\Omega_c", "frequência de corte de −3 dB (rad/s)"),), "DTA §3.2.A"),
        Equation(r"p_k = \Omega_c\, e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{2N}\right]}, \quad k = 0, \dots, N-1",
                 "Polos no semiplano esquerdo", "Igualmente espaçados de $\\pi/N$ sobre o círculo de raio $\\Omega_c$.",
                 (Variable(r"p_k", "k-ésimo polo"),), "DTA §3.2.A"),
        Equation(r"H_a(s) = \frac{\Omega_c}{s + \Omega_c}\prod_{k=1}^{(N-1)/2}\frac{\Omega_c^2}{s^2 + 2\Omega_c\cos\theta_k\, s + \Omega_c^2}, \qquad \theta_k = \frac{k\pi}{N}",
                 "Forma em cascata (N ímpar)",
                 "Uma seção de 1ª ordem (polo real em $-\\Omega_c$) e $(N-1)/2$ biquads, cada um com ganho unitário em $s = 0$.",
                 (Variable(r"\theta_k", "ângulo do k-ésimo par de polos medido a partir do eixo real negativo"),), "Errata E6 (corrige o DTA §4)"),
        Equation(r"\tau(\Omega) = -\frac{d\phi}{d\Omega} = \sum_k \frac{-\sigma_k}{\sigma_k^2 + (\Omega - \omega_k)^2}",
                 "Atraso de grupo (analítico)", "Soma das contribuições de cada polo $p_k = \\sigma_k + j\\omega_k$.",
                 (Variable(r"\phi(\Omega)", "fase de $H_a(j\\Omega)$"),
                  Variable(r"\sigma_k, \omega_k", "partes real e imaginária do polo")), ""),
    ),

    # 6 ──────────────────────────────────────────────────────────────────────
    steps=(
        Step("Converter as frequências para rad/s",
             "As fórmulas usam frequência angular.",
             r"\Omega_p = 2\pi \cdot [[Fp:.0f]] = [[Wp:.2f]]\ \text{rad/s}, \qquad \Omega_s = 2\pi \cdot [[Fs:.0f]] = [[Ws:.2f]]\ \text{rad/s}",
             (Value(r"\Omega_p", "[[Wp:.2f]] rad/s", "calculado", "borda de passagem"),
              Value(r"\Omega_s", "[[Ws:.2f]] rad/s", "calculado", "borda de rejeição"))),
        Step("Calcular os termos da fórmula da ordem",
             "Numerador: a razão entre as exigências de passagem e de rejeição. Denominador: a razão entre as bordas, que só depende de $F_p/F_s = [[ratio:g]]$.",
             r"\frac{10^{[[Rp10:g]]}-1}{10^{[[As10:g]]}-1} = \frac{[[num:.5f]]}{[[den:.2f]]}, \qquad \log_{10}(\cdot) = [[log_ratio:.4f]], \qquad 2\log_{10}([[ratio:g]]) = [[log_w:.5f]]",
             (Value(r"10^{R_p/10}-1", "[[num:.5f]]", "calculado", "termo de passagem"),
              Value(r"10^{A_s/10}-1", "[[den:.2f]]", "calculado", "termo de rejeição"))),
        Step("Determinar a ordem mínima",
             "Dividindo os logaritmos e arredondando **para cima**, porque a ordem precisa ser inteira e não pode ficar abaixo do necessário:",
             r"N = \left\lceil \frac{[[log_ratio:.4f]]}{[[log_w:.5f]]} \right\rceil = \lceil [[N_real:.4f]] \rceil = [[N]]",
             (Value(r"N", "[[N]]", "resultado", "ordem mínima"),),
             note="$[[N_real:.4f]]$ está muito perto do inteiro [[N]], então o arredondamento deixa **pouca folga** na rejeição. Isso é verificado no último passo."),
        Step("Calcular a frequência de corte",
             "Convenção do livro: atender $R_p$ exatamente em $\\Omega_p$ (ver aprofundamento).",
             r"\Omega_c = \frac{[[Wp:.2f]]}{([[num:.5f]])^{1/[[N2]]}} = \frac{[[Wp:.2f]]}{[[num_root:.5f]]} = [[Omega_c:.2f]]\ \text{rad/s} \quad (F_c = [[Fc:.2f]]\ \text{Hz})",
             (Value(r"\Omega_c", "[[Omega_c:.4f]] rad/s", "calculado", "corte de −3 dB"),
              Value(r"\text{critério de } \Omega_c", "$R_p$ exato em $\\Omega_p$", "escolhido", "convenção do livro (`afd_butt`)"))),
        Step("Localizar os polos",
             "Com $N = [[N]]$, os polos ficam espaçados de $180°/[[N]] = [[pole_step:g]]°$ no semiplano esquerdo. Como $N$ é ímpar, um deles cai exatamente sobre o eixo real negativo:\n\n[[pole_table]]",
             r"p_k = [[Omega_c:.2f]]\; e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{[[N2]]}\right]}",
             (Value(r"\text{polo real}", "$-\\Omega_c$", "calculado", "gera a seção de 1ª ordem"),
              Value(r"\text{pares conjugados}", "[[n_pairs]]", "calculado", "geram os biquads"))),
        Step("Formar as seções da cascata",
             "Um par conjugado a um ângulo $\\theta$ do eixo real negativo é $p = \\Omega_c(-\\cos\\theta \\pm j\\sin\\theta)$, e o produto $(s-p)(s-p^*)$ vale $s^2 + 2\\Omega_c\\cos\\theta\\,s + \\Omega_c^2$. Os ângulos são **[[theta_list]]**, ou seja, $\\theta_k = k\\pi/[[N]]$. Cada numerador é igual ao termo constante do denominador, o que dá ganho 1 em $s = 0$.\n\n[[section_table]]",
             r"\theta_k = \frac{k\pi}{[[N]]},\ k = 1,\dots,[[n_pairs]] \qquad \Longrightarrow \qquad G = [[gain:g]]",
             (Value(r"\theta_k", "[[theta_list]]", "calculado", "ângulos dos pares"),
              Value(r"G", "[[gain:g]]", "resultado", "ganho global: $H_a(0) = 1$")),
             note="**Errata E6.** O DTA usa $\\theta_k = (2k-1)\\pi/18$ (10°, 30°, 50°, 70°). Essa é a fórmula para **N par**, em que não há polo real. Com $N = [[N]]$, os ângulos corretos são $k\\pi/[[N]]$."),
        Step("Verificar as duas exigências e a minimalidade",
             "Avaliando $|H_a(j\\Omega)|$ nas bordas, e também o mesmo projeto com uma ordem a menos:",
             r"-20\log_{10}|H_a(j\Omega_p)| = [[loss_wp:.4f]]\ \text{dB}, \quad -20\log_{10}|H_a(j\Omega_s)| = [[att_ws:.2f]]\ \text{dB}, \quad N=[[Nm1]]:\ [[att_lower:.2f]]\ \text{dB}",
             (Value(r"\text{perda em } F_p", "[[loss_wp:.4f]] dB", "resultado", "≤ [[Rp:g]] dB (igualdade por construção)"),
              Value(r"\text{atenuação em } F_s", "[[att_ws:.2f]] dB", "resultado", "≥ [[As:g]] dB, folga de [[margin_ws:.2f]] dB"),
              Value(r"N - 1 = [[Nm1]]", "[[att_lower:.2f]] dB", "criterio", "< [[As:g]] dB: confirma que [[N]] é o mínimo"))),
    ),

    # 7 ──────────────────────────────────────────────────────────────────────
    implementation=r"""
O projeto segue as fórmulas do livro, implementadas explicitamente no pacote `dsp_av2`. As rotinas do SciPy entram só como verificação independente nos testes.

1. **Especificação** (`specs.P8_13`): frequências em Hz convertidas por $\Omega = 2\pi F$.
2. **Ordem e corte** (`butter_order`, `butter_cutoff`): passos 2 a 4.
3. **Polos** (`butter_poles`): passo 5. Coincidem com `scipy.signal.butter(..., analog=True)` em ~$10^{-13}$.
4. **Cascata** (`forms.cascade`): agrupa os pares conjugados em biquads com ganho DC unitário (passo 6). Um teste confirma que o produto das seções reproduz exatamente a resposta em frequência do filtro.
5. **Análise**: magnitude a partir dos polos (`freqs_zpk`), atraso de grupo pela fórmula analítica (conferida com $-d\phi/d\Omega$ numérico) e $h(t)$ por simulação, conferida com a transformada inversa de Laplace por resíduos.
""",
    code_roles=(
        CodeRole("dsp_av2.iir.prototypes.butter_order", "Passos 2–3: ordem mínima N"),
        CodeRole("dsp_av2.iir.prototypes.butter_cutoff", "Passo 4: Ω_c com R_p exato em Ω_p"),
        CodeRole("dsp_av2.iir.prototypes.butter_poles", "Passo 5: polos no semiplano esquerdo"),
        CodeRole("dsp_av2.iir.forms.cascade", "Passo 6: seções de 1ª e 2ª ordem"),
        CodeRole("dsp_av2.iir.analysis.group_delay", "Atraso de grupo analítico"),
        CodeRole("dsp_av2.validation.validate_analog_lowpass", "Passo 7: verificação das bordas e da minimalidade"),
    ),

    # 8 ──────────────────────────────────────────────────────────────────────
    figures=(
        Figure("poles", "Polos no plano $s$", r"""
Os [[N]] polos ficam sobre o círculo de raio $\Omega_c$ ($F_c = [[Fc_k:.2f]]$ kHz), espaçados de [[pole_step:g]]°, todos no semiplano esquerdo (filtro estável). O polo sobre o eixo real gera a seção de 1ª ordem, e cada par simétrico gera um biquad. Os ângulos medidos a partir do eixo real negativo, [[theta_list]], são os $\theta_k$ da cascata.
"""),
        Figure("magnitude", "Magnitude", r"""
Plana até perto de [[Fp_k:g]] kHz e decrescendo sem ondulações (monotônica). Em $F_p$ a magnitude vale $10^{-R_p/20}$, exatamente no limite, porque $\Omega_c$ foi escolhido assim.
"""),
        Figure("db", "Magnitude em dB", r"""
A curva cruza $-[[Rp:g]]$ dB em [[Fp_k:g]] kHz e chega a $-[[att_ws:.2f]]$ dB em [[Fs_k:g]] kHz, só [[margin_ws:.2f]] dB além do exigido, porque $N_{\text{real}} = [[N_real:.3f]]$ quase não precisou ser arredondado.
"""),
        Figure("group_delay", "Atraso de grupo", r"""
Começa em [[gd0_ms:.3f]] ms (DC) e chega a [[gd_peak_ms:.3f]] ms perto de [[gd_peak_kHz:.1f]] kHz, junto ao corte. O pico vem do par de polos mais próximo do eixo $j\Omega$ (o de maior $\theta$). Componentes do sinal perto de $F_c$ sofrem atraso maior que as de baixa frequência.
"""),
        Figure("impulse", "Resposta ao impulso", r"""
Oscilação amortecida com frequência próxima de $F_c$ (período ≈ $1/F_c$), que se extingue em menos de 1 ms. A oscilação é a "assinatura" no tempo dos polos complexos de baixo amortecimento.
"""),
    ),

    # 9 ──────────────────────────────────────────────────────────────────────
    verification=r"""
Três critérios: (1) a perda em toda a faixa $0 \le \Omega \le \Omega_p$ não excede $R_p$; (2) a atenuação em toda a faixa $\Omega_s \le \Omega \le 100\,\Omega_s$ é pelo menos $A_s$; (3) **minimalidade**: o mesmo projeto com ordem $N - 1$ não alcança $A_s$ em $\Omega_s$. As perdas são medidas em relação ao ganho máximo $|H_a(j0)| = 1$.
""",

    # 10 ─────────────────────────────────────────────────────────────────────
    conclusion=r"""
- Ordem mínima **$N = [[N]]$** ($\lceil [[N_real:.4f]] \rceil$) e corte **$\Omega_c = [[Omega_c:.2f]]$ rad/s** ($F_c = [[Fc:.2f]]$ Hz).
- Cascata: uma seção de 1ª ordem $\Omega_c/(s + \Omega_c)$ e [[n_pairs]] biquads com $\theta_k = k\pi/[[N]]$ ([[theta_list]]), com ganho global $G = [[gain:g]]$.
- Perda de [[loss_wp:.4f]] dB em [[Fp_k:g]] kHz (igual ao limite, por construção) e atenuação de [[att_ws:.2f]] dB em [[Fs_k:g]] kHz. Com $N = [[Nm1]]$ seriam [[att_lower:.2f]] dB, insuficiente.
- **Correção do DTA (E6):** os ângulos $(2k-1)\pi/18$ do DTA valem para ordem par. Para $N = [[N]]$, o correto é $k\pi/[[N]]$.
""",
)
