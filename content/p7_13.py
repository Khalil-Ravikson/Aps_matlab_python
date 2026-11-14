r"""Conteúdo didático — P7.13 (rejeita-faixa de comprimento mínimo, janela de Kaiser).

Fonte técnica: DTA §2.1 item 5, §3.1.D e §4; Ingle & Proakis, Seção 7.5
(conforme DTA). Correção E1 de docs/errata_dta.md. Todos os números de
resultado vêm de :func:`context`, que executa ``exercises.p7_13``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_kaiser
from dsp_av2.specs import P7_13 as SPEC
from dsp_av2.validation import validate_fir_tolerance
from exercises import p7_13

from .model import (CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value,
                    Variable)


@lru_cache(maxsize=1)
def objects() -> dict:
    """Objetos calculados pelo núcleo (projetos, busca, relatórios)."""
    est = p7_13.estimate()
    search = p7_13.search()
    final = search.best
    return {
        "spec": SPEC,
        "estimate": est,
        "estimate_report": validate_fir_tolerance(est),
        "search": search,
        "search_reports": {M: validate_fir_tolerance(design_kaiser(SPEC, M=M)) for M, _ in search.tried},
        "final": final,
        "final_report": p7_13.validate(final),
    }


@lru_cache(maxsize=1)
def context() -> dict:
    """Valores usados nos marcadores ``[[...]]`` do texto."""
    o = objects()
    est, final, s = o["estimate"], o["final"], o["search"]
    b1, b2, b3 = SPEC.bands
    As = est.params["As_design_dB"]
    dw = est.transition_width
    x = (As - 7.95) / (2.285 * dw * np.pi)
    er, fr = o["estimate_report"], o["final_report"]

    rows = ["| $M$ | Tipo | máx. desvio na passagem | máx. $\\lvert H\\rvert$ na rejeição | atende? |",
            "|---|---|---|---|---|"]
    for M, ok in s.tried[-5:]:
        r = o["search_reports"][M]
        rows.append(f"| {M} | {r.linear_phase_type} | {r.checks[0].measured:.5f} | "
                    f"{r.checks[1].measured:.5f} | {'✓ sim' if ok else '✗ não'} |")

    return {
        # dados do enunciado
        "b1_lo": b1.lo, "b1_hi": b1.hi, "b2_lo": b2.lo, "b2_hi": b2.hi, "b3_lo": b3.lo, "b3_hi": b3.hi,
        "d1": b1.delta, "d2": b2.delta,
        "g1_lo": 1 - b1.delta, "g1_hi": 1 + b1.delta,
        # cálculos de Kaiser
        "delta": est.params["delta_min"], "As": As, "As_m21": As - 21, "beta": est.params["beta"],
        "dw": dw, "x": x, "x_ceil": int(np.ceil(round(x, 9))), "M_est": est.M,
        "wc1": est.cutoffs[0], "wc2": est.cutoffs[1],
        # verificação da estimativa
        "est_dev_pass": er.checks[0].measured, "est_dev_stop": er.checks[1].measured,
        "est_excess_pct": 100 * (er.checks[1].measured / b2.delta - 1),
        # busca e resultado final
        "search_lo": s.tried[0][0], "search_table": "\n".join(rows),
        "M": final.M, "alpha": final.alpha, "type": fr.linear_phase_type,
        "dev_pass": fr.checks[0].measured, "dev_stop": fr.checks[1].measured,
        "margin_stop": fr.checks[1].margin, "margin_pass": fr.checks[0].margin,
        "n_grid": analysis.DENSE_GRID_POINTS,
        "pass_slack_ratio": b1.delta / fr.checks[0].measured,
    }


CONTENT = ExerciseContent(
    id="p7_13",
    number="P7.13",
    chapter=7,
    title="Rejeita-faixa de comprimento mínimo com janela de Kaiser",
    tagline="FIR · rejeita-faixa · janela de Kaiser · comprimento mínimo",
    highlights=(
        Value(r"M_{\min}", "[[M]]", "resultado", "comprimento mínimo verificado"),
        Value(r"\beta", "[[beta:.4f]]", "calculado", "parâmetro da janela de Kaiser"),
        Value(r"A_s", "[[As:.2f]] dB", "calculado", "atenuação de projeto"),
        Value(r"M_{\text{est}}", "[[M_est]]", "calculado", "estimativa de Kaiser: não atende"),
    ),
    errata=("E1",),
    result_keys=("M", "M_est", "beta", "As", "dev_stop", "est_dev_stop"),

    # 2 ──────────────────────────────────────────────────────────────────────
    asked=r"""
Projetar um **filtro FIR de fase linear do tipo rejeita-faixa**: ele deve deixar passar as frequências baixas e altas e bloquear uma faixa central.

- Nas faixas de passagem, o ganho pode variar no máximo ±2.5% em torno de 1.
- Na faixa de rejeição, o ganho não pode passar de 0.005, ou seja, 0.5% da amplitude de entrada.

O projeto deve usar a **janela de Kaiser**, e o enunciado pede o **menor comprimento** $M$ possível. $M$ é o número de coeficientes do filtro, e portanto o custo de implementação: cada amostra de saída exige $M$ multiplicações.

Também são pedidos dois gráficos: a **resposta de amplitude** $H_r(\omega)$ e a **magnitude em dB**.
""",

    # 3 ──────────────────────────────────────────────────────────────────────
    data_note=r"""
O enunciado dá as exigências como faixas de $|H(e^{j\omega})|$. Reescrevê-las como **ganho ideal $G$ ± tolerância $\delta$** é uma simples releitura dos mesmos números: $0.975 \le |H| \le 1.025$ equivale a $G = 1$ e $\delta_1 = 0.025$.
""",
    data_rows=(
        Value(r"\text{Passagem inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "faixa onde o sinal deve passar"),
        Value(r"\text{Rejeição}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", "faixa a ser bloqueada"),
        Value(r"\text{Passagem superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", "faixa onde o sinal deve passar"),
        Value(r"\delta_1", r"$[[d1:g]]$", "dado", r"tolerância na passagem: $[[g1_lo:g]] \le |H| \le [[g1_hi:g]]$"),
        Value(r"\delta_2", r"$[[d2:g]]$", "dado", r"tolerância na rejeição: $|H| \le [[d2:g]]$"),
        Value(r"\text{Método}", "janela de Kaiser, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    # 4 ──────────────────────────────────────────────────────────────────────
    theory=r"""
**Fase linear.** Se os coeficientes são simétricos, $h(n) = h(M-1-n)$, a resposta em frequência se fatora em

$$H(e^{j\omega}) = H_r(\omega)\,e^{-j\omega\alpha}, \qquad \alpha = \frac{M-1}{2}.$$

$H_r(\omega)$ é uma função **real** (a resposta de amplitude) e o termo exponencial é um atraso puro de $\alpha$ amostras, igual para todas as frequências. O filtro não distorce a forma de onda na faixa de passagem: só a atrasa.

**Tipo 1 × Tipo 2.** Com $M$ ímpar (Tipo 1), $\alpha$ é inteiro e não há restrição. Com $M$ par (Tipo 2), $H_r(\pi) = 0$ obrigatoriamente. Um rejeita-faixa precisa deixar passar $\omega = \pi$, então **só o Tipo 1 serve**: $M$ tem de ser ímpar.

**Método da janela.** O filtro ideal tem resposta ao impulso $h_d(n)$ infinita. Para obter um FIR, multiplica-se $h_d(n)$ por uma janela $w(n)$ de comprimento $M$. A multiplicação no tempo equivale a uma convolução na frequência, com dois efeitos:

- o **lóbulo principal** da janela alarga as bordas e cria a **faixa de transição**;
- os **lóbulos laterais** criam **ondulações** na passagem e na rejeição, de amplitudes praticamente iguais ($\delta_{\text{passagem}} \approx \delta_{\text{rejeição}}$).

**Janela de Kaiser.** O parâmetro $\beta$ controla a troca entre largura do lóbulo principal e altura dos lóbulos laterais. Fórmulas empíricas (Seção 7.5 do livro, conforme o DTA) dão $\beta$ a partir da atenuação desejada e uma **estimativa** de $M$ a partir da largura de transição.
""",
    deepening=(
        Deepening("Por que projetar pela menor tolerância?", r"""
No método da janela, as ondulações da passagem e da rejeição saem praticamente iguais, porque ambas vêm dos mesmos lóbulos laterais. Não dá para pedir $\delta_1 = 0.025$ numa faixa e $\delta_2 = 0.005$ na outra: o projeto atende as duas apenas se a ondulação comum for $\le \min(\delta_1, \delta_2) = 0.005$.

A consequência é que a passagem fica **bem melhor que o necessário**, e essa folga é desperdiçada. Métodos que ponderam cada faixa separadamente (Parks-McClellan, Seção 7.6) aproveitam a folga e obtêm filtros mais curtos.
"""),
        Deepening("O que é a função de Bessel $I_0$?", r"""
$I_0(x) = \sum_{k=0}^{\infty} \left[\frac{(x/2)^k}{k!}\right]^2$ é a função de Bessel modificada de primeira espécie e ordem zero. Na janela de Kaiser, ela produz uma forma em sino cuja "abertura" é controlada por $\beta$: com $\beta = 0$ a janela é retangular, e quanto maior $\beta$, mais suaves as bordas, maior a atenuação e mais larga a transição.
"""),
        Deepening("Por que a estimativa de $M$ pode falhar?", r"""
A fórmula $M \approx \frac{A_s - 7.95}{2.285\,\Delta\omega} + 1$ é um **ajuste de curvas** obtido empiricamente por Kaiser. Ela acerta com erro típico de ±1–2 amostras, mas não garante o atendimento. Quando o enunciado pede o comprimento **mínimo**, a resposta precisa ser **medida**, e é isso que este exercício mostra.
"""),
    ),

    # 5 ──────────────────────────────────────────────────────────────────────
    equations=(
        Equation(r"A_s = -20\log_{10}\delta", "Atenuação de projeto",
                 "Converte a tolerância linear $\\delta$ em decibéis.",
                 (Variable(r"\delta", "tolerância linear usada no projeto"),
                  Variable(r"A_s", "atenuação mínima correspondente (dB)")), "DTA §3.1.A"),
        Equation(r"\beta = \begin{cases} 0.1102(A_s - 8.7), & A_s \ge 50 \\ 0.5842(A_s-21)^{0.4} + 0.07886(A_s-21), & 21 < A_s < 50 \\ 0, & A_s \le 21 \end{cases}",
                 "Parâmetro de forma de Kaiser",
                 "Fórmula empírica que escolhe a forma da janela para atingir $A_s$.",
                 (Variable(r"\beta", "parâmetro de forma da janela de Kaiser"),), "DTA §3.1.D"),
        Equation(r"M \approx \left\lceil \frac{A_s - 7.95}{2.285\,\Delta\omega} \right\rceil + 1",
                 "Estimativa do comprimento",
                 "Estimativa empírica do número de coeficientes necessário. É um ponto de partida, não uma garantia.",
                 (Variable(r"\Delta\omega", "largura da faixa de transição (rad/amostra)"),
                  Variable(r"\lceil\cdot\rceil", "arredondamento para cima (teto)"),
                  Variable(r"M", "comprimento do filtro (número de coeficientes)")), "DTA §3.1.D"),
        Equation(r"w(n) = \frac{I_0\!\left(\beta\sqrt{1-\left(\frac{2n}{M-1}-1\right)^2}\right)}{I_0(\beta)}, \quad 0 \le n \le M-1",
                 "Janela de Kaiser", "Coeficientes da janela.",
                 (Variable(r"I_0", "função de Bessel modificada de ordem zero"),
                  Variable(r"n", "índice do coeficiente")), "DTA §3.1.D"),
        Equation(r"h_d(n) = \delta(n-\alpha) - \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}",
                 "Resposta ideal do rejeita-faixa",
                 "Um passa-tudo (impulso atrasado) menos um passa-faixa ideal.",
                 (Variable(r"\omega_{c1}, \omega_{c2}", "frequências de corte, no meio das transições"),
                  Variable(r"\alpha = (M-1)/2", "atraso (centro de simetria)"),
                  Variable(r"\delta(n-\alpha)", "impulso unitário atrasado; exige $\\alpha$ inteiro, ou seja, $M$ ímpar")), "DTA §3.1.E"),
        Equation(r"h(n) = h_d(n)\,w(n)", "Filtro projetado",
                 "Truncamento e suavização da resposta ideal pela janela.", (), "DTA §3.1"),
        Equation(r"H_r(\omega) = \mathrm{Re}\left\{H(e^{j\omega})\,e^{j\omega\alpha}\right\}", "Resposta de amplitude",
                 "Remove o atraso linear e deixa a resposta real, com sinal, usada no gráfico pedido.",
                 (Variable(r"H(e^{j\omega})", "resposta em frequência (DTFT de $h(n)$)"),), "Ingle & Proakis, Seção 7.2"),
    ),

    # 6 ──────────────────────────────────────────────────────────────────────
    steps=(
        Step("Ler as tolerâncias",
             "As duas tolerâncias vêm diretamente do enunciado.",
             r"\delta_1 = [[d1:g]] \;(\text{passagem}), \qquad \delta_2 = [[d2:g]] \;(\text{rejeição})",
             (Value(r"\delta_1", "[[d1:g]]", "dado", "tolerância de passagem"),
              Value(r"\delta_2", "[[d2:g]]", "dado", "tolerância de rejeição"))),
        Step("Escolher a tolerância de projeto",
             "Como a janela produz ondulações praticamente iguais nas duas faixas (ver Fundamentação), o projeto precisa usar a **mais exigente** das duas.",
             r"\delta = \min(\delta_1, \delta_2) = \min([[d1:g]],\, [[d2:g]]) = [[delta:g]]",
             (Value(r"\delta", "[[delta:g]]", "escolhido", "tolerância comum de projeto"),)),
        Step("Converter em atenuação",
             "A fórmula de $\\beta$ é escrita em decibéis, então convertemos $\\delta$.",
             r"A_s = -20\log_{10}([[delta:g]]) = [[As:.4f]]\ \text{dB}",
             (Value(r"A_s", "[[As:.4f]] dB", "calculado", "atenuação de projeto"),)),
        Step("Calcular o parâmetro de forma $\\beta$",
             "Como $21 < A_s = [[As:.2f]] < 50$, vale o **segundo ramo** da fórmula de Kaiser. Substituindo $A_s - 21 = [[As_m21:.4f]]$:",
             r"\beta = 0.5842\,([[As_m21:.4f]])^{0.4} + 0.07886\,([[As_m21:.4f]]) = [[beta:.4f]]",
             (Value(r"\beta", "[[beta:.4f]]", "calculado", "forma da janela de Kaiser"),)),
        Step("Medir a largura de transição",
             "Há duas transições: da passagem inferior para a rejeição e da rejeição para a passagem superior. Ambas têm a mesma largura:",
             r"\Delta\omega = [[b2_lo:g]]\pi - [[b1_hi:g]]\pi = [[b3_lo:g]]\pi - [[b2_hi:g]]\pi = [[dw:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "calculado", "largura de transição"),)),
        Step("Estimar o comprimento pela fórmula de Kaiser",
             "Substituindo $A_s$ e $\\Delta\\omega$ na fórmula empírica. O resultado já é ímpar, como exige o rejeita-faixa (Tipo 1).",
             r"M \approx \left\lceil \frac{[[As:.4f]] - 7.95}{2.285 \cdot [[dw:g]]\pi} \right\rceil + 1 = \lceil [[x:.2f]] \rceil + 1 = [[x_ceil]] + 1 = [[M_est]]",
             (Value(r"M_{\text{est}}", "[[M_est]]", "calculado", "estimativa de Kaiser (não verificada)"),)),
        Step("Posicionar as frequências de corte",
             "O filtro ideal tem degraus abruptos. Cada corte $\\omega_c$ é posto no **meio** da sua transição, para que a ondulação se distribua simetricamente em torno da borda.",
             r"\omega_{c1} = \frac{[[b1_hi:g]]\pi + [[b2_lo:g]]\pi}{2} = [[wc1:g]]\pi, \qquad \omega_{c2} = \frac{[[b2_hi:g]]\pi + [[b3_lo:g]]\pi}{2} = [[wc2:g]]\pi",
             (Value(r"\omega_{c1}", r"$[[wc1:g]]\pi$", "calculado", "corte inferior"),
              Value(r"\omega_{c2}", r"$[[wc2:g]]\pi$", "calculado", "corte superior"))),
        Step("Construir o filtro com a estimativa e **medir**",
             "Com $M = [[M_est]]$, $\\beta = [[beta:.4f]]$ e os cortes acima, calcula-se $h(n) = h_d(n)\\,w(n)$ e mede-se $|H(e^{j\\omega})|$ numa grade de [[n_grid]] frequências que inclui as bordas exatas.",
             r"\max_{\text{rejeição}} |H| = [[est_dev_stop:.5f]] \;>\; \delta_2 = [[d2:g]]",
             (Value(r"\max|H|_{\text{rej}}", "[[est_dev_stop:.5f]]", "resultado", "medido com M = [[M_est]]"),
              Value(r"\max|H|_{\text{rej}} \le \delta_2", "✗ não atende", "criterio", "excede o limite em [[est_excess_pct:.0f]]%")),
             note="**A estimativa de Kaiser não atende à especificação.** Este é o valor que o DTA apresenta como resposta (errata E1)."),
        Step("Buscar o comprimento mínimo",
             "Mantendo $\\beta$ fixo (ele depende só de $A_s$), testam-se os comprimentos **ímpares** em ordem crescente, a partir de $M = [[search_lo]]$. O primeiro que atende às três tolerâncias é, por construção, o mínimo. Últimos comprimentos testados:\n\n[[search_table]]",
             r"M_{\min} = [[M]], \qquad \alpha = \frac{M-1}{2} = [[alpha:g]]",
             (Value(r"M_{\min}", "[[M]]", "resultado", "menor comprimento que atende (Tipo [[type]])"),
              Value(r"\alpha", "[[alpha:g]]", "resultado", "atraso em amostras"))),
    ),

    # 7 ──────────────────────────────────────────────────────────────────────
    implementation=r"""
A resolução é executada pelo pacote `dsp_av2`, que implementa as equações acima sem usar `fir1` nem `firwin`. O fluxo é:

1. **Especificação** (`specs.P7_13`): as faixas e tolerâncias do enunciado, transcritas sem cálculo.
2. **Projeto com a estimativa** (`design_kaiser`): calcula $\delta$, $A_s$, $\beta$, $\Delta\omega$, $M_{\text{est}}$, os cortes, $h_d(n)$, $w(n)$ e $h(n)$, exatamente nas etapas 2 a 7.
3. **Medição** (`validate_fir_tolerance`): avalia $|H(e^{j\omega})|$ em [[n_grid]] frequências, com as bordas incluídas, e compara o desvio máximo de cada faixa com sua tolerância.
4. **Busca** (`search_min_length`): repete 2 e 3 para comprimentos crescentes e admissíveis (só ímpares) e para no primeiro que atende.
5. **Gráficos**: desenhados a partir dos mesmos vetores calculados.

As janelas e respostas ideais são conferidas por testes automatizados contra implementações independentes (`numpy.kaiser`, `scipy.signal.firwin`), que **não** são usadas na solução.
""",
    code_roles=(
        CodeRole("dsp_av2.fir.design.design_kaiser", "Etapas 2–7: δ, A_s, β, Δω, estimativa de M, cortes e h(n)"),
        CodeRole("dsp_av2.fir.ideal.ideal_bandstop", "Resposta ideal h_d(n) do rejeita-faixa"),
        CodeRole("dsp_av2.fir.windows.kaiser", "Coeficientes da janela de Kaiser"),
        CodeRole("dsp_av2.validation.validate_fir_tolerance", "Etapa 8: mede |H| e compara com G ± δ"),
        CodeRole("dsp_av2.fir.design.search_min_length", "Etapa 9: busca do menor M que atende"),
        CodeRole("exercises.p7_13.search", "Liga tudo: estimativa → busca → projeto final"),
    ),

    # 8 ──────────────────────────────────────────────────────────────────────
    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
Os [[M]] coeficientes são **simétricos** em torno de $n = \alpha = [[alpha:g]]$, o que garante a fase linear. O coeficiente central é o maior, porque contém o impulso $\delta(n-\alpha)$ do passa-tudo. As amostras decaem para as bordas pela ação da janela de Kaiser.
"""),
        Figure("amplitude", "Resposta de amplitude $H_r(\\omega)$", r"""
As faixas sombreadas mostram as regiões permitidas $G \pm \delta$. Nas passagens, $H_r$ fica perto de 1 com folga: o desvio é [[dev_pass:.4f]], contra os [[d1:g]] permitidos, ou seja, cerca de [[pass_slack_ratio:.0f]]× menor que o permitido (a folga desperdiçada discutida na Fundamentação). Na rejeição, $H_r$ oscila em torno de zero dentro de $\pm[[d2:g]]$.
"""),
        Figure("db", "Magnitude em dB", r"""
A linha tracejada marca $-A_s = -[[As:.2f]]$ dB, o equivalente a $\delta_2$. Os lóbulos da rejeição ficam todos abaixo dela. Os mais altos estão junto às bordas $[[b2_lo:g]]\pi$ e $[[b2_hi:g]]\pi$, onde a margem é menor.
"""),
        Figure("compare", "Por que $M = [[M_est]]$ falha e $M = [[M]]$ atende", r"""
Zoom na faixa de rejeição, em escala linear, comparando o projeto com a estimativa de Kaiser e o projeto final. Com $M = [[M_est]]$, os lóbulos junto às bordas ultrapassam o limite $|H| = [[d2:g]]$. Com $M = [[M]]$, a transição fica mais estreita e todos os lóbulos ficam abaixo do limite.
"""),
    ),

    # 9 ──────────────────────────────────────────────────────────────────────
    verification=r"""
O critério é o do enunciado: em cada faixa, o **desvio máximo** de $|H(e^{j\omega})|$ em relação ao ganho ideal não pode exceder a tolerância. A medição usa [[n_grid]] frequências uniformes em $[0, \pi]$, **acrescidas das bordas exatas** das faixas, porque o pior ponto costuma estar justamente na borda.
""",

    # 10 ─────────────────────────────────────────────────────────────────────
    conclusion=r"""
- Pelas fórmulas de Kaiser: $A_s = [[As:.2f]]$ dB e $\beta = [[beta:.4f]]$. A estimativa é $M = [[M_est]]$.
- **A estimativa não atende**: com $M = [[M_est]]$, $\max|H| = [[est_dev_stop:.5f]] > [[d2:g]]$ na rejeição.
- O **comprimento mínimo verificado é $M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$), com $\max|H| = [[dev_stop:.5f]]$ na rejeição (folga de [[margin_stop:.4f]]) e desvio de [[dev_pass:.4f]] na passagem.
- Lição de método: fórmulas de projeto são **estimativas**. Quando o enunciado pede o mínimo, só a medição da resposta em frequência garante a resposta.
""",
)
