r"""Conteúdo didático — P7.14 (passa-faixa de comprimento mínimo, janela de Kaiser).

Fonte técnica: DTA §2.1 item 6, §3.1.D e §4 (conforme DTA). Correção E2 de
docs/errata_dta.md (M par é admissível).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_kaiser
from dsp_av2.specs import P7_14 as SPEC
from dsp_av2.validation import validate_fir_tolerance
from exercises import p7_14

from .helpers import md_table, ok
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable


@lru_cache(maxsize=1)
def objects() -> dict:
    est = p7_14.estimate()
    search = p7_14.search()
    final = search.best
    around = range(final.M - 3, final.M + 3)
    return {
        "spec": SPEC, "estimate": est, "search": search, "final": final, "final_report": p7_14.validate(final),
        "around": {M: (design_kaiser(SPEC, M=M), validate_fir_tolerance(design_kaiser(SPEC, M=M))) for M in around},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    est, final, s, fr = o["estimate"], o["final"], o["search"], o["final_report"]
    b1, b2, b3 = SPEC.bands
    As = est.params["As_design_dB"]
    dw = est.transition_width
    x = (As - 7.95) / (2.285 * dw * np.pi)
    rows = []
    for M, (d, r) in o["around"].items():
        c = [ch.measured for ch in r.checks]
        rows.append((M, r.linear_phase_type, f"{c[0]:.5f}", f"{c[1]:.5f}", f"{c[2]:.5f}", ok(r.passed)))
    dta = o["around"][final.M + 1][1]
    return {
        "b1_hi": b1.hi, "b2_lo": b2.lo, "b2_hi": b2.hi, "b3_lo": b3.lo,
        "d1": b2.delta, "d2": b1.delta, "g_lo": 1 - b2.delta, "g_hi": 1 + b2.delta,
        "delta": est.params["delta_min"], "As": As, "As_m21": As - 21, "beta": est.params["beta"],
        "dw": dw, "x": x, "x_ceil": int(np.ceil(round(x, 9))), "M_est": est.M,
        "wc1": est.cutoffs[0], "wc2": est.cutoffs[1],
        "search_lo": s.tried[0][0], "n_tried": len(s.tried),
        "table": md_table(["$M$", "Tipo", "rej. inferior", "passagem", "rej. superior", "atende?"], rows),
        "M": final.M, "alpha": final.alpha, "type": fr.linear_phase_type,
        "dev_lo": fr.checks[0].measured, "dev_pass": fr.checks[1].measured, "dev_hi": fr.checks[2].measured,
        "margin_hi": fr.checks[2].margin, "margin_lo": fr.checks[0].margin,
        "Hr_pi": float(analysis.amplitude_response(final.h, np.array([np.pi]))[0]),
        "M_dta": final.M + 1, "dta_passes": dta.passed, "dta_dev_hi": dta.checks[2].measured,
        "n_grid": analysis.DENSE_GRID_POINTS,
    }


CONTENT = ExerciseContent(
    id="p7_14", number="P7.14", chapter=7,
    title="Passa-faixa de comprimento mínimo com janela de Kaiser",
    tagline="FIR · passa-faixa · janela de Kaiser · comprimento mínimo · Tipo 2",
    highlights=(
        Value(r"M_{\min}", "[[M]]", "resultado", "comprimento mínimo (Tipo [[type]])"),
        Value(r"\beta", "[[beta:.4f]]", "calculado", "parâmetro da janela de Kaiser"),
        Value(r"A_s", "[[As:.2f]] dB", "calculado", "atenuação de projeto"),
        Value(r"\alpha", "[[alpha:g]]", "resultado", "atraso (semi-inteiro: M par)"),
    ),
    errata=("E2",),
    result_keys=("M", "beta", "dev_hi", "dev_lo"),

    asked=r"""
Projetar um **filtro FIR passa-faixa de fase linear** com a **janela de Kaiser** e encontrar o **menor comprimento** $M$ que atende às tolerâncias:

- passagem $[[b2_lo:g]]\pi$–$[[b2_hi:g]]\pi$: $[[g_lo:g]] \le |H| \le [[g_hi:g]]$;
- rejeições $\omega \le [[b1_hi:g]]\pi$ e $\omega \ge [[b3_lo:g]]\pi$: $|H| \le [[d2:g]]$.

Também são pedidos os gráficos de $H_r(\omega)$ e da magnitude em dB. A diferença em relação ao P7.13 é que, num passa-faixa, **$M$ par é permitido**, e isso muda a resposta.
""",
    data_note=r"""
Reescrita como ganho ideal ± tolerância: passagem com $G = 1$, $\delta_1 = [[d1:g]]$; rejeições com $G = 0$, $\delta_2 = [[d2:g]]$.
""",
    data_rows=(
        Value(r"\text{Rejeição inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", r"$|H| \le [[d2:g]]$"),
        Value(r"\text{Passagem}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", r"$[[g_lo:g]] \le |H| \le [[g_hi:g]]$"),
        Value(r"\text{Rejeição superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", r"$|H| \le [[d2:g]]$"),
        Value(r"\delta_1", "[[d1:g]]", "dado", "tolerância na passagem"),
        Value(r"\delta_2", "[[d2:g]]", "dado", "tolerância nas rejeições"),
        Value(r"\text{Método}", "Kaiser, comprimento mínimo, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Kaiser** (Seção 7.5 do livro, conforme o DTA): o parâmetro $\beta$ escolhe a altura dos lóbulos laterais e uma fórmula empírica estima $M$. Como no P7.13, projeta-se pela menor tolerância, porque a janela iguala as ondulações das faixas.

**Tipo 1 × Tipo 2.** Com $M$ par (Tipo 2), $\alpha = (M-1)/2$ é semi-inteiro e $H_r(\pi) = 0$ obrigatoriamente. Num **passa-faixa**, $\omega = \pi$ pertence à rejeição superior, onde o ganho desejado já é zero. O zero forçado não atrapalha, e **$M$ par é admissível** (decisão D4 do plano). Por isso a busca do mínimo deve testar **todos** os comprimentos, pares e ímpares.

**Assimetria do Tipo 2.** O zero obrigatório em $\pi$ "puxa" a resposta da rejeição superior. Os desvios das duas rejeições deixam de ser iguais, o que não acontece no Tipo 1.
""",
    deepening=(
        Deepening("Por que o DTA chegou a M ímpar?", r"""
O DTA forçou $M$ ímpar ("$[[M]] \to [[M_dta]]$, Tipo 1"). Essa restrição só se justifica quando o filtro precisa de ganho em $\omega = \pi$ (passa-altas, rejeita-faixa, escada com último degrau não nulo). Aqui não é o caso, então o mínimo deve ser procurado também entre os pares (item E2 da errata).
"""),
    ),

    equations=(
        Equation(r"A_s = -20\log_{10}\min(\delta_1,\delta_2)", "Atenuação de projeto", "", (), "DTA §3.1.D"),
        Equation(r"\beta = 0.5842(A_s-21)^{0.4} + 0.07886(A_s-21), \quad 21 < A_s < 50", "Parâmetro de Kaiser (ramo usado)", "",
                 (Variable(r"\beta", "forma da janela"),), "DTA §3.1.D"),
        Equation(r"M \approx \left\lceil \frac{A_s - 7.95}{2.285\,\Delta\omega} \right\rceil + 1", "Estimativa do comprimento", "Ponto de partida, não garantia.", (), "DTA §3.1.D"),
        Equation(r"h_d(n) = \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}", "Resposta ideal do passa-faixa",
                 "Válida também para $\\alpha$ semi-inteiro ($M$ par).", (), "DTA §3.1.E"),
        Equation(r"H_r(\pi) = 0 \quad (\text{Tipo 2})", "Zero obrigatório do Tipo 2", "Consequência da simetria com $M$ par.", (), "DTA §3.1.B"),
    ),

    steps=(
        Step("Tolerância e atenuação de projeto", "Projeta-se pela menor tolerância.",
             r"\delta = \min([[d1:g]],\ [[d2:g]]) = [[delta:g]] \;\Rightarrow\; A_s = -20\log_{10}([[delta:g]]) = [[As:.4f]]\ \text{dB}",
             (Value(r"\delta", "[[delta:g]]", "escolhido", "a tolerância mais exigente"),
              Value(r"A_s", "[[As:.4f]] dB", "calculado", "atenuação de projeto"))),
        Step("Parâmetro de forma β", "Como $21 < A_s < 50$, vale o segundo ramo da fórmula.",
             r"\beta = 0.5842\,([[As_m21:.4f]])^{0.4} + 0.07886\,([[As_m21:.4f]]) = [[beta:.4f]]",
             (Value(r"\beta", "[[beta:.4f]]", "calculado", "forma da janela"),)),
        Step("Transição, cortes e estimativa de M",
             "As duas transições têm $[[dw:g]]\\pi$. Os cortes ficam no meio delas. **Não** se força $M$ ímpar.",
             r"\omega_{c1} = [[wc1:g]]\pi,\ \ \omega_{c2} = [[wc2:g]]\pi, \qquad M \approx \lceil [[x:.2f]] \rceil + 1 = [[M_est]]",
             (Value(r"M_{\text{est}}", "[[M_est]]", "calculado", "estimativa de Kaiser"),)),
        Step("Buscar o comprimento mínimo (pares e ímpares)",
             "Com $\\beta$ fixo, testam-se todos os comprimentos a partir de $M = [[search_lo]]$ ([[n_tried]] testados). Os vizinhos do resultado:\n\n[[table]]",
             r"M_{\min} = [[M]]\ (\text{Tipo } [[type]]), \qquad \alpha = [[alpha:g]]",
             (Value(r"M_{\min}", "[[M]]", "resultado", "igual à estimativa"),
              Value(r"\alpha", "[[alpha:g]]", "resultado", "semi-inteiro, entre duas amostras")),
             note="A estimativa acertou e nenhum comprimento menor atende. O $M = [[M_dta]]$ do DTA também atende, mas não é o mínimo (errata E2)."),
        Step("Conferir o zero do Tipo 2 e a faixa crítica",
             "Com $M$ par, $H_r(\\pi)$ tem de ser zero. Os desvios das rejeições ficam assimétricos.",
             r"H_r(\pi) = [[Hr_pi:tex2]] \approx 0, \qquad \text{rej. inferior: } [[dev_lo:.5f]], \quad \text{rej. superior: } [[dev_hi:.5f]] \le [[d2:g]]",
             (Value(r"\text{faixa crítica}", "rejeição superior", "resultado", "folga de [[margin_hi:.5f]]"),)),
    ),

    implementation=r"""
1. **Estimativa** (`design_kaiser`): $\delta$, $A_s$, $\beta$, $\Delta\omega$, $M_{\text{est}}$, cortes e $h(n)$, sem impor paridade, porque o passa-faixa não exige ganho em $\pi$.
2. **Busca** (`search_min_length` com `admissible_lengths`, que inclui pares): mede cada candidato com `validate_fir_tolerance` ([[n_grid]] frequências com as bordas) e para no primeiro que atende.
""",
    code_roles=(
        CodeRole("dsp_av2.fir.design.design_kaiser", "Passos 1–3"),
        CodeRole("dsp_av2.fir.design.admissible_lengths", "Comprimentos admissíveis: pares e ímpares aqui"),
        CodeRole("dsp_av2.fir.design.search_min_length", "Passo 4: menor M que atende"),
        CodeRole("dsp_av2.validation.validate_fir_tolerance", "Medição dos desvios em cada faixa"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
[[M]] coeficientes simétricos em torno de $\alpha = [[alpha:g]]$: o centro de simetria fica **entre** duas amostras, a marca do Tipo 2.
"""),
        Figure("amplitude", "Resposta de amplitude $H_r(\\omega)$", r"""
$H_r$ fica dentro das faixas permitidas em todas as bandas. Em $\omega = \pi$ ela vale exatamente zero (Tipo 2), e por isso a ondulação da rejeição superior é diferente da inferior.
"""),
        Figure("db", "Magnitude em dB", r"""
A linha de $-[[As:.0f]]$ dB corresponde a $\delta_2 = [[d2:g]]$. O lóbulo mais alto da rejeição superior encosta nela (desvio de [[dev_hi:.5f]]). É a faixa crítica.
"""),
    ),

    verification=r"""
Em cada faixa, o desvio máximo de $|H(e^{j\omega})|$ em relação ao ganho ideal deve ser no máximo a tolerância. A grade tem [[n_grid]] frequências e inclui as bordas exatas.
""",
    conclusion=r"""
- Kaiser: $A_s = [[As:.2f]]$ dB e $\beta = [[beta:.4f]]$, com estimativa $M = [[M_est]]$.
- **Comprimento mínimo verificado: $M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$). Igual à estimativa, e nenhum comprimento menor atende.
- Faixa crítica: rejeição superior ([[dev_hi:.5f]] contra [[d2:g]]), por causa do zero obrigatório do Tipo 2 em $\omega = \pi$.
- O DTA dá $M = [[M_dta]]$ por forçar $M$ ímpar, o que não é necessário num passa-faixa (E2).
""",
)
