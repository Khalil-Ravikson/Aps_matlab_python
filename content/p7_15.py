r"""Conteúdo didático — P7.15 (filtro em escada, Kaiser × Parks-McClellan do Exemplo 7.26).

Fonte técnica: DTA §2.1 item 7, §3.1.D e §4; Exemplo 7.26 (pp. 367–368, valor M = 49
informado pelo usuário, decisão D6). Correções E3 e E4 de docs/errata_dta.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis, ideal
from dsp_av2.fir import windows as W
from dsp_av2.specs import EXAMPLE_7_26_M, P7_15 as SPEC
from exercises import p7_15

from .helpers import md_table
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable

AS_TRIALS = (60, 57, 54, 51)


def _min_M_for(As: float) -> int:
    beta = W.kaiser_beta(As)
    w = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
    for M in range(40, 120):
        h = ideal.ideal_multiband([0.35 * np.pi, 0.75 * np.pi], [b.gain for b in SPEC.bands], M) * W.kaiser(M, beta)
        if all(dv <= b.delta for dv, b in zip(analysis.band_deviations(h, SPEC.bands, w), SPEC.bands)):
            return M
    raise RuntimeError


@lru_cache(maxsize=1)
def objects() -> dict:
    est = p7_15.estimate()
    search = p7_15.search()
    final = search.best
    rz = p7_15.remez_search()
    return {
        "spec": SPEC, "estimate": est, "search": search, "final": final, "final_report": p7_15.validate(final),
        "remez": rz, "as_trials": [(As, W.kaiser_beta(As), _min_M_for(As)) for As in AS_TRIALS],
        "overlay": {"curves": {f"Kaiser, M = {final.M}": final.h, f"Parks-McClellan (remez), M = {rz.M_min}": rz.h},
                    "xlim": (0, 1), "ylim": (-90, 5), "limit_db": 20 * np.log10(SPEC.bands[2].delta),
                    "limit_label": f"20·log10(δ3) = {20 * np.log10(SPEC.bands[2].delta):.0f} dB",
                    "edges": list(SPEC.edges)},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    est, final, s, fr, rz = o["estimate"], o["final"], o["search"], o["final_report"], o["remez"]
    b1, b2, b3 = SPEC.bands
    As = est.params["As_design_dB"]
    x = (As - 7.95) / (2.285 * est.transition_width * np.pi)
    rz_rows = [(M, "✓ sim" if okk else "✗ não", f"{a:.2f}") for M, okk, a in rz.tried[-5:]]
    trials = o["as_trials"]
    best_trial = min(trials, key=lambda t: t[2])
    return {
        "b1_hi": b1.hi, "b2_lo": b2.lo, "b2_hi": b2.hi, "b3_lo": b3.lo,
        "g1": b1.gain, "g2": b2.gain, "g3": b3.gain, "d1": b1.delta, "d2": b2.delta, "d3": b3.delta,
        "delta": est.params["delta_min"], "As": As, "beta": est.params["beta"], "As_m87": As - 8.7,
        "dw": est.transition_width, "x": x, "x_ceil": int(np.ceil(round(x, 9))), "M_est": est.M,
        "wc1": est.cutoffs[0], "wc2": est.cutoffs[1],
        "search_tail": ", ".join(f"{M}: {'✓' if okk else '✗'}" for M, okk in s.tried[-5:]),
        "M": final.M, "alpha": final.alpha, "type": fr.linear_phase_type,
        "dev1": fr.checks[0].measured, "dev2": fr.checks[1].measured, "dev3": fr.checks[2].measured,
        "margin3": fr.checks[2].margin,
        "M_dta": est.M + 1, "M_pm": rz.M_min, "M_book": EXAMPLE_7_26_M,
        "w1": d_w(SPEC, 0), "w2": d_w(SPEC, 1), "w3": d_w(SPEC, 2),
        "rz_table": md_table(["$M$", "atende?", "$A_s$ da banda 3 (dB)"], rz_rows),
        "ratio": final.M / rz.M_min, "pct": 100 * (final.M / rz.M_min - 1),
        "pct_dta": 100 * ((est.M + 1) / rz.M_min - 1), "pct_dta_red": 100 * (1 - rz.M_min / (est.M + 1)),
        "trials_table": md_table(["$A_s$ de projeto (dB)", "$\\beta$", "$M$ mínimo"],
                                 [(f"{a:g}", f"{b:.4f}", m) for a, b, m in trials]),
        "best_As": best_trial[0], "best_M": best_trial[2], "best_pct": 100 * (best_trial[2] / rz.M_min - 1),
        "n_grid": analysis.DENSE_GRID_POINTS,
    }


def d_w(spec, k):
    dmin = min(b.delta for b in spec.bands)
    return dmin / spec.bands[k].delta


CONTENT = ExerciseContent(
    id="p7_15", number="P7.15", chapter=7,
    title="Filtro em escada: janela de Kaiser × Parks-McClellan",
    tagline="FIR · multibanda (escada) · Kaiser · comparação com o Exemplo 7.26",
    highlights=(
        Value(r"M_{\text{Kaiser}}", "[[M]]", "resultado", "comprimento mínimo verificado"),
        Value(r"M_{\text{PM}}", "[[M_pm]]", "resultado", "Exemplo 7.26 (reproduzido com remez)"),
        Value(r"M_{\text{K}}/M_{\text{PM}}", "[[ratio:.2f]]", "resultado", "Kaiser ≈ [[pct:.0f]]% mais longo"),
        Value(r"\beta", "[[beta:.4f]]", "calculado", "A_s = [[As:.0f]] dB"),
    ),
    errata=("E3", "E4"),
    result_keys=("M", "beta", "dev3", "best_M"),

    asked=r"""
Projetar com a **janela de Kaiser** o filtro em **escada** do Exemplo 7.26, que tem três faixas com ganhos decrescentes:

| Banda | Faixa | Ganho ideal | Tolerância |
|---|---|---|---|
| 1 | $0 \le \omega \le [[b1_hi:g]]\pi$ | [[g1:g]] | $\delta_1 = [[d1:g]]$ |
| 2 | $[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$ | [[g2:g]] | $\delta_2 = [[d2:g]]$ |
| 3 | $[[b3_lo:g]]\pi \le \omega \le \pi$ | [[g3:g]] | $\delta_3 = [[d3:g]]$ |

Depois, **comparar o comprimento** com o do Exemplo 7.26, que usa o algoritmo de Parks-McClellan (equiripple), e plotar a magnitude em dB.
""",
    data_note=r"""
As tolerâncias são muito diferentes entre as bandas: a banda 1 tolera 10 vezes mais erro que a banda 3. Essa diferença é a chave da comparação final.
""",
    data_rows=(
        Value(r"\text{Banda 1}", r"$0 \le \omega \le [[b1_hi:g]]\pi$, $G = [[g1:g]]$", "dado", r"$\delta_1 = [[d1:g]]$"),
        Value(r"\text{Banda 2}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$, $G = [[g2:g]]$", "dado", r"$\delta_2 = [[d2:g]]$"),
        Value(r"\text{Banda 3}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$, $G = [[g3:g]]$", "dado", r"$\delta_3 = [[d3:g]]$"),
        Value(r"M_{\text{Ex. 7.26}}", "[[M_book]]", "dado", "Parks-McClellan, pp. 367–368 (informado; ver D6)"),
        Value(r"\text{Método}", "janela de Kaiser, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Resposta ideal em degraus.** Um filtro de vários ganhos é uma soma de passa-baixas ideais ponderados pela **altura de cada degrau**:

$$h_d(n) = G_1 h_{LP,1}(n) + G_2[h_{LP,2}(n) - h_{LP,1}(n)] + \dots$$

Com $G = (1,\ 0.5,\ 0)$, isso dá $h_d = 0.5\,h_{LP}(\omega_{c1}) + 0.5\,h_{LP}(\omega_{c2})$.

**Kaiser.** Como sempre no método da janela, as ondulações saem praticamente iguais em todas as bandas, e projeta-se pela menor tolerância ($\delta_3$).

**Parks-McClellan** (Seção 7.6). Minimiza o **erro ponderado** máximo, distribuindo a ondulação de forma equiripple. Com pesos $W_k = \delta_{\min}/\delta_k$, cada banda usa **toda** a sua tolerância, e por isso o filtro sai mais curto.

**Paridade.** O ganho em $\omega = \pi$ é zero (banda 3), então o Tipo 2 é admissível e a busca inclui comprimentos pares.
""",
    deepening=(
        Deepening("Por que a janela perde tanto?", r"""
No projeto final, os desvios medidos são [[dev1:.5f]], [[dev2:.5f]] e [[dev3:.5f]], todos praticamente iguais, embora as tolerâncias sejam [[d1:g]], [[d2:g]] e [[d3:g]]. As bandas 1 e 2 ficam **muito melhores que o necessário**, e essa folga é desperdiçada. O equiripple a aproveita: por isso precisa de menos coeficientes.
"""),
        Deepening("Ajustar o A_s de Kaiser resolveria? (decisão D7)", r"""
Os degraus têm altura 0.5, e a ondulação do método da janela escala com a altura do salto. Isso sugere que $A_s = [[As:.0f]]$ dB é conservador. Testando outros valores, cada um com seu $\beta$ e seu comprimento mínimo:

[[trials_table]]

O melhor caso ($A_s = [[best_As:g]]$ dB) chega a $M = [[best_M]]$, ainda **~[[best_pct:.0f]]% acima** do equiripple. A diferença é estrutural. O projeto oficial segue o critério do livro ($A_s = [[As:.0f]]$ dB).
"""),
    ),

    equations=(
        Equation(r"h_d(n) = \sum_{k=1}^{K-1} (G_k - G_{k+1})\,h_{LP}(n;\omega_{c,k}) + G_K\,\delta(n-\alpha)", "Resposta ideal em degraus",
                 "Forma telescópica da soma de degraus; o termo $\\delta(n-\\alpha)$ só existe se $G_K \\ne 0$.",
                 (Variable(r"G_k", "ganho da banda k"), Variable(r"\omega_{c,k}", "corte entre as bandas k e k+1")), "DTA §3.1.E"),
        Equation(r"\beta = 0.1102(A_s - 8.7), \quad A_s \ge 50", "Parâmetro de Kaiser (ramo usado)", "", (), "DTA §3.1.D"),
        Equation(r"M \approx \left\lceil \frac{A_s - 7.95}{2.285\,\Delta\omega} \right\rceil + 1", "Estimativa do comprimento", "", (), "DTA §3.1.D"),
        Equation(r"\min_{h} \max_{\omega} W(\omega)\,\big|H_r(\omega) - G(\omega)\big|, \qquad W_k = \frac{\delta_{\min}}{\delta_k}", "Critério de Parks-McClellan",
                 "Minimax do erro ponderado. Os pesos fazem o erro ponderado igual em todas as bandas.",
                 (Variable(r"W_k", "peso da banda k"),), "Ingle & Proakis, Seção 7.6"),
    ),

    steps=(
        Step("Tolerância, atenuação e β", "Projeta-se pela menor tolerância ($\\delta_3$). Como $A_s \\ge 50$, vale o primeiro ramo de $\\beta$.",
             r"\delta = [[delta:g]] \Rightarrow A_s = [[As:.0f]]\ \text{dB}, \qquad \beta = 0.1102\,([[As_m87:.1f]]) = [[beta:.4f]]",
             (Value(r"A_s", "[[As:.0f]] dB", "calculado", "a partir de $\\delta_3$"),
              Value(r"\beta", "[[beta:.4f]]", "calculado", "forma da janela"))),
        Step("Cortes e resposta ideal",
             "As duas transições têm $[[dw:g]]\\pi$. Cortes no meio delas. Como cada degrau tem altura 0.5, os dois passa-baixas entram com peso 0.5.",
             r"\omega_{c1} = [[wc1:g]]\pi,\ \ \omega_{c2} = [[wc2:g]]\pi, \qquad h_d = 0.5\,h_{LP}([[wc1:g]]\pi) + 0.5\,h_{LP}([[wc2:g]]\pi)",
             (Value(r"\omega_{c1}, \omega_{c2}", r"$[[wc1:g]]\pi$, $[[wc2:g]]\pi$", "calculado", "cortes"),)),
        Step("Estimar M", "Fórmula de Kaiser, sem forçar paridade (ganho 0 em π).",
             r"M \approx \left\lceil \frac{[[As:.0f]] - 7.95}{2.285 \cdot [[dw:g]]\pi} \right\rceil + 1 = \lceil [[x:.2f]] \rceil + 1 = [[M_est]]",
             (Value(r"M_{\text{est}}", "[[M_est]]", "calculado", "estimativa"),)),
        Step("Buscar o comprimento mínimo",
             "Com $\\beta$ fixo, testam-se todos os comprimentos (pares e ímpares) em ordem crescente. Últimos testados: [[search_tail]].",
             r"M_{\min} = [[M]]\ (\text{Tipo } [[type]]), \qquad \text{desvios: } [[dev1:.5f]],\ [[dev2:.5f]],\ [[dev3:.5f]]",
             (Value(r"M_{\min}", "[[M]]", "resultado", "comprimento mínimo com Kaiser"),
              Value(r"\text{banda crítica}", "3", "resultado", "folga de [[margin3:.5f]]")),
             note="O DTA dá $M = [[M_dta]]$ porque força $M$ ímpar, o que não é necessário aqui (errata E3)."),
        Step("Reproduzir o Exemplo 7.26 (Parks-McClellan)",
             "Para não depender só do valor citado, o equiripple é projetado com `scipy.signal.remez` (equivalente ao `firpm`), com pesos $W = ([[w1:g]],\\ [[w2:g]],\\ [[w3:g]])$:\n\n[[rz_table]]",
             r"M_{\text{PM}} = [[M_pm]] \quad (\text{livro: } [[M_book]])",
             (Value(r"M_{\text{PM}}", "[[M_pm]]", "resultado", "confere com o Exemplo 7.26"),)),
        Step("Comparar os comprimentos", "Razão entre os comprimentos mínimos.",
             r"\frac{M_{\text{Kaiser}}}{M_{\text{PM}}} = \frac{[[M]]}{[[M_pm]]} = [[ratio:.3f]] \quad (\text{Kaiser } [[pct:.0f]]\%\ \text{mais longo})",
             (Value(r"\text{razão}", "[[ratio:.3f]]", "resultado", "Kaiser exige ≈ [[pct:.0f]]% mais coeficientes"),),
             note="O DTA afirma que o equiripple “reduz a ordem em ≈ 35%”. Esse número é $1 - [[M_book]]/[[M_dta]]$, uma redução calculada a partir do Kaiser e com o $M$ errado. Com o mínimo correto, Kaiser é [[pct:.0f]]% mais longo (errata E4)."),
    ),

    implementation=r"""
1. **Kaiser** (`design_kaiser` + `search_min_length`): resposta ideal em degraus (`ideal_multiband`), $\beta$ e busca do menor $M$ medido em [[n_grid]] frequências com as bordas.
2. **Equiripple** (`remez_search`): `scipy.signal.remez` com os pesos $W_k$, testando $M$ crescente até atender às três tolerâncias.

Usar `remez` não viola o enunciado, que proíbe só o `fir1`. Ele entra apenas na comparação, nunca no projeto pedido.
""",
    code_roles=(
        CodeRole("dsp_av2.fir.ideal.ideal_multiband", "Resposta ideal em degraus"),
        CodeRole("dsp_av2.fir.design.design_kaiser", "Passos 1–3"),
        CodeRole("exercises.p7_15.search", "Passo 4: menor M com Kaiser"),
        CodeRole("exercises.p7_15.remez_weights", "Pesos W_k = δ_min/δ_k"),
        CodeRole("exercises.p7_15.remez_search", "Passo 5: menor M equiripple"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
[[M]] coeficientes, simétricos em torno de $\alpha = [[alpha:g]]$ (Tipo [[type]]). O par central é a soma dos dois passa-baixas com peso 0.5.
"""),
        Figure("amplitude", "Resposta de amplitude $H_r(\\omega)$", r"""
Os três degraus (1, 0.5 e 0) aparecem com clareza, cada um dentro da sua faixa permitida. Repare como as faixas das bandas 1 e 2 são largas perto da ondulação real: é a folga desperdiçada.
"""),
        Figure("db", "Magnitude em dB (o gráfico pedido)", r"""
O degrau de ganho 0.5 aparece em $20\log_{10}0.5 \approx -6$ dB. A banda 3 fica abaixo de $-[[As:.0f]]$ dB, com os lóbulos mais altos junto à borda $[[b3_lo:g]]\pi$.
"""),
        Figure("overlay", "Kaiser ($M = [[M]]$) × Parks-McClellan ($M = [[M_pm]]$)", r"""
O equiripple tem ondulação de **mesma altura** ao longo de cada banda, e maior nas bandas 1 e 2, onde a tolerância permite. Com isso consegue atender à banda 3 usando [[M_pm]] coeficientes, contra [[M]] do Kaiser.
"""),
    ),

    verification=r"""
Em cada banda, o desvio máximo de $|H|$ em relação ao ganho ideal deve ser no máximo a tolerância ([[n_grid]] frequências com as bordas). O equiripple da comparação é verificado com o mesmo critério.
""",
    conclusion=r"""
- Kaiser: $A_s = [[As:.0f]]$ dB e $\beta = [[beta:.4f]]$, com estimativa $M = [[M_est]]$. **Comprimento mínimo verificado: $M = [[M]]$** (Tipo [[type]]). O DTA dá [[M_dta]] por forçar $M$ ímpar (E3).
- Exemplo 7.26 (Parks-McClellan): **$M = [[M_pm]]$**, confirmado de forma independente com `remez`.
- **Kaiser exige ≈ [[pct:.0f]]% mais coeficientes** ([[M]]/[[M_pm]]). Os "~35%" do DTA são outra conta, feita com o $M$ errado (E4).
- A causa é estrutural: a janela iguala as ondulações de todas as bandas, e o equiripple pondera cada banda pela sua tolerância. Nem ajustando $A_s$ o Kaiser passa de $M \approx [[best_M]]$.
""",
)
