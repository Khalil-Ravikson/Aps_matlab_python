r"""Conteúdo didático — P7.9 (passa-faixa com janela de Hann).

Fonte técnica: DTA §2.1 item 1, §3.1 e §4; Ingle & Proakis, Tabela 7.1 e
Seção 7.4 (conforme DTA). Item V5 de docs/errata_dta.md.
"""

from __future__ import annotations

from functools import lru_cache

from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_9 as SPEC
from exercises import p7_09

from .helpers import fir_db_context
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable


@lru_cache(maxsize=1)
def objects() -> dict:
    d = p7_09.design()
    return {"spec": SPEC, "final": d, "final_report": p7_09.validate(d)}


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    d = o["final"]
    ctx = fir_db_context(SPEC, d, o["final_report"])
    k = FIXED_WINDOWS["hann"].transition_factor
    ctx |= {"k": k, "As_win": FIXED_WINDOWS["hann"].min_stopband_db, "ratio": k / d.transition_width}
    return ctx


CONTENT = ExerciseContent(
    id="p7_09", number="P7.9", chapter=7,
    title="Passa-faixa FIR com janela de Hann",
    tagline="FIR · passa-faixa · janela de Hann · Tabela 7.1",
    highlights=(
        Value("M", "[[M]]", "resultado", "comprimento (Tipo [[type]], α = [[alpha:g]])"),
        Value(r"R_p", "[[rp:.3f]] dB", "resultado", "medido (exigido ≤ [[Rp_spec:g]] dB)"),
        Value(r"A_s", "[[as_:.2f]] dB", "resultado", "medido (exigido ≥ [[As_spec:g]] dB)"),
        Value(r"\omega_{c1},\ \omega_{c2}", r"[[wc1:g]]π, [[wc2:g]]π", "calculado", "frequências de corte"),
    ),
    errata=("V5",),
    result_keys=("M", "rp", "as_"),

    asked=r"""
Projetar um **filtro FIR passa-faixa de fase linear** que deixe passar as frequências entre $[[b2_lo:g]]\pi$ e $[[b2_hi:g]]\pi$ e atenue as de fora, usando obrigatoriamente a **janela de Hann**.

- Na passagem, a ondulação não pode passar de $R_p = [[Rp_spec:g]]$ dB.
- Nas duas rejeições, a atenuação deve ser de pelo menos $A_s = [[As_spec:g]]$ dB.

O enunciado pede a resposta ao impulso $h(n)$ e a magnitude em dB. Não fixa $M$: o comprimento sai do método, a partir da largura de transição.
""",
    data_note=r"""
As duas faixas de transição têm larguras **diferentes** ($[[t1:g]]\pi$ à esquerda e $[[t2:g]]\pi$ à direita). Isso influencia a escolha de $\Delta\omega$ no passo 1.
""",
    data_rows=(
        Value(r"\text{Rejeição inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "a ser atenuada"),
        Value(r"\text{Passagem}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", "a ser preservada"),
        Value(r"\text{Rejeição superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", "a ser atenuada"),
        Value(r"R_p", "[[Rp_spec:g]] dB", "dado", "ondulação máxima na passagem"),
        Value(r"A_s", "[[As_spec:g]] dB", "dado", "atenuação mínima nas rejeições"),
        Value(r"\text{Método}", "janela de Hann, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Método da janela.** O passa-faixa ideal tem resposta ao impulso infinita. Multiplicá-la por uma janela $w(n)$ de comprimento $M$ produz um FIR. No domínio da frequência, a resposta ideal é convoluída com o espectro da janela, e isso tem dois efeitos:

- a **largura do lóbulo principal** da janela define a **faixa de transição** (inversamente proporcional a $M$);
- a **altura dos lóbulos laterais** define a **atenuação** alcançável, que não depende de $M$.

Por isso, no método das janelas fixas, **a janela decide $A_s$ e o comprimento decide $\Delta\omega$**.

**Tabela 7.1 do livro** (conforme o DTA). Para cada janela fixa, ela dá a largura de transição aproximada e a atenuação mínima. Para a Hann: $\Delta\omega \approx 6.2\pi/M$ e $A_s \approx 44$ dB. A janela é imposta pelo enunciado, mas convém conferir se ela é capaz de atender: $44 \ge [[As_spec:g]]$ dB.

**Fase linear.** Com coeficientes simétricos, $H(e^{j\omega}) = H_r(\omega)\,e^{-j\omega\alpha}$, com $\alpha = (M-1)/2$. O atraso é o mesmo em todas as frequências.
""",
    deepening=(
        Deepening("Por que usar a menor das duas transições?", r"""
O método da janela produz transições de **mesma largura** em todas as bordas, porque elas vêm do mesmo lóbulo principal. Se o projeto usasse a transição mais larga ($[[t2:g]]\pi$), a borda mais estreita ($[[t1:g]]\pi$) não seria atendida. Usa-se então a mais exigente: $\Delta\omega = \min([[t1:g]]\pi, [[t2:g]]\pi)$. Do lado direito sobra folga.
"""),
        Deepening("Como R_p e A_s são medidos?", r"""
Seguindo a convenção do livro (função `freqz_m`), a magnitude é expressa em dB **relativa ao seu máximo**:

$$R_p = -\min_{\omega \in \text{passagem}} |H|_{dB}, \qquad A_s = -\max_{\omega \in \text{rejeição}} |H|_{dB}.$$

A grade de frequências **inclui as bordas exatas**. Uma grade que pula as bordas pode superestimar $A_s$ (ver P7.10).
"""),
    ),

    equations=(
        Equation(r"\Delta\omega = \min(\omega_{1p} - \omega_{1s},\ \omega_{2s} - \omega_{2p})", "Largura de transição",
                 "A transição mais estreita da especificação.",
                 (Variable(r"\omega_{1s}, \omega_{1p}", "bordas de rejeição e passagem inferiores"),
                  Variable(r"\omega_{2p}, \omega_{2s}", "bordas de passagem e rejeição superiores")), "DTA §3.1"),
        Equation(r"M = \left\lceil \frac{6.2\pi}{\Delta\omega} \right\rceil + 1", "Comprimento pela Tabela 7.1 (Hann)",
                 "Comprimento necessário para a transição da janela de Hann caber em $\\Delta\\omega$.",
                 (Variable(r"M", "número de coeficientes"),), "DTA §3.1.C"),
        Equation(r"w(n) = 0.5 - 0.5\cos\left(\frac{2\pi n}{M-1}\right), \quad 0 \le n \le M-1", "Janela de Hann",
                 "Janela com zeros nas extremidades (decisão D1 do plano).", (), "DTA §3.1.C"),
        Equation(r"\omega_c = \frac{\omega_p + \omega_s}{2}", "Frequência de corte ideal",
                 "No meio de cada transição.", (), "DTA §4"),
        Equation(r"h_d(n) = \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}", "Resposta ideal do passa-faixa",
                 "Diferença entre dois passa-baixas ideais.",
                 (Variable(r"\alpha = (M-1)/2", "atraso (centro de simetria)"),), "DTA §3.1.E"),
        Equation(r"h(n) = h_d(n)\,w(n)", "Filtro projetado", "Resposta ideal truncada e suavizada pela janela.", (), "DTA §3.1"),
    ),

    steps=(
        Step("Determinar a largura de transição",
             "A especificação tem duas transições de larguras diferentes. Como a janela gera a mesma transição em todas as bordas, vale a menor (ver aprofundamento).",
             r"\Delta\omega_1 = [[b2_lo:g]]\pi - [[b1_hi:g]]\pi = [[t1:g]]\pi,\quad \Delta\omega_2 = [[b3_lo:g]]\pi - [[b2_hi:g]]\pi = [[t2:g]]\pi \;\Rightarrow\; \Delta\omega = [[dw:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "escolhido", "a transição mais estreita"),)),
        Step("Conferir a janela imposta",
             "O enunciado impõe a Hann. Pela Tabela 7.1, ela atinge cerca de [[As_win:g]] dB, mais do que os [[As_spec:g]] dB exigidos.",
             r"A_{s,\text{Hann}} \approx [[As_win:g]]\ \text{dB} \;\ge\; [[As_spec:g]]\ \text{dB}",
             (Value(r"\text{janela}", "Hann", "dado", "imposta pelo enunciado"),
              Value(r"A_{s,\text{Hann}}", "[[As_win:g]] dB", "criterio", "valor nominal da Tabela 7.1"))),
        Step("Calcular o comprimento",
             "Pela Tabela 7.1, a transição da Hann vale $6.2\\pi/M$. Isolando $M$ e arredondando para cima:",
             r"M = \left\lceil \frac{[[k:g]]\pi}{[[dw:g]]\pi} \right\rceil + 1 = \lceil [[ratio:.2f]] \rceil + 1 = [[M]]",
             (Value("M", "[[M]]", "calculado", "comprimento (ímpar → Tipo [[type]])"),
              Value(r"\alpha", "[[alpha:g]]", "calculado", "atraso: $(M-1)/2$"))),
        Step("Posicionar as frequências de corte",
             "Cada corte vai no meio da respectiva transição.",
             r"\omega_{c1} = \frac{[[b1_hi:g]]\pi + [[b2_lo:g]]\pi}{2} = [[wc1:g]]\pi, \qquad \omega_{c2} = \frac{[[b2_hi:g]]\pi + [[b3_lo:g]]\pi}{2} = [[wc2:g]]\pi",
             (Value(r"\omega_{c1}", r"$[[wc1:g]]\pi$", "calculado", "corte inferior"),
              Value(r"\omega_{c2}", r"$[[wc2:g]]\pi$", "calculado", "corte superior"))),
        Step("Construir o filtro e medir",
             "Com $M = [[M]]$, calcula-se $h(n) = h_d(n)\\,w(n)$ e mede-se a magnitude em [[n_grid]] frequências com as bordas incluídas.",
             r"R_p = [[rp:.4f]]\ \text{dB} \le [[Rp_spec:g]], \qquad A_s = [[as_:.2f]]\ \text{dB} \ge [[As_spec:g]]",
             (Value(r"R_p", "[[rp:.4f]] dB", "resultado", "folga de [[rp_margin:.3f]] dB"),
              Value(r"A_s", "[[as_:.2f]] dB", "resultado", "folga de [[as_margin:.2f]] dB"))),
    ),

    implementation=r"""
1. **Especificação** (`specs.P7_9`): faixas, $R_p$, $A_s$ e método.
2. **Projeto** (`design_fixed_window(SPEC, "hann")`): transições, $\Delta\omega$, cortes, $M$ pela Tabela 7.1 (com proteção contra ruído de ponto flutuante no teto), $h_d(n)$, $w(n)$ e $h(n)$.
3. **Medição** (`validate_fir_db`): $R_p$ e $A_s$ na grade do livro ([[n_book]] pontos) e numa grade densa ([[n_grid]] pontos), ambas com as bordas exatas.

Nos testes, `scipy.signal.firwin` (o equivalente ao `fir1`, proibido na solução) reproduz $h(n)$ até $10^{-16}$.
""",
    code_roles=(
        CodeRole("dsp_av2.fir.design.design_fixed_window", "Passos 1–4: Δω, M, cortes e h(n)"),
        CodeRole("dsp_av2.fir.ideal.ideal_bandpass", "Resposta ideal do passa-faixa"),
        CodeRole("dsp_av2.fir.windows.hann", "Coeficientes da janela de Hann"),
        CodeRole("dsp_av2.validation.validate_fir_db", "Passo 5: mede Rp e As e compara com o enunciado"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
Os [[M]] coeficientes são simétricos em torno de $n = [[alpha:g]]$ (fase linear, Tipo [[type]]). As extremidades valem zero, porque a janela de Hann se anula nas bordas.
"""),
        Figure("db", "Magnitude em dB", r"""
A passagem fica em 0 dB entre $[[b2_lo:g]]\pi$ e $[[b2_hi:g]]\pi$. Nas rejeições, o lóbulo lateral mais alto fica a $-[[as_:.2f]]$ dB, abaixo da linha de $-[[As_spec:g]]$ dB e perto do valor nominal da Hann (≈ [[As_win:g]] dB). A transição da direita é visivelmente mais folgada que a da esquerda (passo 1).
"""),
        Figure("passband", "Zoom na passagem (verificação de $R_p$)", r"""
A ondulação máxima é de [[rp:.3f]] dB, bem abaixo do limite de [[Rp_spec:g]] dB. No método da janela, as ondulações da passagem e da rejeição têm amplitude semelhante: $\delta_1 \approx [[delta1_meas:.4f]]$ e $\delta_2 \approx [[delta2_meas:.4f]]$.
"""),
    ),

    verification=r"""
Critério do livro: com a magnitude em dB normalizada pelo máximo, $R_p$ é a maior queda na passagem e $A_s$ a menor atenuação nas rejeições. As duas grades incluem as frequências de borda exatas.
""",
    conclusion=r"""
- Hann com **$M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$), cortes em $[[wc1:g]]\pi$ e $[[wc2:g]]\pi$.
- **Atende** à especificação: $R_p = [[rp:.3f]]$ dB ≤ [[Rp_spec:g]] dB e $A_s = [[as_:.2f]]$ dB ≥ [[As_spec:g]] dB.
- O $A_s$ medido coincide com o nominal da Hann: a janela fixa a atenuação e $M$ só controla a transição.
- Os valores do DTA (44.00 dB e 0.1019 dB) diferem na terceira casa decimal por causa da grade de avaliação (V5).
""",
)
