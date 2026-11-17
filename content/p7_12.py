r"""Conteúdo didático — P7.12 (passa-altas com janela fixa escolhida).

Fonte técnica: DTA §2.1 item 4, §3.1 e §4; Ingle & Proakis, Tabela 7.1 (conforme
DTA). Item V3 de docs/errata_dta.md.
"""

from __future__ import annotations

from functools import lru_cache

from dsp_av2.fir.design import design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_12 as SPEC
from exercises import p7_12

from .helpers import fir_db_context, fir_db_measure, md_table, ok
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable

HAMMING_SCAN = range(35, 96, 10)
BLACKMAN_SCAN = range(51, 60, 2)
NAMES = {"hann": "Hann", "hamming": "Hamming", "blackman": "Blackman"}


@lru_cache(maxsize=1)
def objects() -> dict:
    d = p7_12.design()
    ham = [(M, *fir_db_measure(design_fixed_window(SPEC, "hamming", M=M).h, SPEC)) for M in HAMMING_SCAN]
    bla = [(M, *fir_db_measure(design_fixed_window(SPEC, "blackman", M=M).h, SPEC)) for M in BLACKMAN_SCAN]
    return {
        "spec": SPEC, "final": d, "final_report": p7_12.validate(d), "candidates": p7_12.evaluate_candidates(),
        "hamming_rows": ham, "blackman_rows": bla,
        "scan": {"x": [r[0] for r in ham], "y": [r[1] for r in ham], "limit": SPEC.rp_db, "better": "below",
                 "xlabel": "M (janela de Hamming)", "ylabel": "Rp medido (dB)",
                 "limit_label": f"Rp exigido = {SPEC.rp_db:g} dB"},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    d = o["final"]
    ctx = fir_db_context(SPEC, d, o["final_report"])
    x = 10 ** (SPEC.rp_db / 20)
    delta1 = (x - 1) / (x + 1)
    cand_rows, by = [], {}
    for c in o["candidates"]:
        by[c.window] = c
        if c.design is None:
            cand_rows.append((NAMES[c.window], f"{c.table_as_db:g} dB", "—", "—", "—", "descartada: $A_s$ nominal < exigido"))
        else:
            rp, as_ = c.report.checks[2].measured, c.report.checks[3].measured
            cand_rows.append((NAMES[c.window], f"{c.table_as_db:g} dB", c.design.M, f"{rp:.5f}", f"{as_:.2f}",
                              "**atende**" if c.accepted else "não atende ($R_p$)"))
    ham = o["hamming_rows"]
    bla = o["blackman_rows"]
    bla_first = [r for r in bla if r[1] <= SPEC.rp_db and r[2] >= SPEC.as_db][0]
    k = FIXED_WINDOWS["blackman"].transition_factor
    ctx |= {
        "delta1": delta1, "delta1_db": -20 * __import__("math").log10(delta1),
        "cand_table": md_table(["Janela", "$A_s$ tabela", "$M$", "$R_p$ medido (dB)", "$A_s$ medido (dB)", "resultado"], cand_rows),
        "M_ham": by["hamming"].design.M, "rp_ham": by["hamming"].report.checks[2].measured,
        "rp_ham_min": min(r[1] for r in ham), "rp_ham_max": max(r[1] for r in ham),
        "ham_M_lo": ham[0][0], "ham_M_hi": ham[-1][0],
        "ham_table": md_table(["$M$", "$R_p$ Hamming (dB)"], [(M, f"{rp:.4f}") for M, rp, _ in ham]),
        "bla_table": md_table(["$M$", "$R_p$ (dB)", "$A_s$ (dB)", "atende?"],
                              [(M, f"{rp:.5f}", f"{as_:.2f}", ok(rp <= SPEC.rp_db and as_ >= SPEC.as_db)) for M, rp, as_ in bla]),
        "M_bla_first": bla_first[0], "rp_bla_first": bla_first[1], "rp_bla_first_margin": SPEC.rp_db - bla_first[1],
        "k": k, "ratio": k / d.transition_width, "As_win": FIXED_WINDOWS["blackman"].min_stopband_db,
        "As_hann": FIXED_WINDOWS["hann"].min_stopband_db, "As_ham": FIXED_WINDOWS["hamming"].min_stopband_db,
    }
    return ctx


CONTENT = ExerciseContent(
    id="p7_12", number="P7.12", chapter=7,
    title="Passa-altas FIR com janela fixa escolhida",
    tagline="FIR · passa-altas · escolha justificada da janela · Rp muito exigente",
    highlights=(
        Value(r"\text{janela}", "Blackman", "escolhido", "única que atende Rp"),
        Value("M", "[[M]]", "resultado", "comprimento (Tipo [[type]], α = [[alpha:g]])"),
        Value(r"R_p", "[[rp:.5f]] dB", "resultado", "medido (exigido ≤ [[Rp_spec:g]] dB)"),
        Value(r"A_s", "[[as_:.2f]] dB", "resultado", "medido (exigido ≥ [[As_spec:g]] dB)"),
    ),
    errata=("V3",),
    result_keys=("M", "rp", "as_", "rp_ham"),

    asked=r"""
Projetar um **filtro FIR passa-altas** com transição entre $[[b1_hi:g]]\pi$ e $[[b2_lo:g]]\pi$, usando **uma das janelas fixas, a escolher**.

- Atenuação de pelo menos $A_s = [[As_spec:g]]$ dB na rejeição.
- Ondulação de no máximo $R_p = [[Rp_spec:g]]$ dB na passagem. É uma exigência **muito** rigorosa.

O gráfico pedido é o **zoom da passagem em dB**, para verificar $R_p$. O ponto central do exercício é **justificar a escolha da janela**.
""",
    data_note=r"""
Converter $R_p$ em tolerância linear (passo 1) mostra por que este exercício é diferente dos anteriores: quem decide a janela é a ondulação da passagem, não a atenuação.
""",
    data_rows=(
        Value(r"\text{Rejeição}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "a ser atenuada"),
        Value(r"\text{Passagem}", r"$[[b2_lo:g]]\pi \le \omega \le \pi$", "dado", "a ser preservada"),
        Value(r"A_s", "[[As_spec:g]] dB", "dado", "atenuação mínima"),
        Value(r"R_p", "[[Rp_spec:g]] dB", "dado", "ondulação máxima na passagem"),
        Value(r"\text{Método}", "uma janela fixa, sem `fir1`", "dado", "a janela fica a critério do projetista"),
    ),

    theory=r"""
**Ondulações iguais nas duas faixas.** No método da janela, a ondulação da passagem ($\delta_1$) e a da rejeição ($\delta_2$) vêm dos mesmos lóbulos laterais e saem com amplitudes semelhantes. Uma exigência rigorosa em **qualquer** das faixas obriga a usar uma janela de lóbulos baixos.

**A janela fixa a ondulação; $M$ fixa a transição.** Aumentar $M$ estreita a transição, mas **não reduz** os lóbulos laterais, que são uma propriedade da forma da janela. Se uma janela não atende $R_p$ com um comprimento, não atende com nenhum.

**Passa-altas e paridade.** O passa-altas precisa de ganho 1 em $\omega = \pi$, então só o Tipo 1 serve: $M$ ímpar.

**Candidatas** (Tabela 7.1, conforme o DTA): Hann (≈ [[As_hann:g]] dB, $6.2\pi/M$), Hamming (≈ [[As_ham:g]] dB, $6.6\pi/M$), Blackman (≈ [[As_win:g]] dB, $11\pi/M$).
""",
    deepening=(
        Deepening("Como R_p vira uma tolerância linear?", r"""
Por definição, $R_p = -20\log_{10}\frac{1-\delta_1}{1+\delta_1}$. Isolando $\delta_1$:

$$\delta_1 = \frac{10^{R_p/20} - 1}{10^{R_p/20} + 1}.$$

Com $R_p = [[Rp_spec:g]]$ dB, $\delta_1 \approx [[delta1:tex4]]$. Como a janela produz $\delta_2 \approx \delta_1$, isso equivale a exigir lóbulos laterais em torno de $-20\log_{10}\delta_1 \approx [[delta1_db:.1f]]$ dB, bem mais que os [[As_spec:g]] dB pedidos para a rejeição.
"""),
    ),

    equations=(
        Equation(r"\delta_1 = \frac{10^{R_p/20}-1}{10^{R_p/20}+1}", "Ondulação linear da passagem",
                 "Inversão da definição de $R_p$.", (Variable(r"\delta_1", "desvio máximo do ganho em torno de 1"),), "DTA §3.1.A"),
        Equation(r"M = \left\lceil \frac{11\pi}{\Delta\omega} \right\rceil + 1", "Comprimento pela Tabela 7.1 (Blackman)", "", (), "DTA §3.1.C"),
        Equation(r"w(n) = 0.42 - 0.5\cos\left(\frac{2\pi n}{M-1}\right) + 0.08\cos\left(\frac{4\pi n}{M-1}\right)", "Janela de Blackman", "", (), "DTA §3.1.C"),
        Equation(r"h_d(n) = \delta(n-\alpha) - \frac{\sin[\omega_c(n-\alpha)]}{\pi(n-\alpha)}", "Resposta ideal do passa-altas",
                 "Passa-tudo menos passa-baixas.", (Variable(r"\omega_c", "corte, no meio da transição"),), "DTA §3.1.E"),
        Equation(r"h(n) = h_d(n)\,w(n)", "Filtro projetado", "", (), "DTA §3.1"),
    ),

    steps=(
        Step("Traduzir R_p em tolerância linear",
             "Para comparar com o que cada janela consegue (ver aprofundamento).",
             r"\delta_1 = \frac{10^{[[Rp_spec:g]]/20}-1}{10^{[[Rp_spec:g]]/20}+1} = [[delta1:tex4]] \;\Rightarrow\; -20\log_{10}\delta_1 \approx [[delta1_db:.1f]]\ \text{dB}",
             (Value(r"\delta_1", "$[[delta1:tex4]]$", "calculado", "tolerância linear na passagem"),)),
        Step("Transição, corte e paridade",
             "Uma só transição. O passa-altas exige $M$ ímpar.",
             r"\Delta\omega = [[b2_lo:g]]\pi - [[b1_hi:g]]\pi = [[dw:g]]\pi, \qquad \omega_c = \frac{[[b1_hi:g]]\pi + [[b2_lo:g]]\pi}{2} = [[wc1:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "calculado", "largura de transição"),
              Value(r"\omega_c", r"$[[wc1:g]]\pi$", "calculado", "corte"))),
        Step("Avaliar cada janela fixa",
             "Cada candidata é testada primeiro pelo $A_s$ nominal e depois pela **medição** do filtro projetado com o $M$ da Tabela 7.1:\n\n[[cand_table]]",
             "", (Value(r"\text{Hamming}", "$R_p$ = [[rp_ham:.4f]] dB", "resultado", "atende $A_s$, mas $R_p$ fica ~10× acima do limite"),)),
        Step("Confirmar que aumentar M não salva a Hamming",
             "Se o problema fosse o comprimento, mais coeficientes resolveriam. A medição mostra que não resolvem:\n\n[[ham_table]]",
             r"R_{p,\text{Hamming}} \in [[[rp_ham_min:.3f]],\ [[rp_ham_max:.3f]]]\ \text{dB} \;>\; [[Rp_spec:g]]\ \text{dB} \quad \text{para } M = [[ham_M_lo]]\ldots[[ham_M_hi]]",
             (Value(r"\text{conclusão}", "trocar de janela", "criterio", "a ondulação é da forma da janela"),)),
        Step("Escolher a Blackman e calcular o comprimento",
             "É a única com lóbulos baixos o suficiente (nominal ≈ [[As_win:g]] dB, coerente com os ≈ [[delta1_db:.0f]] dB do passo 1).",
             r"M = \left\lceil \frac{[[k:g]]\pi}{[[dw:g]]\pi} \right\rceil + 1 = [[ratio:.0f]] + 1 = [[M_table]] \;\xrightarrow{M\ \text{ímpar}}\; [[M]], \qquad \alpha = [[alpha:g]]",
             (Value(r"\text{janela}", "Blackman", "escolhido", "justificada pelos passos 1, 3 e 4"),
              Value(r"M", "[[M]]", "calculado", "[[M_table]] pela tabela, +1 pela paridade")),
             note="O DTA escreve $\\lceil 11\\pi/0.2\\pi\\rceil + 1 = [[M]]$. A conta dá [[M_table]]; o [[M]] vem da regra de paridade (item V3 da errata)."),
        Step("Construir o filtro e medir", "Medição em [[n_grid]] frequências, com as bordas.",
             r"R_p = [[rp:.5f]]\ \text{dB} \le [[Rp_spec:g]], \qquad A_s = [[as_:.2f]]\ \text{dB} \ge [[As_spec:g]]",
             (Value(r"R_p", "[[rp:.5f]] dB", "resultado", "folga de [[rp_margin:.5f]] dB"),
              Value(r"A_s", "[[as_:.2f]] dB", "resultado", "folga de [[as_margin:.1f]] dB"))),
        Step("Observar o comprimento mínimo com Blackman (complementar)",
             "O enunciado não pede o mínimo. Mesmo assim:\n\n[[bla_table]]",
             r"M = [[M_bla_first]]:\ R_p = [[rp_bla_first:.5f]]\ \text{dB}\ (\text{folga de } [[rp_bla_first_margin:tex2]]\ \text{dB})",
             (Value(r"M = [[M_bla_first]]", "também atende", "resultado", "com margem desprezível"),)),
    ),

    implementation=r"""
1. **Candidatas** (`evaluate_candidates`): para cada janela fixa, descarta pelo $A_s$ nominal ou projeta com o $M$ da tabela e mede $R_p$ e $A_s$.
2. **Escolha** (`design`): a primeira janela, na ordem de comprimento crescente, que atende às duas exigências medidas.
3. **Medição** (`validate_fir_db`): $R_p$ e $A_s$ nas grades de [[n_book]] e [[n_grid]] pontos com as bordas.
""",
    code_roles=(
        CodeRole("exercises.p7_12.evaluate_candidates", "Passo 3: avalia Hann, Hamming e Blackman"),
        CodeRole("exercises.p7_12.design", "Passo 5: escolhe a primeira janela que atende"),
        CodeRole("dsp_av2.fir.design.design_fixed_window", "Projeto com janela fixa (com ajuste de paridade)"),
        CodeRole("dsp_av2.validation.validate_fir_db", "Passo 6: mede Rp e As"),
    ),

    figures=(
        Figure("passband", "Zoom na passagem: o gráfico pedido", r"""
A ondulação na passagem é de milésimos de dB: o máximo desvio é [[rp:.5f]] dB, abaixo da linha de $-[[Rp_spec:g]]$ dB. As oscilações diminuem à medida que $\omega$ se afasta da borda $[[b2_lo:g]]\pi$.
"""),
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
Simétrica em torno de $n = [[alpha:g]]$, com o coeficiente central próximo de $1 - \omega_c/\pi = 0.5$ (passa-tudo menos passa-baixas).
"""),
        Figure("db", "Magnitude em dB", r"""
A rejeição chega a −[[as_:.1f]] dB, muito além dos [[As_spec:g]] dB exigidos. É o efeito colateral de escolher a janela pela exigência de $R_p$.
"""),
        Figure("scan", "$R_p$ da Hamming em função de $M$", r"""
Para todos os comprimentos testados, a Hamming fica entre [[rp_ham_min:.3f]] e [[rp_ham_max:.3f]] dB, sempre acima do limite de [[Rp_spec:g]] dB. A ondulação não depende do comprimento.
"""),
    ),

    verification=r"""
Critério do livro (dB normalizado, grades com as bordas). O limite decisivo é $R_p$.
""",
    conclusion=r"""
- A exigência $R_p = [[Rp_spec:g]]$ dB equivale a lóbulos de ≈ [[delta1_db:.0f]] dB, e é ela que define a janela.
- Hann: descartada pelo $A_s$ nominal. Hamming: atende $A_s$, mas $R_p$ fica entre [[rp_ham_min:.3f]] e [[rp_ham_max:.3f]] dB para qualquer $M$.
- **Blackman com $M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$): $R_p = [[rp:.5f]]$ dB e $A_s = [[as_:.2f]]$ dB. **Atende.**
- $M = [[M_bla_first]]$ também atenderia, com margem desprezível. A conta da tabela dá [[M_table]]; o [[M]] vem da paridade (V3).
""",
)
