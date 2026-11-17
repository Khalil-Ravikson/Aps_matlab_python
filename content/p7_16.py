r"""Conteúdo didático — P7.16 (passa-faixa com a janela fixa de menor comprimento).

Fonte técnica: DTA §2.1 item 8, §3.1 e §4; Ingle & Proakis, Tabela 7.1 (conforme
DTA). Correção E5 de docs/errata_dta.md (decisão D5: critério medido).
"""

from __future__ import annotations

from functools import lru_cache

from dsp_av2.fir.design import design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_16 as SPEC
from exercises import p7_16

from .helpers import fir_db_context, fir_db_measure, md_table, ok
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable

NAMES = {"hann": "Hann", "hamming": "Hamming", "blackman": "Blackman"}


@lru_cache(maxsize=1)
def objects() -> dict:
    results = p7_16.evaluate_windows()
    d = p7_16.design()
    near = {}
    for r in results:
        if r.window in ("hann", "hamming"):
            near[r.window] = [(M, *fir_db_measure(design_fixed_window(SPEC, r.window, M=M).h, SPEC))
                              for M in range(r.M_min - 3, r.M_min + 2)]
    return {
        "spec": SPEC, "final": d, "final_report": p7_16.validate(d), "results": results, "near": near,
        "bars": {"categories": [NAMES[r.window] for r in results],
                 "series": {"Tabela 7.1": [r.M_table for r in results], "mínimo medido": [r.M_min for r in results]},
                 "ylabel": "comprimento M"},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    d = o["final"]
    ctx = fir_db_context(SPEC, d, o["final_report"])
    by = {r.window: r for r in o["results"]}
    rows = [(NAMES[r.window], f"{FIXED_WINDOWS[r.window].min_stopband_db:g} dB", f"{FIXED_WINDOWS[r.window].transition_factor:g}π/M",
             r.M_table, r.M_min) for r in o["results"]]
    tab_best = min(o["results"], key=lambda r: r.M_table)
    for w in ("hann", "hamming"):
        ctx[f"near_{w}"] = md_table(["$M$", "$R_p$ (dB)", "$A_s$ (dB)", "atende?"],
                                    [(M, f"{rp:.4f}", f"{as_:.2f}", ok(rp <= SPEC.rp_db and as_ >= SPEC.as_db))
                                     for M, rp, as_ in o["near"][w]])
    ctx |= {
        "win_table": md_table(["Janela", "$A_s$ nominal", "transição", "$M$ pela tabela", "$M$ mínimo medido"], rows),
        "M_tab_hann": by["hann"].M_table, "M_tab_ham": by["hamming"].M_table, "M_tab_bla": by["blackman"].M_table,
        "M_min_hann": by["hann"].M_min, "M_min_ham": by["hamming"].M_min, "M_min_bla": by["blackman"].M_min,
        "tab_best": NAMES[tab_best.window], "tab_best_M": tab_best.M_table,
        "As_hann": FIXED_WINDOWS["hann"].min_stopband_db, "As_ham": FIXED_WINDOWS["hamming"].min_stopband_db,
        "spare_hann": FIXED_WINDOWS["hann"].min_stopband_db - SPEC.as_db,
        "spare_ham": FIXED_WINDOWS["hamming"].min_stopband_db - SPEC.as_db,
        "win": NAMES[d.window],
    }
    return ctx


CONTENT = ExerciseContent(
    id="p7_16", number="P7.16", chapter=7,
    title="Passa-faixa com a janela fixa de menor comprimento",
    tagline="FIR · passa-faixa · comparação de janelas fixas · comprimento mínimo",
    highlights=(
        Value(r"\text{janela}", "[[win]]", "resultado", "menor comprimento medido"),
        Value("M", "[[M]]", "resultado", "Tipo [[type]], α = [[alpha:g]]"),
        Value(r"A_s", "[[as_:.2f]] dB", "resultado", "medido (exigido ≥ [[As_spec:g]] dB)"),
        Value(r"M_{\text{tabela}}", "[[tab_best]], [[tab_best_M]]", "calculado", "resposta pelo critério da Tabela 7.1"),
    ),
    errata=("E5",),
    result_keys=("M", "as_", "rp"),

    asked=r"""
Projetar um **filtro FIR passa-faixa** com uma **janela fixa** que tenha o **menor comprimento possível** e atenda a:

- passagem $[[b2_lo:g]]\pi$–$[[b2_hi:g]]\pi$ com $R_p \le [[Rp_spec:g]]$ dB;
- rejeições $\omega \le [[b1_hi:g]]\pi$ e $\omega \ge [[b3_lo:g]]\pi$ com $A_s \ge [[As_spec:g]]$ dB.

Pedem-se a magnitude em dB e o stem de $h(n)$. Ao contrário dos anteriores, aqui **a escolha da janela faz parte da resposta**.
""",
    data_note=r"""
As bordas são as mesmas do P7.11, que exigia 50 dB com a Hamming. Aqui bastam [[As_spec:g]] dB, o que abre espaço para janelas e comprimentos menores.
""",
    data_rows=(
        Value(r"\text{Rejeição inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "a ser atenuada"),
        Value(r"\text{Passagem}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", "a ser preservada"),
        Value(r"\text{Rejeição superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", "a ser atenuada"),
        Value(r"R_p", "[[Rp_spec:g]] dB", "dado", "ondulação máxima"),
        Value(r"A_s", "[[As_spec:g]] dB", "dado", "atenuação mínima"),
        Value(r"\text{Objetivo}", "janela fixa de menor $M$", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Duas leituras de "menor comprimento".**

1. **Pela Tabela 7.1** (critério do DTA): para cada janela com $A_s$ nominal suficiente, $M = \lceil k\pi/\Delta\omega\rceil + 1$, e escolhe-se o menor.
2. **Pela medição** (decisão D5 do plano): para cada janela, testam-se os comprimentos e toma-se o menor que atende $R_p$ e $A_s$ **medidos**. Este é o critério adotado, porque é o que de fato responde "menor comprimento que satisfaz as especificações".

**Candidatas.** Retangular (≈ 21 dB) e Bartlett (≈ 25 dB) não chegam a 40 dB. Restam Hann (≈ [[As_hann:g]] dB), Hamming (≈ [[As_ham:g]] dB) e Blackman (≈ 74 dB).

**Paridade livre.** O passa-faixa não exige ganho em $\pi$, então pares e ímpares são testados.
""",
    deepening=(
        Deepening("Por que a ordem Hann × Hamming se inverte?", r"""
O $\Delta\omega$ da Tabela 7.1 é a largura de transição necessária para a resposta atingir a **atenuação plena da janela**: ≈ [[As_hann:g]] dB na Hann e ≈ [[As_ham:g]] dB na Hamming. Este enunciado exige só [[As_spec:g]] dB.

- A **Hamming** tem [[spare_ham:g]] dB de sobra. A resposta cruza −[[As_spec:g]] dB bem antes do fim da transição tabelada, e o filtro pode encolher de [[M_tab_ham]] para [[M_min_ham]].
- A **Hann** tem só [[spare_hann:g]] dB de sobra, e encolhe pouco: de [[M_tab_hann]] para [[M_min_hann]].

Pela tabela, a Hann parece melhor por ter a transição mais estreita. Na medição, a Hamming vence porque tem mais atenuação para trocar por transição.
"""),
    ),

    equations=(
        Equation(r"M_{\text{tabela}} = \left\lceil \frac{k\pi}{\Delta\omega} \right\rceil + 1, \quad k = 6.2\ (\text{Hann}),\ 6.6\ (\text{Hamming}),\ 11\ (\text{Blackman})",
                 "Comprimento pela Tabela 7.1", "", (Variable("k", "fator de largura de transição da janela"),), "DTA §3.1.C"),
        Equation(r"M_{\min} = \min\{M : R_p(M) \le R_{p,\text{exig}} \ \text{e}\ A_s(M) \ge A_{s,\text{exig}}\}", "Comprimento mínimo medido",
                 "O menor comprimento cuja resposta medida atende às duas exigências.", (), "Decisão D5"),
        Equation(r"h_d(n) = \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}", "Resposta ideal do passa-faixa", "", (), "DTA §3.1.E"),
    ),

    steps=(
        Step("Transição e cortes", "Iguais às do P7.11.",
             r"\Delta\omega = [[dw:g]]\pi, \qquad \omega_{c1} = [[wc1:g]]\pi,\ \ \omega_{c2} = [[wc2:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "calculado", "largura de transição"),)),
        Step("Comparar as janelas pelos dois critérios",
             "Para cada janela admissível, o comprimento pela Tabela 7.1 e o menor comprimento medido (busca em todos os $M$):\n\n[[win_table]]",
             "", (Value(r"\text{tabela}", "[[tab_best]], M = [[tab_best_M]]", "calculado", "critério do DTA"),
                  Value(r"\text{medido}", "[[win]], M = [[M]]", "resultado", "critério adotado (D5)"))),
        Step("Ver onde cada janela passa a atender",
             "Os comprimentos vizinhos ao mínimo de cada janela. **Hann:**\n\n[[near_hann]]\n\n**Hamming:**\n\n[[near_hamming]]",
             "", (Value(r"M_{\min,\text{Hann}}", "[[M_min_hann]]", "resultado", "primeiro M da Hann que atende"),
                  Value(r"M_{\min,\text{Hamming}}", "[[M_min_ham]]", "resultado", "primeiro M da Hamming que atende"))),
        Step("Projeto final e medição", "A janela de menor comprimento medido.",
             r"\text{[[win]]},\ M = [[M]],\ \alpha = [[alpha:g]]: \quad R_p = [[rp:.4f]]\ \text{dB}, \quad A_s = [[as_:.2f]]\ \text{dB}",
             (Value(r"R_p", "[[rp:.4f]] dB", "resultado", "≤ [[Rp_spec:g]] dB"),
              Value(r"A_s", "[[as_:.2f]] dB", "resultado", "≥ [[As_spec:g]] dB, folga de [[as_margin:.2f]] dB"))),
    ),

    implementation=r"""
1. **Avaliação das janelas** (`evaluate_windows`): para cada janela com $A_s$ nominal suficiente, calcula $M$ pela tabela e busca o menor $M$ que atende (`search_min_length`, pares e ímpares), medindo com `validate_fir_db`.
2. **Escolha** (`design`): a janela com o menor $M$ mínimo medido.
""",
    code_roles=(
        CodeRole("exercises.p7_16.evaluate_windows", "Passos 2–3: tabela e busca para cada janela"),
        CodeRole("exercises.p7_16.design", "Passo 4: janela de menor comprimento medido"),
        CodeRole("dsp_av2.fir.design.search_min_length", "Busca do menor M que atende"),
        CodeRole("dsp_av2.validation.validate_fir_db", "Medição de Rp e As"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$ (stem, pedido)", r"""
[[M]] coeficientes simétricos em torno de $\alpha = [[alpha:g]]$ (Tipo [[type]]).
"""),
        Figure("db", "Log-magnitude em dB (pedido)", r"""
Os lóbulos das rejeições ficam logo abaixo de $-[[As_spec:g]]$ dB (o mais alto a −[[as_:.2f]] dB). É o comportamento esperado de um comprimento mínimo: margem pequena.
"""),
        Figure("passband", "Zoom na passagem (verificação de $R_p$)", r"""
Ondulação de [[rp:.3f]] dB, bem abaixo dos [[Rp_spec:g]] dB permitidos. Neste exercício quem limita é $A_s$.
"""),
        Figure("bars", "Comprimento por janela: Tabela 7.1 × mínimo medido", r"""
Pela tabela, a Hann é a menor ([[M_tab_hann]]). Medindo, a Hamming cai para [[M_min_ham]] e passa à frente da Hann ([[M_min_hann]]). A Blackman, com muito mais atenuação que o necessário, fica longe ([[M_min_bla]]).
"""),
    ),

    verification=r"""
Critério do livro (dB normalizado, grades de [[n_book]] e [[n_grid]] pontos com as bordas exatas).
""",
    conclusion=r"""
- **Pelo critério da Tabela 7.1** (o do DTA), a resposta seria [[tab_best]] com $M = [[tab_best_M]]$.
- **Pelo critério medido** (adotado): **[[win]] com $M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$), com $R_p = [[rp:.3f]]$ dB e $A_s = [[as_:.2f]]$ dB. **Atende.**
- Hann chega a no mínimo $M = [[M_min_hann]]$, e Blackman a $M = [[M_min_bla]]$.
- A inversão vem da sobra de atenuação: a Hamming troca seus [[spare_ham:g]] dB excedentes por uma transição mais estreita (E5).
""",
)
