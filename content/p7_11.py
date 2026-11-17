r"""Conteúdo didático — P7.11 (passa-faixa com janela de Hamming).

Fonte técnica: DTA §2.1 item 3, §3.1 e §4; Ingle & Proakis, Tabela 7.1 (conforme
DTA). Item V2 de docs/errata_dta.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis as fa
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_11 as SPEC
from exercises import p7_11

from .helpers import fir_db_context, fir_db_measure, md_table, ok, worst_stop_point
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable

SCAN = range(62, 69)


@lru_cache(maxsize=1)
def objects() -> dict:
    d = p7_11.design()
    rows = []
    for M in SCAN:
        dm = design_fixed_window(SPEC, "hamming", M=M)
        rows.append((M, fa.linear_phase_type(dm.h), *fir_db_measure(dm.h, SPEC)))
    return {
        "spec": SPEC, "final": d, "final_report": p7_11.validate(d), "scan_rows": rows,
        "scan": {"x": [r[0] for r in rows], "y": [r[3] for r in rows], "limit": SPEC.as_db, "better": "above",
                 "xlabel": "M", "ylabel": "As medido (dB)", "limit_label": f"As exigido = {SPEC.as_db:g} dB",
                 "highlight": d.M, "highlight_label": f"M = {d.M} (Tabela 7.1)"},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    d = o["final"]
    ctx = fir_db_context(SPEC, d, o["final_report"])
    k = FIXED_WINDOWS["hamming"].transition_factor
    worst_w, worst_db = worst_stop_point(d.h, SPEC)
    rp_2049, as_2049 = fa.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, np.linspace(0, np.pi, 2049))
    rows = [(M, t, f"{rp:.4f}", f"{as_:.2f}", ok(rp <= SPEC.rp_db and as_ >= SPEC.as_db)) for M, t, rp, as_ in o["scan_rows"]]
    passing = [r for r in o["scan_rows"] if r[2] <= SPEC.rp_db and r[3] >= SPEC.as_db]
    first = passing[0]
    ctx |= {"k": k, "As_win": FIXED_WINDOWS["hamming"].min_stopband_db, "ratio": k / d.transition_width,
            "worst_w": worst_w, "worst_db": -worst_db, "as_2049": as_2049, "rp_2049": rp_2049,
            "scan_table": md_table(["$M$", "Tipo", "$R_p$ (dB)", "$A_s$ (dB)", "atende?"], rows),
            "M_first": first[0], "type_first": first[1], "as_first": first[3],
            "scan_lo": SCAN[0], "scan_hi": SCAN[-1]}
    return ctx


CONTENT = ExerciseContent(
    id="p7_11", number="P7.11", chapter=7,
    title="Passa-faixa FIR com janela de Hamming",
    tagline="FIR · passa-faixa · janela de Hamming · Tabela 7.1",
    highlights=(
        Value("M", "[[M]]", "resultado", "comprimento (Tipo [[type]], α = [[alpha:g]])"),
        Value(r"R_p", "[[rp:.3f]] dB", "resultado", "medido (exigido ≤ [[Rp_spec:g]] dB)"),
        Value(r"A_s", "[[as_:.2f]] dB", "resultado", "medido (exigido ≥ [[As_spec:g]] dB)"),
        Value(r"\omega_{c1},\ \omega_{c2}", r"[[wc1:g]]π, [[wc2:g]]π", "calculado", "frequências de corte"),
    ),
    errata=("V2",),
    result_keys=("M", "rp", "as_", "as_2049", "as_first"),

    asked=r"""
Projetar um **filtro FIR passa-faixa** estreito (passagem entre $[[b2_lo:g]]\pi$ e $[[b2_hi:g]]\pi$) com a **janela de Hamming**.

- Ondulação na passagem de no máximo $R_p = [[Rp_spec:g]]$ dB.
- Atenuação de pelo menos $A_s = [[As_spec:g]]$ dB nas duas rejeições.

O enunciado pede $h(n)$ e a magnitude em dB. O procedimento é o do P7.10, agora para um passa-faixa.
""",
    data_note=r"""
As bordas deste exercício são as mesmas do P7.16, que exige apenas 40 dB e pede o comprimento mínimo. Vale comparar os dois.
""",
    data_rows=(
        Value(r"\text{Rejeição inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "a ser atenuada"),
        Value(r"\text{Passagem}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", "a ser preservada"),
        Value(r"\text{Rejeição superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", "a ser atenuada"),
        Value(r"R_p", "[[Rp_spec:g]] dB", "dado", "ondulação máxima na passagem"),
        Value(r"A_s", "[[As_spec:g]] dB", "dado", "atenuação mínima"),
        Value(r"\text{Método}", "janela de Hamming, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Passa-faixa ideal** = diferença de dois passa-baixas ideais, com cortes $\omega_{c1} < \omega_{c2}$.

**Paridade livre.** Um passa-faixa não exige ganho em $\omega = \pi$, que fica na rejeição. Por isso $M$ par (Tipo 2, com $H_r(\pi) = 0$ obrigatório) também é admissível (decisão D4 do plano). Aqui a Tabela 7.1 já dá um valor ímpar.

**Janela de Hamming:** $\Delta\omega \approx 6.6\pi/M$ e $A_s \approx 53$ dB (Tabela 7.1, conforme o DTA).
""",
    deepening=(
        Deepening("A atenuação cresce sempre com M?", r"""
Não necessariamente. Ao variar $M$, mudam duas coisas: a largura da transição e a posição exata dos lóbulos em relação às bordas. A varredura do passo 5 mostra $A_s$ maior com $M = [[M_first]]$ do que com $M = [[M]]$. Por isso "comprimento mínimo" só pode ser afirmado medindo cada candidato (P7.13–P7.16), e não pela fórmula.
"""),
    ),

    equations=(
        Equation(r"M = \left\lceil \frac{6.6\pi}{\Delta\omega} \right\rceil + 1", "Comprimento pela Tabela 7.1 (Hamming)", "", (), "DTA §3.1.C"),
        Equation(r"w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{M-1}\right)", "Janela de Hamming", "", (), "DTA §3.1.C"),
        Equation(r"h_d(n) = \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}", "Resposta ideal do passa-faixa",
                 "Diferença de dois passa-baixas ideais.", (Variable(r"\alpha = (M-1)/2", "atraso"),), "DTA §3.1.E"),
        Equation(r"h(n) = h_d(n)\,w(n)", "Filtro projetado", "", (), "DTA §3.1"),
    ),

    steps=(
        Step("Determinar a largura de transição", "As duas transições são iguais.",
             r"\Delta\omega = [[b2_lo:g]]\pi - [[b1_hi:g]]\pi = [[b3_lo:g]]\pi - [[b2_hi:g]]\pi = [[dw:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "calculado", "largura de transição"),)),
        Step("Conferir a janela e calcular o comprimento",
             "A Hamming é imposta pelo enunciado e atende ao $A_s$ exigido pela tabela ([[As_win:g]] ≥ [[As_spec:g]] dB).",
             r"M = \left\lceil \frac{[[k:g]]\pi}{[[dw:g]]\pi} \right\rceil + 1 = [[ratio:.0f]] + 1 = [[M]], \qquad \alpha = [[alpha:g]]",
             (Value("M", "[[M]]", "calculado", "comprimento (Tipo [[type]])"),)),
        Step("Posicionar as frequências de corte", "No meio de cada transição.",
             r"\omega_{c1} = \frac{[[b1_hi:g]]\pi + [[b2_lo:g]]\pi}{2} = [[wc1:g]]\pi, \qquad \omega_{c2} = \frac{[[b2_hi:g]]\pi + [[b3_lo:g]]\pi}{2} = [[wc2:g]]\pi",
             (Value(r"\omega_{c1}, \omega_{c2}", r"$[[wc1:g]]\pi$, $[[wc2:g]]\pi$", "calculado", "cortes"),)),
        Step("Construir o filtro e medir", "Medição em [[n_grid]] frequências, com as bordas.",
             r"R_p = [[rp:.4f]]\ \text{dB}, \qquad A_s = [[as_:.2f]]\ \text{dB}",
             (Value(r"R_p", "[[rp:.4f]] dB", "resultado", "≤ [[Rp_spec:g]] dB"),
              Value(r"A_s", "[[as_:.2f]] dB", "resultado", "folga de [[as_margin:.2f]] dB; pior ponto em $[[worst_w:.4f]]\\pi$"))),
        Step("Observar os comprimentos vizinhos (complementar)",
             "O enunciado não pede o mínimo, mas a varredura mostra um fato útil para o P7.16:\n\n[[scan_table]]",
             r"M = [[M_first]]\ (\text{Tipo } [[type_first]]):\ A_s = [[as_first:.2f]]\ \text{dB} \ge [[As_spec:g]]",
             (Value(r"M = [[M_first]]", "atende", "resultado", "comprimento par (Tipo [[type_first]]) também atende"),),
             note="$M = [[M_first]]$ também atende, com $A_s$ **maior** que o de $M = [[M]]$. A resposta segue a regra da Tabela 7.1, mas fica claro que a tabela é uma estimativa."),
    ),

    implementation=r"""
1. **Projeto** (`design_fixed_window(SPEC, "hamming")`): $\Delta\omega$, $M$, cortes e $h(n)$.
2. **Medição** (`validate_fir_db`): $R_p$ e $A_s$ nas grades de [[n_book]] e [[n_grid]] pontos com as bordas.
3. **Varredura complementar**: o mesmo projeto para cada $M$ de [[scan_lo]] a [[scan_hi]], pares e ímpares.
""",
    code_roles=(
        CodeRole("dsp_av2.fir.design.design_fixed_window", "Passos 1–3"),
        CodeRole("dsp_av2.fir.ideal.ideal_bandpass", "Resposta ideal do passa-faixa"),
        CodeRole("dsp_av2.validation.validate_fir_db", "Passo 4: mede Rp e As"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
Simétrica em torno de $n = [[alpha:g]]$ (Tipo [[type]]). A oscilação dos coeficientes tem período próximo de $2\pi/\omega_0$, em que $\omega_0 = (\omega_{c1}+\omega_{c2})/2$ é o centro da passagem.
"""),
        Figure("db", "Magnitude em dB", r"""
Passagem estreita em torno de $[[wc1:g]]\pi$–$[[wc2:g]]\pi$. Os lóbulos das rejeições ficam abaixo de $-[[As_spec:g]]$ dB. O ponto mais crítico é a borda $[[worst_w:.4f]]\pi$, a −[[worst_db:.2f]] dB.
"""),
        Figure("passband", "Zoom na passagem (verificação de $R_p$)", r"""
Ondulação máxima de [[rp:.3f]] dB, contra [[Rp_spec:g]] dB permitidos.
"""),
        Figure("scan", "$A_s$ medido para comprimentos vizinhos", r"""
A atenuação não cresce de forma monotônica com $M$. $M = [[M_first]]$ (par) atende e supera $M = [[M]]$, que é o valor da Tabela 7.1.
"""),
    ),

    verification=r"""
Critério do livro (dB normalizado, grades com as bordas exatas). $R_p$ é a maior queda na passagem e $A_s$ a menor atenuação nas rejeições.
""",
    conclusion=r"""
- Hamming com **$M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$), cortes em $[[wc1:g]]\pi$ e $[[wc2:g]]\pi$.
- **Atende**: $R_p = [[rp:.3f]]$ dB e $A_s = [[as_:.2f]]$ dB (folga de [[as_margin:.2f]] dB).
- $M = [[M_first]]$ (Tipo [[type_first]]) também atenderia. A tabela é uma estimativa, e a atenuação não é monotônica em $M$.
- O DTA informa 51.46 dB e 0.0453 dB. A grade de 2049 pontos, que não contém as bordas, dá [[as_2049:.2f]] dB e [[rp_2049:.4f]] dB, a mesma origem do item V1. Os valores corretos são os medidos com as bordas (V2).
""",
)
