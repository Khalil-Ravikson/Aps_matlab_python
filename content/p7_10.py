r"""Conteúdo didático — P7.10 (rejeita-faixa com janela de Hamming).

Fonte técnica: DTA §2.1 item 2, §3.1 e §4; Ingle & Proakis, Tabela 7.1 (conforme
DTA). Item V1 de docs/errata_dta.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis as fa
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.fir.windows import FIXED_WINDOWS
from dsp_av2.specs import P7_10 as SPEC
from exercises import p7_10

from .helpers import fir_db_context, fir_db_measure, md_table, ok, worst_stop_point
from .model import CodeRole, Deepening, Equation, ExerciseContent, Figure, Step, Value, Variable

SCAN = range(61, 70, 2)


@lru_cache(maxsize=1)
def objects() -> dict:
    d = p7_10.design()
    scan = [(M, *fir_db_measure(design_fixed_window(SPEC, "hamming", M=M).h, SPEC)) for M in SCAN]
    return {
        "spec": SPEC, "final": d, "final_report": p7_10.validate(d), "scan_rows": scan,
        "scan": {"x": [r[0] for r in scan], "y": [r[2] for r in scan], "limit": SPEC.as_db, "better": "above",
                 "xlabel": "M (ímpares)", "ylabel": "As medido (dB)", "limit_label": f"As exigido = {SPEC.as_db:g} dB",
                 "highlight": d.M, "highlight_label": f"M = {d.M}"},
        "overlay_stop": {"curves": {f"Hamming, M = {d.M}": d.h}, "xlim": (0.36, 0.64), "ylim": (-75, -35),
                         "normalized": True, "limit_db": -SPEC.as_db, "limit_label": f"−As = −{SPEC.as_db:g} dB",
                         "edges": [0.4, 0.6]},
    }


@lru_cache(maxsize=1)
def context() -> dict:
    o = objects()
    d = o["final"]
    ctx = fir_db_context(SPEC, d, o["final_report"])
    k = FIXED_WINDOWS["hamming"].transition_factor
    worst_w, worst_db = worst_stop_point(d.h, SPEC)
    _, as_2049 = fa.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, np.linspace(0, np.pi, 2049))
    a = int(d.alpha)
    odd_zero = max(abs(d.h[a + m]) for m in range(1, a + 1, 2))
    rows = [(M, f"{rp:.4f}", f"{as_:.2f}", ok(rp <= SPEC.rp_db and as_ >= SPEC.as_db)) for M, rp, as_ in o["scan_rows"]]
    prev = [r for r in o["scan_rows"] if r[0] == d.M - 2][0]
    ctx |= {"k": k, "As_win": FIXED_WINDOWS["hamming"].min_stopband_db, "ratio": k / d.transition_width,
            "worst_w": worst_w, "worst_db": -worst_db, "as_2049": as_2049, "odd_zero": odd_zero,
            "scan_table": md_table(["$M$", "$R_p$ (dB)", "$A_s$ (dB)", "atende?"], rows),
            "Mprev": prev[0], "as_prev": prev[2], "half_bw": (d.cutoffs[1] - d.cutoffs[0]) / 2,
            "scan_lo": SCAN[0], "scan_hi": SCAN[-1]}
    return ctx


CONTENT = ExerciseContent(
    id="p7_10", number="P7.10", chapter=7,
    title="Rejeita-faixa FIR com janela de Hamming",
    tagline="FIR · rejeita-faixa · janela de Hamming · Tabela 7.1",
    highlights=(
        Value("M", "[[M]]", "resultado", "comprimento (Tipo [[type]], α = [[alpha:g]])"),
        Value(r"R_p", "[[rp:.3f]] dB", "resultado", "medido (exigido ≤ [[Rp_spec:g]] dB)"),
        Value(r"A_s", "[[as_:.2f]] dB", "resultado", "medido: folga de só [[as_margin:.2f]] dB"),
        Value(r"\omega_{c1},\ \omega_{c2}", r"[[wc1:g]]π, [[wc2:g]]π", "calculado", "frequências de corte"),
    ),
    errata=("V1",),
    result_keys=("M", "rp", "as_", "as_2049"),

    asked=r"""
Projetar um **filtro FIR rejeita-faixa** que elimine a faixa $[[b2_lo:g]]\pi$–$[[b2_hi:g]]\pi$ e preserve o restante, usando obrigatoriamente a **janela de Hamming**.

- Nas duas passagens, a ondulação não pode passar de $R_p = [[Rp_spec:g]]$ dB.
- Na rejeição, a atenuação deve ser de pelo menos $A_s = [[As_spec:g]]$ dB.

O enunciado pede $h(n)$ e a magnitude em dB.
""",
    data_note=r"""
Um rejeita-faixa precisa de ganho 1 em $\omega = \pi$ (passagem superior). Esse fato restringe a paridade de $M$ no passo 3.
""",
    data_rows=(
        Value(r"\text{Passagem inferior}", r"$0 \le \omega \le [[b1_hi:g]]\pi$", "dado", "a ser preservada"),
        Value(r"\text{Rejeição}", r"$[[b2_lo:g]]\pi \le \omega \le [[b2_hi:g]]\pi$", "dado", "a ser eliminada"),
        Value(r"\text{Passagem superior}", r"$[[b3_lo:g]]\pi \le \omega \le \pi$", "dado", "a ser preservada"),
        Value(r"R_p", "[[Rp_spec:g]] dB", "dado", "ondulação máxima nas passagens"),
        Value(r"A_s", "[[As_spec:g]] dB", "dado", "atenuação mínima na rejeição"),
        Value(r"\text{Método}", "janela de Hamming, sem `fir1`", "dado", "imposto pelo enunciado"),
    ),

    theory=r"""
**Rejeita-faixa ideal.** É o complemento de um passa-faixa: um passa-tudo (o impulso atrasado $\delta(n-\alpha)$) menos um passa-faixa ideal.

**Paridade.** Com $M$ par (Tipo 2), a resposta de amplitude tem zero obrigatório em $\omega = \pi$, porque $\alpha$ é semi-inteiro. Além disso, $\delta(n-\alpha)$ nem existe como sequência. Um rejeita-faixa precisa passar $\omega = \pi$, então **só $M$ ímpar serve** (Tipo 1).

**Janela de Hamming** (Tabela 7.1, conforme o DTA): $\Delta\omega \approx 6.6\pi/M$ e $A_s \approx 53$ dB. Como $53 \ge [[As_spec:g]]$, ela é capaz de atender.

**Cuidado numérico.** A razão $6.6\pi/\Delta\omega$ deveria dar um inteiro exato, mas em ponto flutuante pode sair $66.000\ldots01$. Nesse caso o teto subiria uma unidade e $M$ ficaria uma amostra maior que o correto. O código arredonda a razão antes do teto.
""",
    deepening=(
        Deepening("Por que a folga em A_s é tão pequena?", r"""
A tabela atribui ≈ 53 dB à Hamming, mas o filtro medido dá [[as_:.2f]] dB. O pior ponto da rejeição **não é um lóbulo lateral**: os lóbulos ficam em torno de −52 dB. O pior ponto é a própria borda $\omega = [[worst_w:.4f]]\pi$, onde a transição ainda não terminou.

Ou seja, a largura $6.6\pi/M$ é aproximada: com $M = [[M]]$, a resposta só alcança o nível dos lóbulos um pouco depois da borda. Um passa-baixas Hamming com o mesmo $M$ e a mesma transição tem a mesma margem, então o efeito não vem de haver duas transições.
"""),
        Deepening("Por que metade dos coeficientes é zero?", r"""
A rejeição está centrada em $(\omega_{c1} + \omega_{c2})/2 = \pi/2$. Pela identidade $\sin a - \sin b = 2\cos\frac{a+b}{2}\sin\frac{a-b}{2}$, o termo passa-faixa da resposta ideal vale

$$\frac{2\cos[\tfrac{\pi}{2}(n-\alpha)]\,\sin[[[half_bw:g]]\pi\,(n-\alpha)]}{\pi(n-\alpha)}, \qquad [[half_bw:g]]\pi = \frac{\omega_{c2}-\omega_{c1}}{2},$$

e o cosseno se anula sempre que $n - \alpha$ é ímpar. Medido no filtro: $\max |h(\alpha \pm \text{ímpar})| = [[odd_zero:.1e]]$, ou seja, zero numérico.
"""),
    ),

    equations=(
        Equation(r"M = \left\lceil \frac{6.6\pi}{\Delta\omega} \right\rceil + 1", "Comprimento pela Tabela 7.1 (Hamming)",
                 "Comprimento necessário para a transição da Hamming caber em $\\Delta\\omega$.",
                 (Variable(r"\Delta\omega", "largura de transição"),), "DTA §3.1.C"),
        Equation(r"w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{M-1}\right)", "Janela de Hamming", "Coeficientes da janela.", (), "DTA §3.1.C"),
        Equation(r"h_d(n) = \delta(n-\alpha) - \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}",
                 "Resposta ideal do rejeita-faixa", "Passa-tudo menos passa-faixa.",
                 (Variable(r"\delta(n-\alpha)", "impulso atrasado; exige $\\alpha$ inteiro ($M$ ímpar)"),
                  Variable(r"\omega_{c1}, \omega_{c2}", "cortes no meio das transições")), "DTA §3.1.E"),
        Equation(r"h(n) = h_d(n)\,w(n)", "Filtro projetado", "", (), "DTA §3.1"),
        Equation(r"R_p = -\min_{\text{passagem}} |H|_{dB}, \qquad A_s = -\max_{\text{rejeição}} |H|_{dB}", "Medição (convenção do livro)",
                 "Com $|H|_{dB}$ normalizado pelo máximo.", (), "Ingle & Proakis, `freqz_m`"),
    ),

    steps=(
        Step("Determinar a largura de transição",
             "As duas transições têm a mesma largura.",
             r"\Delta\omega = [[b2_lo:g]]\pi - [[b1_hi:g]]\pi = [[b3_lo:g]]\pi - [[b2_hi:g]]\pi = [[dw:g]]\pi",
             (Value(r"\Delta\omega", r"$[[dw:g]]\pi$", "calculado", "largura de transição"),)),
        Step("Conferir a janela imposta",
             "A Hamming é imposta pelo enunciado. A Tabela 7.1 indica que ela é capaz de atender ao $A_s$ exigido.",
             r"A_{s,\text{Hamming}} \approx [[As_win:g]]\ \text{dB} \;\ge\; [[As_spec:g]]\ \text{dB}",
             (Value(r"A_{s,\text{Hamming}}", "[[As_win:g]] dB", "criterio", "valor nominal da Tabela 7.1"),)),
        Step("Calcular o comprimento e verificar a paridade",
             "Pela Tabela 7.1. O resultado já é ímpar, como o rejeita-faixa exige (Tipo 1).",
             r"M = \left\lceil \frac{[[k:g]]\pi}{[[dw:g]]\pi} \right\rceil + 1 = [[ratio:.0f]] + 1 = [[M]], \qquad \alpha = [[alpha:g]]",
             (Value("M", "[[M]]", "calculado", "comprimento (Tipo [[type]])"),
              Value(r"\alpha", "[[alpha:g]]", "calculado", "atraso"))),
        Step("Posicionar as frequências de corte",
             "No meio de cada transição.",
             r"\omega_{c1} = [[wc1:g]]\pi, \qquad \omega_{c2} = [[wc2:g]]\pi",
             (Value(r"\omega_{c1}, \omega_{c2}", r"$[[wc1:g]]\pi$, $[[wc2:g]]\pi$", "calculado", "cortes"),)),
        Step("Construir o filtro e medir",
             "Medição em [[n_grid]] frequências, com as bordas incluídas.",
             r"R_p = [[rp:.4f]]\ \text{dB}, \qquad A_s = [[as_:.4f]]\ \text{dB}",
             (Value(r"R_p", "[[rp:.4f]] dB", "resultado", "≤ [[Rp_spec:g]] dB"),
              Value(r"A_s", "[[as_:.4f]] dB", "resultado", "≥ [[As_spec:g]] dB, folga de [[as_margin:.3f]] dB")),
             note="A folga em $A_s$ é de só [[as_margin:.2f]] dB. O pior ponto é a borda $[[worst_w:.4f]]\\pi$ (ver aprofundamento)."),
        Step("Conferir se o comprimento é mínimo (complementar)",
             "O enunciado não pede comprimento mínimo, mas a varredura dos ímpares vizinhos mostra que o valor da tabela é também o menor que atende:\n\n[[scan_table]]",
             r"M = [[Mprev]]: A_s = [[as_prev:.2f]]\ \text{dB} < [[As_spec:g]] \quad\Rightarrow\quad M = [[M]] \text{ é o menor ímpar que atende}",
             (Value(r"M_{\min,\text{ímpar}}", "[[M]]", "resultado", "coincide com a Tabela 7.1"),)),
    ),

    implementation=r"""
1. **Projeto** (`design_fixed_window(SPEC, "hamming")`): $\Delta\omega$, $M$ (com o arredondamento protegido), ajuste de paridade, cortes e $h(n)$.
2. **Medição** (`validate_fir_db`): $R_p$ e $A_s$ nas grades de [[n_book]] e [[n_grid]] pontos, ambas com as bordas.
3. **Varredura complementar**: o mesmo projeto para $M$ ímpar de [[scan_lo]] a [[scan_hi]].
""",
    code_roles=(
        CodeRole("dsp_av2.fir.design.design_fixed_window", "Passos 1–4"),
        CodeRole("dsp_av2.fir.windows.fixed_window_length", "M pela Tabela 7.1, com teto protegido contra ruído numérico"),
        CodeRole("dsp_av2.fir.ideal.ideal_bandstop", "Resposta ideal do rejeita-faixa (exige M ímpar)"),
        CodeRole("dsp_av2.validation.validate_fir_db", "Passo 5: mede Rp e As"),
    ),

    figures=(
        Figure("impulse", "Resposta ao impulso $h(n)$", r"""
Simétrica em torno de $n = [[alpha:g]]$. O coeficiente central, que contém o impulso do passa-tudo, domina. As amostras em $\alpha \pm 1, \alpha \pm 3, \ldots$ são nulas (aprofundamento).
"""),
        Figure("db", "Magnitude em dB", r"""
Passagens em 0 dB e uma rejeição profunda entre $[[b2_lo:g]]\pi$ e $[[b2_hi:g]]\pi$. Os lóbulos internos ficam perto de −52 dB. Junto às bordas, a curva passa rente à linha de $-[[As_spec:g]]$ dB.
"""),
        Figure("passband", "Zoom nas passagens (verificação de $R_p$)", r"""
Ondulação máxima de [[rp:.3f]] dB, muito abaixo do limite de [[Rp_spec:g]] dB.
"""),
        Figure("overlay_stop", "Zoom na rejeição: onde está o pior ponto", r"""
Em escala de dB ampliada, fica claro que os lóbulos internos têm folga, e que é a borda $[[worst_w:.4f]]\pi$, a −[[worst_db:.2f]] dB, que define $A_s$.
"""),
        Figure("scan", "$A_s$ medido para comprimentos ímpares vizinhos", r"""
Com $M = [[Mprev]]$, a transição não cabe e $A_s$ cai para [[as_prev:.2f]] dB. Com $M = [[M]]$, a especificação é atendida.
"""),
    ),

    verification=r"""
Critério do livro: com $|H|_{dB}$ normalizado, $R_p$ é a maior queda nas passagens e $A_s$ a menor atenuação na rejeição. As grades incluem as bordas: aqui isso é decisivo, porque o pior ponto **é** a borda.
""",
    conclusion=r"""
- Hamming com **$M = [[M]]$** (Tipo [[type]], $\alpha = [[alpha:g]]$), cortes em $[[wc1:g]]\pi$ e $[[wc2:g]]\pi$.
- **Atende**: $R_p = [[rp:.3f]]$ dB e $A_s = [[as_:.2f]]$ dB, com folga de só [[as_margin:.2f]] dB. O pior ponto é a borda da rejeição, não um lóbulo lateral.
- $M = [[M]]$ é também o menor ímpar que atende.
- O DTA informa 51.11 dB, valor que só aparece numa grade de 2049 pontos que **não contém as bordas** (com ela, obtém-se [[as_2049:.2f]] dB). O valor correto é [[as_:.2f]] dB (V1).
""",
)
