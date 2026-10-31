"""Notebook p7_11."""

from .common import SETUP, code, md


def cells():
    return [
        md(r"""
# P7.11 — Filtro passa-faixa FIR com janela de Hamming

**Referência:** Ingle & Proakis, *DSP Using MATLAB*, 3ª ed., p. 379 · DTA §2.1 (item 3) e §4 (Resolução P7.11).

> *Design a bandpass filter using the Hamming window design technique. The specifications are: lower stopband edge: 0.3π, As = 50 dB; upper stopband edge: 0.6π; lower passband edge: 0.4π, Rp = 0.5 dB; upper passband edge: 0.5π. Plot the impulse response and the magnitude response (in dB) of the designed filter. Do not use the fir1 function.*

| Parâmetro | Valor |
|---|---|
| Rejeição inferior | $0 \le \omega \le 0.3\pi$, $A_s = 50$ dB |
| Passagem | $0.4\pi \le \omega \le 0.5\pi$, $R_p = 0.5$ dB |
| Rejeição superior | $0.6\pi \le \omega \le \pi$ |
| Método | janela de Hamming (imposta), sem `fir1` |
"""),
        code(SETUP),
        md(r"""
## 1. Método

O procedimento é o do P7.10 aplicado a um passa-faixa.

**Largura de transição:** $\Delta\omega = 0.4\pi - 0.3\pi = 0.6\pi - 0.5\pi = 0.1\pi$.

**Janela e comprimento.** A Hamming tem $A_s \approx 53$ dB $\ge 50$ dB (Tabela 7.1):

$$M = \left\lceil \frac{6.6\pi}{0.1\pi} \right\rceil + 1 = 67 \quad\Rightarrow\quad \text{Tipo 1},\ \alpha = 33$$

O passa-faixa não exige ganho em $\omega = \pi$, então M par (Tipo 2) seria permitido (decisão D4). Aqui a tabela já dá um valor ímpar.

**Cortes ideais:** $\omega_{c1} = \frac{0.3\pi + 0.4\pi}{2} = 0.35\pi$, $\omega_{c2} = \frac{0.5\pi + 0.6\pi}{2} = 0.55\pi$.

$$h_d(n) = \frac{\sin[\omega_{c2}(n-\alpha)] - \sin[\omega_{c1}(n-\alpha)]}{\pi(n-\alpha)}, \qquad h(n) = h_d(n)\,w_{Hamming}(n)$$
"""),
        code(r'''
import numpy as np
import matplotlib.pyplot as plt

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.specs import P7_11 as SPEC
from exercises import p7_11

d = p7_11.design()
print(f"Δω = {d.transition_width:g}π")
print(f"ωc1 = {d.cutoffs[0]:g}π, ωc2 = {d.cutoffs[1]:g}π")
print(f"M = {d.M}  (Tabela 7.1: {d.M_table}),  α = {d.alpha:g}")
print("Tipo de fase linear:", analysis.linear_phase_type(d.h))
'''),
        md(r"""
## 2. Resultados

Pedidos pelo enunciado: $h(n)$ e magnitude em dB. O terceiro painel (zoom na passagem) mostra a verificação de $R_p$.
"""),
        code(r'''
fig = p7_11.plot(d)
plt.show()
'''),
        md(r"""
## 3. Validação contra a especificação
"""),
        code(r'''
print(p7_11.validate(d))

w = analysis.frequency_grid(SPEC.edges, 200001)
db = analysis.magnitude_db(analysis.frequency_response(d.h, w))
stop = (w <= 0.3 * np.pi + 1e-12) | (w >= 0.6 * np.pi - 1e-12)
i = np.argmax(np.where(stop, db, -np.inf))
print(f"\nPior ponto da rejeição: ω = {w[i] / np.pi:.4f}π, |H| = {db[i]:.4f} dB")
'''),
        md(r"""
Como no P7.10, o pior ponto da rejeição é uma borda, aqui $0.6\pi$, no fim da transição. Desta vez a folga é maior (≈ 1.2 dB).

### 3.1 Observação: comprimentos vizinhos

O enunciado não pede comprimento mínimo, mas a varredura mostra um fato útil para o P7.16, que tem as mesmas bordas: **M = 66 (Tipo 2) também atende**, e com atenuação maior que M = 67.
"""),
        code(r'''
wg = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
print(" M  Tipo   Rp [dB]   As [dB]   atende?")
for M in range(62, 69):
    dm = design_fixed_window(SPEC, "hamming", M=M)
    rp, as_ = analysis.measure_rp_as(dm.h, SPEC.passbands, SPEC.stopbands, wg)
    ok = rp <= SPEC.rp_db and as_ >= SPEC.as_db
    print(f"{M:3d}   {analysis.linear_phase_type(dm.h)}    {rp:7.4f}   {as_:7.2f}   {'sim' if ok else 'não'}")
'''),
        md(r"""
A atenuação não cresce de forma monotônica com M: 66 → 52.0 dB e 67 → 51.2 dB. O nível dos lóbulos e a posição exata do fim da transição em relação à borda mudam de um comprimento para o outro. Por isso, "comprimento mínimo" só pode ser afirmado com uma varredura medida, e não pela fórmula da tabela (decisão D2).

### 3.2 Verificação independente e comparação com o DTA
"""),
        code(r'''
from scipy.signal import firwin
ref = firwin(d.M, list(d.cutoffs), window="hamming", pass_zero=False, scale=False)
print("max |h − h_firwin| =", np.max(np.abs(d.h - ref)))
rp_2049, as_2049 = analysis.measure_rp_as(d.h, SPEC.passbands, SPEC.stopbands, np.linspace(0, np.pi, 2049))
print(f"Grade de 2049 pontos (sem as bordas): Rp = {rp_2049:.4f} dB, As = {as_2049:.2f} dB")
'''),
        md(r"""
O DTA informa $A_s = 51.46$ dB e $R_p = 0.0453$ dB. A grade de 2049 pontos, que pula as bordas, reproduz $R_p$ exatamente e $A_s$ quase exatamente (51.49 dB): a mesma origem da divergência do P7.10. Com as bordas na grade, os valores corretos são $A_s = 51.20$ dB e $R_p = 0.049$ dB.

## 4. Conclusão

- Com a janela de Hamming e $M = 67$ (Tipo 1, $\alpha = 33$), o filtro atende à especificação: $R_p \approx 0.049$ dB $\le 0.5$ dB e $A_s \approx 51.20$ dB $\ge 50$ dB.
- O pior ponto da rejeição é a borda $0.6\pi$, a mesma causa da margem estreita do P7.10.
- $M = 66$ (Tipo 2) também atende. Como o enunciado só pede o projeto pela tabela, a resposta é $M = 67$, mas o fato mostra que a tabela é uma estimativa.
"""),
    ]
