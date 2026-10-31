"""Notebook p7_12."""

from .common import SETUP, code, md


def cells():
    return [
        md(r"""
# P7.12 — Filtro passa-altas FIR com janela fixa

**Referência:** Ingle & Proakis, *DSP Using MATLAB*, 3ª ed., p. 379 · DTA §2.1 (item 4) e §4 (Resolução P7.12).

> *Design a highpass filter using one of the fixed window functions. The specifications are: stopband edge: 0.4π, As = 50 dB; passband edge: 0.6π, Rp = 0.004 dB. Plot the zoomed magnitude response (in dB) of the designed filter in the passband to verify the passband ripple Rp. Do not use the fir1 function.*

| Parâmetro | Valor |
|---|---|
| Rejeição | $0 \le \omega \le 0.4\pi$, $A_s = 50$ dB |
| Passagem | $0.6\pi \le \omega \le \pi$, $R_p = 0.004$ dB |
| Método | uma janela fixa (a escolher), sem `fir1` |
"""),
        code(SETUP),
        md(r"""
## 1. O ponto central: $R_p = 0.004$ dB é muito exigente

Convertendo para ondulação linear: $R_p = -20\log_{10}\frac{1-\delta_1}{1+\delta_1}$, logo

$$\delta_1 = \frac{10^{R_p/20} - 1}{10^{R_p/20} + 1} \approx 2.3\times10^{-4} \quad\Rightarrow\quad -20\log_{10}\delta_1 \approx 73\ \text{dB}$$

No método da janela, a ondulação da passagem e a da rejeição têm aproximadamente a mesma amplitude, $\delta_1 \approx \delta_2$, porque as duas vêm dos lóbulos laterais da janela. Exigir $\delta_1 \approx 2.3\times10^{-4}$ equivale, na prática, a exigir uma janela com ~73 dB de lóbulos. Quem decide a janela é o $R_p$, não o $A_s = 50$ dB.

**Largura de transição e corte:** $\Delta\omega = 0.6\pi - 0.4\pi = 0.2\pi$, $\omega_c = 0.5\pi$.

**Paridade:** passa-altas exige $H(\pi) \neq 0$, logo M ímpar (Tipo 1).

## 2. Escolha da janela

Cada janela fixa da Tabela 7.1 é avaliada: primeiro pelo $A_s$ nominal da tabela e depois pela medição de $R_p$ e $A_s$ no filtro projetado.
"""),
        code(r'''
import numpy as np
import matplotlib.pyplot as plt

from dsp_av2.fir import analysis
from dsp_av2.fir.design import design_fixed_window
from dsp_av2.specs import P7_12 as SPEC
from exercises import p7_12

delta1 = (10 ** (SPEC.rp_db / 20) - 1) / (10 ** (SPEC.rp_db / 20) + 1)
print(f"δ1 = {delta1:.3e}  →  {-20 * np.log10(delta1):.1f} dB\n")

print("Janela     A_s tabela   M    Rp medido [dB]   As medido [dB]   resultado")
for c in p7_12.evaluate_candidates():
    if c.design is None:
        print(f"{c.window:9s}   {c.table_as_db:5.0f} dB     —         —                —          descartada (A_s < 50 dB)")
    else:
        rp, as_ = c.report.checks[2].measured, c.report.checks[3].measured
        print(f"{c.window:9s}   {c.table_as_db:5.0f} dB   {c.design.M:3d}     {rp:8.5f}         {as_:7.2f}        "
              f"{'ATENDE' if c.accepted else 'não atende (Rp)'}")
'''),
        md(r"""
A Hamming atende $A_s$ com folga, mas sua ondulação de passagem ($\approx 0.04$ dB) é 10 vezes maior que o permitido. **Aumentar M não resolve.** M estreita a transição, mas não reduz os lóbulos laterais, que são uma propriedade da forma da janela:
"""),
        code(r'''
wg = analysis.frequency_grid(SPEC.edges, analysis.DENSE_GRID_POINTS)
print(" M    Rp Hamming [dB]")
for M in (35, 45, 55, 65, 75, 85, 95):
    rp, _ = analysis.measure_rp_as(design_fixed_window(SPEC, "hamming", M=M).h,
                                   SPEC.passbands, SPEC.stopbands, wg)
    print(f"{M:3d}      {rp:.4f}")
'''),
        md(r"""
**Escolha: janela de Blackman** ($A_s$ nominal de 74 dB, coerente com a estimativa de ~73 dB acima).

$$M = \left\lceil \frac{11\pi}{0.2\pi} \right\rceil + 1 = 55 + 1 = 56 \;\xrightarrow{\text{M ímpar}}\; 57, \qquad \alpha = 28$$

> O DTA escreve "$\lceil 11\pi/0.2\pi \rceil + 1 = 57$". A conta dá 56, e o 57 vem da regra de paridade do passa-altas. O resultado final é o mesmo, mas o motivo é outro.

$$h(n) = \left[\delta(n-\alpha) - \frac{\sin[\omega_c(n-\alpha)]}{\pi(n-\alpha)}\right] w_{Blackman}(n)$$
"""),
        code(r'''
d = p7_12.design()
print(f"Janela: {d.window};  M = {d.M} (Tabela 7.1: {d.M_table});  α = {d.alpha:g};  "
      f"Tipo {analysis.linear_phase_type(d.h)}")
'''),
        md(r"""
## 3. Resultados

O gráfico pedido é o terceiro painel, **zoom na passagem** em escala de milésimos de dB. Os dois primeiros ($h(n)$ e $|H|$ completa) dão contexto.
"""),
        code(r'''
fig = p7_12.plot(d)
plt.show()
'''),
        md(r"""
## 4. Validação contra a especificação
"""),
        code(r'''
print(p7_12.validate(d))

from scipy.signal import firwin
ref = firwin(d.M, 0.5, window="blackman", pass_zero=False, scale=False)
print("\nmax |h − h_firwin| =", np.max(np.abs(d.h - ref)))
'''),
        md(r"""
### 4.1 Observação: comprimento mínimo com Blackman

O enunciado não pede comprimento mínimo. Mesmo assim, a varredura mostra que $M = 55$ também atende, com folga de só ≈ $7\times10^{-5}$ dB em $R_p$:
"""),
        code(r'''
print(" M    Rp [dB]    As [dB]   atende?")
for M in (51, 53, 55, 57, 59):
    dm = design_fixed_window(SPEC, "blackman", M=M)
    rp, as_ = analysis.measure_rp_as(dm.h, SPEC.passbands, SPEC.stopbands, wg)
    print(f"{M:3d}   {rp:.5f}   {as_:6.2f}    {'sim' if rp <= SPEC.rp_db and as_ >= SPEC.as_db else 'não'}")
'''),
        md(r"""
## 5. Conclusão

- A especificação $R_p = 0.004$ dB, e não $A_s$, é o que dita a janela: corresponde a $\delta_1 \approx 2.3\times10^{-4}$ (~73 dB). Só a Blackman (74 dB) chega lá.
- A Hamming atende $A_s$, mas tem $R_p \approx 0.03$–$0.04$ dB para qualquer $M$. Aumentar o comprimento não reduz a ondulação.
- Projeto final: Blackman, $M = 57$ (Tipo 1, $\alpha = 28$), com $R_p \approx 0.00296$ dB $\le 0.004$ dB e $A_s \approx 75.25$ dB $\ge 50$ dB. Os valores medidos coincidem com os do DTA (0.002942 dB e 75.25 dB).
- $M = 55$ também atenderia, mas com margem desprezível em $R_p$. $M = 57$ segue a regra da tabela.
"""),
    ]
