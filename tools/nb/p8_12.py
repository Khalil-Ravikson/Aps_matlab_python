"""Notebook p8_12."""

from .common import SETUP, code, md


def cells():
    return [
        md(r"""
# P8.12 — Protótipo analógico Butterworth passa-baixas

**Referência:** Ingle & Proakis, *DSP Using MATLAB*, 3ª ed., p. 465 · DTA §2.2 (item 9), §3.2.A e §4 (Resolução P8.12).

> *Design an analog Butterworth lowpass filter that has a 0.25 dB or better ripple at 500 rad/sec and at least 50 dB of attenuation at 2000 rad/sec. Determine the system function in a rational function form. Plot the magnitude response, the log-magnitude response in dB, the phase response, and the impulse response of the filter.*

| Parâmetro | Valor |
|---|---|
| Borda de passagem | $\Omega_p = 500$ rad/s, $R_p = 0.25$ dB |
| Borda de rejeição | $\Omega_s = 2000$ rad/s, $A_s = 50$ dB |
| Forma pedida | racional, $H_a(s) = B(s)/A(s)$ |
"""),
        code(SETUP),
        md(r"""
## 1. Método

O Butterworth é maximamente plano: $|H_a(j\Omega)|^2 = \dfrac{1}{1 + (\Omega/\Omega_c)^{2N}}$.

**Ordem.** Aplicando as duas restrições, $-10\log_{10}|H_a(j\Omega_p)|^2 = R_p$ e $-10\log_{10}|H_a(j\Omega_s)|^2 = A_s$, e eliminando $\Omega_c$:

$$N = \left\lceil \frac{\log_{10}\left[(10^{R_p/10}-1)/(10^{A_s/10}-1)\right]}{2\log_{10}(\Omega_p/\Omega_s)} \right\rceil = \left\lceil \frac{\log_{10}\left[(10^{0.025}-1)/(10^{5}-1)\right]}{2\log_{10}(500/2000)} \right\rceil = \lceil 5.17 \rceil = 6$$

**Frequência de corte.** O livro escolhe $\Omega_c$ para atender $R_p$ **exatamente** em $\Omega_p$. A folga gerada pelo arredondamento de $N$ vai toda para a rejeição:

$$\Omega_c = \frac{\Omega_p}{(10^{R_p/10}-1)^{1/(2N)}} = \frac{500}{(10^{0.025}-1)^{1/12}} \approx 632.77\ \text{rad/s}$$

**Polos** (semiplano esquerdo, igualmente espaçados num círculo de raio $\Omega_c$):

$$p_k = \Omega_c\, e^{\,j\left[\frac{\pi}{2} + \frac{(2k+1)\pi}{2N}\right]}, \quad k = 0,\dots,5 \qquad\Rightarrow\qquad H_a(s) = \frac{\Omega_c^{6}}{\prod_{k=0}^{5}(s - p_k)}$$
"""),
        code(r'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display

from dsp_av2.iir import analysis, forms
from dsp_av2.specs import P8_12 as SPEC
from exercises import p8_12

proto = p8_12.design()
print(f"N = {proto.N}  (valor antes do teto: {proto.params['N_real']:.4f})")
print(f"Ωc = {proto.params['Omega_c']:.4f} rad/s")
print("\nPolos (rad/s):")
for p in proto.p:
    print(f"  {p.real:10.4f} {p.imag:+10.4f}j   |p| = {abs(p):.4f}   ∠p = {np.degrees(np.angle(p)):7.2f}°")
'''),
        md(r"""
## 2. Função de sistema na forma racional
"""),
        code(r'''
b, a = p8_12.rational_form(proto)
display(Math(r"H_a(s) = \dfrac{" + forms.poly_to_latex(b) + "}{" + forms.poly_to_latex(a) + "}"))
'''),
        md(r"""
O numerador é igual ao termo independente do denominador, $B = a_0 = \Omega_c^6$. Então $H_a(0) = 1$, o ganho DC unitário do Butterworth. Todos os coeficientes de $A(s)$ são positivos, como exige um polinômio de Hurwitz (sistema estável).

## 3. Gráficos pedidos

Magnitude, log-magnitude em dB, fase e resposta ao impulso. A fase é a principal, em $(-\pi, \pi]$, como na `freqs_m` do livro. O salto em ~470 rad/s é só a passagem por $-\pi$, não uma descontinuidade física. A resposta ao impulso foi simulada até a envoltória mais lenta cair 60 dB.
"""),
        code(r'''
fig = p8_12.plot(proto)
plt.show()
'''),
        md(r"""
## 4. Validação contra a especificação

Três verificações: perda máxima em $[0, \Omega_p]$, atenuação mínima em $[\Omega_s, 100\,\Omega_s]$ e **minimalidade** (o mesmo projeto com $N-1 = 5$ não alcança 50 dB em $\Omega_s$).
"""),
        code(r'''
print(p8_12.validate(proto))
'''),
        md(r"""
**Verificações independentes.** Os polos e o ganho conferem com `scipy.signal.butter(..., analog=True)`. A resposta ao impulso simulada (`scipy.signal.impulse`) confere com a transformada inversa de Laplace por resíduos, $h(t) = \sum_k r_k e^{p_k t}$:
"""),
        code(r'''
from scipy import signal
z_sp, p_sp, k_sp = signal.butter(proto.N, proto.params["Omega_c"], analog=True, output="zpk")
print("max |p − p_scipy| =", np.max(np.abs(np.sort_complex(proto.p) - np.sort_complex(p_sp))))
print("k / k_scipy − 1   =", proto.k / k_sp - 1)

t = np.linspace(0, analysis.impulse_time_span(proto), 2001)
h_sim, h_res = analysis.impulse_response(proto, t), analysis.impulse_response_residues(proto, t)
print("max |h_sim − h_res| / max |h| =", np.max(np.abs(h_sim - h_res)) / np.max(np.abs(h_res)))
'''),
        md(r"""
## 5. Conclusão

- $N = 6$ e $\Omega_c \approx 632.77$ rad/s. Os coeficientes de $B(s)/A(s)$ coincidem com os do DTA.
- Em $\Omega_p$ a perda é exatamente 0.25 dB, por construção de $\Omega_c$. Em $\Omega_s$ a atenuação é ≈ 59.97 dB: a folga de ≈ 10 dB vem do arredondamento de $N = 5.17$ para 6.
- Com $N = 5$ a atenuação em $\Omega_s$ seria só ≈ 47.9 dB < 50 dB, o que confirma que 6 é a ordem mínima.
- A resposta ao impulso oscila e decai, com pico perto de 7 ms e duração efetiva de ~40 ms. Isso reflete o par de polos mais próximo do eixo $j\Omega$, a 15° dele ($\angle p = \pm 105°$).
"""),
    ]
