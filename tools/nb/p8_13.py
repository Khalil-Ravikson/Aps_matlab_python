"""Notebook p8_13."""

from .common import SETUP, code, md


def cells():
    return [
        md(r"""
# P8.13 — Protótipo analógico Butterworth passa-baixas em forma de cascata

**Referência:** Ingle & Proakis, *DSP Using MATLAB*, 3ª ed., p. 465 · DTA §2.2 (item 10), §3.2.A e §4 (Resolução P8.13).

> *Design an analog Butterworth lowpass filter that has a 0.5 dB or better ripple at 10 kHz and at least 45 dB of attenuation at 20 kHz. Determine the system function in a cascade form. Plot the magnitude response, the log-magnitude response in dB, the group-delay, and the impulse response of the filter.*

| Parâmetro | Valor |
|---|---|
| Borda de passagem | $F_p = 10$ kHz ($\Omega_p = 2\pi \cdot 10^4$ rad/s), $R_p = 0.5$ dB |
| Borda de rejeição | $F_s = 20$ kHz ($\Omega_s = 2\pi \cdot 2\cdot10^4$ rad/s), $A_s = 45$ dB |
| Forma pedida | cascata |
"""),
        code(SETUP),
        md(r"""
## 1. Método

As fórmulas são as do P8.12, com as frequências convertidas por $\Omega = 2\pi F$. A ordem só depende da razão $\Omega_p/\Omega_s = 1/2$:

$$N = \left\lceil \frac{\log_{10}\left[(10^{0.05}-1)/(10^{4.5}-1)\right]}{2\log_{10}(1/2)} \right\rceil = \lceil 8.99 \rceil = 9$$

$$\Omega_c = \frac{\Omega_p}{(10^{0.05}-1)^{1/18}} \approx 70621\ \text{rad/s} \quad (F_c \approx 11.24\ \text{kHz})$$

**Forma em cascata.** Os 9 polos estão sobre o círculo de raio $\Omega_c$. Como $N$ é ímpar, um deles é real ($s = -\Omega_c$) e os outros 8 formam 4 pares conjugados. Se $\theta_k$ é o ângulo de um par em relação ao eixo real negativo, o par contribui com $s^2 + 2\Omega_c\cos\theta_k\,s + \Omega_c^2$:

$$H_a(s) = \frac{\Omega_c}{s + \Omega_c}\prod_{k=1}^{4}\frac{\Omega_c^2}{s^2 + 2\Omega_c\cos\theta_k\,s + \Omega_c^2}, \qquad \theta_k = \frac{k\pi}{9} = 20°,\ 40°,\ 60°,\ 80°$$

Cada seção tem ganho DC unitário, e o produto dá $H_a(0) = 1$.

> **Errata do DTA.** O DTA escreve $\theta_k = (2k-1)\pi/18$ (10°, 30°, 50°, 70°), a fórmula para $N$ **par**. Com $N = 9$, os polos estão em $\pi/2 + (2k+1)\pi/18$, $k = 0,\ldots,8$, ou seja, em 100°, 120°, …, 260°. A partir do eixo real negativo (180°), isso dá 0° (polo real), ±20°, ±40°, ±60° e ±80°.
"""),
        code(r'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display

from dsp_av2.iir import analysis, forms
from dsp_av2.specs import P8_13 as SPEC
from exercises import p8_13

proto = p8_13.design()
wc = proto.params["Omega_c"]
print(f"N = {proto.N}  (antes do teto: {proto.params['N_real']:.4f})")
print(f"Ωc = {wc:.4f} rad/s  (Fc = {wc / (2 * np.pi):.2f} Hz)\n")
print("Polos (ângulo medido a partir do eixo real negativo):")
for p in sorted(proto.p, key=lambda q: q.imag):
    print(f"  {p.real:12.2f} {p.imag:+12.2f}j    {np.degrees(np.pi - abs(np.angle(p))):6.2f}°")
'''),
        md(r"""
## 2. Função de sistema em cascata
"""),
        code(r'''
c = p8_13.cascade_form(proto)
display(Math(r"H_a(s) = " + forms.cascade_to_latex(c)))
print("θ_k dos biquads (graus):", np.round(np.sort(p8_13.section_angles(proto, c)), 6))
'''),
        md(r"""
As seções estão ordenadas da mais amortecida (polo real, depois $\theta = 20°$) para a menos amortecida ($\theta = 80°$, o par mais próximo do eixo $j\Omega$). Na implementação física de uma cascata essa ordem importa por causa da faixa dinâmica, mas a função de sistema é a mesma em qualquer ordem.

**Por que cascata?** A forma racional deste filtro teria $a_0 = \Omega_c^9 \approx 4.4\times10^{43}$. Coeficientes com essa faixa dinâmica são numericamente frágeis. Em biquads, todos os coeficientes ficam entre $10^4$ e $5\times10^9$.

## 3. Gráficos pedidos

Magnitude, log-magnitude, atraso de grupo e resposta ao impulso. O eixo de frequência está em Hz, como no enunciado (decisão D9).

O atraso de grupo é calculado **analiticamente** a partir dos polos:

$$\tau(\Omega) = -\frac{d\phi}{d\Omega} = \sum_k \frac{-\sigma_k}{\sigma_k^2 + (\Omega - \omega_k)^2}, \qquad p_k = \sigma_k + j\omega_k$$
"""),
        code(r'''
fig = p8_13.plot(proto)
plt.show()
'''),
        md(r"""
O atraso de grupo tem pico perto da frequência de corte ($\approx 11$ kHz), onde a fase varia mais depressa por causa do par de polos de $\theta = 80°$. A resposta ao impulso oscila nessa mesma frequência (período ≈ 0.1 ms).

## 4. Validação contra a especificação
"""),
        code(r'''
print(p8_13.validate(proto))

# Verificação do atraso de grupo analítico contra a derivada numérica da fase
w = np.linspace(0, 2 * SPEC.ws, 200001)
phase = np.unwrap(np.angle(analysis.frequency_response(proto, w)))
tau_num = -np.gradient(phase, w)
tau = analysis.group_delay(proto, w)
print(f"\nmax |τ_analítico − τ_numérico| / max τ = {np.max(np.abs(tau - tau_num)[5:-5]) / tau.max():.2e}")
'''),
        md(r"""
**Margem.** A atenuação em $F_s$ é de 45.05 dB, só 0.05 dB acima do exigido. É a consequência de $N = 8.99$ estar quase exatamente no inteiro 9: o arredondamento deixa pouca folga. Com $N = 8$ a atenuação seria 39.0 dB, o que confirma que 9 é a ordem mínima.

## 5. Conclusão

- $N = 9$ e $\Omega_c \approx 70621$ rad/s ($F_c \approx 11.24$ kHz), valores que coincidem com o DTA.
- Cascata: uma seção de 1ª ordem $\Omega_c/(s+\Omega_c)$ e quatro biquads com $\theta_k = k\pi/9$ (20°, 40°, 60°, 80°). **O DTA usa a fórmula de $N$ par, $(2k-1)\pi/18$, o que é incorreto para $N = 9$.**
- Perda de exatamente 0.5 dB em 10 kHz (por construção de $\Omega_c$) e atenuação de 45.05 dB em 20 kHz.
- O atraso de grupo analítico coincide com $-d\phi/d\Omega$ numérico. Ele começa em ≈ 0.082 ms (DC) e chega a ≈ 0.15 ms perto de $F_c$.
"""),
    ]
