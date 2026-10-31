"""Notebook p8_15."""

from .common import SETUP, code, md


def cells():
    return [
        md(r"""
# P8.15 — Protótipo analógico Chebyshev-I passa-baixas em forma de cascata

**Referência:** Ingle & Proakis, *DSP Using MATLAB*, 3ª ed., pp. 465–466 · DTA §2.2 (item 12), §3.2.B e §4 (Resolução P8.15).

> *Design a lowpass analog Chebyshev-I filter with the following characteristics: A passband ripple of 0.5 dB, passband cutoff frequency of 4 kHz, and stopband attenuation of 45 dB or greater beyond 20 kHz. Determine the system function in a cascade form. Plot the magnitude response, the log-magnitude response in dB, the phase response, and the impulse response of the filter.*

| Parâmetro | Valor |
|---|---|
| Passagem | $F_p = 4$ kHz ($\Omega_p = 2\pi\cdot4\cdot10^3$ rad/s), $R_p = 0.5$ dB |
| Rejeição | $F_s = 20$ kHz ($\Omega_s = 2\pi\cdot2\cdot10^4$ rad/s), $A_s = 45$ dB |
| Forma pedida | cascata |
"""),
        code(SETUP),
        md(r"""
## 1. Método

Mesmas fórmulas do P8.14:

$$\epsilon = \sqrt{10^{0.05}-1} = 0.349311, \qquad g = \frac{\sqrt{10^{4.5}-1}}{\epsilon} = 509.08, \qquad \Omega_r = \frac{20}{4} = 5$$

$$N = \left\lceil \frac{\cosh^{-1} 509.08}{\cosh^{-1} 5} \right\rceil = \left\lceil \frac{6.926}{2.292} \right\rceil = \lceil 3.02 \rceil = 4$$

**Cascata.** Com $N = 4$ (par) não há polo real. Os 4 polos formam 2 pares conjugados, e cada par $p = \sigma \pm j\omega$ gera um biquad normalizado para ganho DC unitário:

$$H_k(s) = \frac{|p|^2}{s^2 - 2\sigma s + |p|^2}$$

O ganho global vem de $N$ par, que põe o DC num vale da ondulação: $G = 1/\sqrt{1+\epsilon^2} = 0.94406$.
"""),
        code(r'''
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math, display

from dsp_av2.iir import analysis, forms
from dsp_av2.specs import P8_15 as SPEC
from exercises import p8_15

proto = p8_15.design()
print(f"ε = {proto.params['epsilon']:.6f};  N = {proto.N}  (antes do teto: {proto.params['N_real']:.4f})\n")
print("Polos (rad/s):")
for p in sorted(proto.p, key=lambda q: q.imag):
    print(f"  {p.real:10.2f} {p.imag:+10.2f}j   (|p|/2π = {abs(p) / (2 * np.pi):7.1f} Hz)")
'''),
        md(r"""
## 2. Função de sistema em cascata
"""),
        code(r'''
c = p8_15.cascade_form(proto)
display(Math(r"H_a(s) = " + forms.cascade_to_latex(c)))
'''),
        md(r"""
Os biquads e o ganho coincidem com o DTA: $s^2 + 8814.21s + 6.7178\times10^8$, $s^2 + 21279.38s + 2.2513\times10^8$ e $G = 0.94406$.

## 3. Gráficos pedidos

Magnitude, log-magnitude, fase (principal, em unidades de π) e resposta ao impulso, com frequência em Hz (decisão D9).
"""),
        code(r'''
fig = p8_15.plot(proto)
plt.show()
'''),
        md(r"""
**Leitura dos gráficos:**

- **Magnitude:** com $N = 4$ há $N + 1 = 5$ extremos em $[0, 4\text{ kHz}]$, começando no vale 0.944 em DC.
- **Fase:** a fase total vai de 0 a $-N\pi/2 = -2\pi$. O traço principal salta de $-\pi$ para $+\pi$ perto de 3.8 kHz e depois tende a 0 (≡ $-2\pi$).
- **Resposta ao impulso:** oscilação amortecida dominada pelo par $-4407 \pm j25541$ (≈ 4.1 kHz, período ≈ 0.25 ms), o par de menor amortecimento.

## 4. Validação contra a especificação
"""),
        code(r'''
print(p8_15.validate(proto))
'''),
        md(r"""
**Folga.** A atenuação em 20 kHz é de 64.5 dB, quase 20 dB acima do exigido. Como $N_{real} = 3.02$ fica logo acima de 3, o arredondamento para 4 gera muita sobra. Com $N = 3$ faltaria pouco (44.58 dB < 45 dB), mas faltaria.

## 5. Conclusão

- $\epsilon = 0.349311$ e $N = 4$ ($\lceil 3.02 \rceil$).
- Cascata: $H_a(s) = 0.94406 \cdot \frac{6.7178\times10^8}{s^2 + 8814.21s + 6.7178\times10^8} \cdot \frac{2.2513\times10^8}{s^2 + 21279.38s + 2.2513\times10^8}$, idêntica ao DTA.
- Perda de 0.5 dB em 4 kHz (equiripple) e atenuação de 64.5 dB em 20 kHz. Com $N = 3$ ficaria 0.42 dB abaixo do exigido, o que confirma $N = 4$ como mínimo.
"""),
    ]
