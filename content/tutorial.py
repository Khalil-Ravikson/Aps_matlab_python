r"""Guia de estudo: do impulso ao projeto de filtros (para quem sabe até resposta ao impulso).

Texto Markdown com marcadores [[...]]; os valores vêm de :func:`context`, que usa o
núcleo e os contextos validados de P7.9 e P8.12. Usado por report/build_guide.py.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dsp_av2.fir import analysis as fa

from . import registry


@lru_cache(maxsize=1)
def context() -> dict:
    h = np.array([0.5, 0.5])
    H = np.abs(fa.frequency_response(h, np.array([0.0, np.pi / 2, np.pi])))
    c9 = {f"f9_{k}": v for k, v in registry.load("p7_09").context().items() if not isinstance(v, str)}
    c12 = {f"i12_{k}": v for k, v in registry.load("p8_12").context().items() if not isinstance(v, str)}
    c9["f9_Mm1"] = c9["f9_M"] - 1
    return {"mm0": H[0], "mm1": H[1], "mm1_db": 20 * np.log10(H[1]), "mm2": H[2]} | c9 | c12


SECTIONS = (
    ("1. Ponto de partida: você já sabe a resposta ao impulso", r"""
Um sistema linear e invariante no tempo (LIT) fica **completamente descrito** pela sua resposta ao impulso $h(n)$. A saída para qualquer entrada é a **convolução**:

$$y(n) = \sum_{k} h(k)\,x(n-k).$$

Leitura prática: cada amostra de saída é uma **média ponderada** das amostras de entrada, e os pesos são os valores de $h(n)$.

**Exemplo que vamos usar o tempo todo: média móvel de 2 pontos.**

$$h(n) = \{0.5,\ 0.5\} \quad\Rightarrow\quad y(n) = 0.5\,x(n) + 0.5\,x(n-1).$$

Intuição: a média suaviza o sinal. Um sinal que varia devagar passa quase inalterado. Um sinal que alterna rápido ($+1, -1, +1, \ldots$) é **cancelado**, porque a média de $+1$ e $-1$ é zero. Ou seja, esse sistema já é um **filtro**: deixa passar o lento e bloqueia o rápido.
"""),
    ("2. De h(n) para a resposta em frequência", r"""
A pergunta que todo projeto de filtro responde é: **o que o sistema faz com cada frequência?**

Se a entrada for uma senoide, $x(n) = \cos(\omega_0 n)$, a saída de um sistema LIT é **a mesma senoide**, só multiplicada por um ganho e deslocada por uma fase:

$$y(n) = |H(e^{j\omega_0})|\,\cos\big(\omega_0 n + \angle H(e^{j\omega_0})\big).$$

O ganho e a fase vêm da **resposta em frequência**, que é a transformada de Fourier de tempo discreto (DTFT) de $h(n)$:

$$H(e^{j\omega}) = \sum_{n} h(n)\,e^{-j\omega n}.$$

**Sobre $\omega$ (frequência digital):** é medida em radianos por amostra e vai de $0$ a $\pi$. $\omega = 0$ é o sinal constante, e $\omega = \pi$ é o sinal que alterna a cada amostra, a frequência mais alta possível (metade da taxa de amostragem). Nos enunciados, "0.35π" quer dizer 35% dessa frequência máxima.

**Aplicando na média móvel:**

$$H(e^{j\omega}) = 0.5 + 0.5\,e^{-j\omega} = e^{-j\omega/2}\cos(\omega/2), \qquad |H(e^{j\omega})| = |\cos(\omega/2)|.$$

| $\omega$ | significado | ganho $\lvert H\rvert$ |
|---|---|---|
| $0$ | sinal constante | [[mm0:.3f]] (passa inteiro) |
| $\pi/2$ | frequência média | [[mm1:.3f]] |
| $\pi$ | alternância máxima | [[mm2:.3f]] (bloqueado) |

É exatamente a intuição do item 1, agora em números: a média móvel é um **passa-baixas**.
"""),
    ("3. Falando em decibéis", r"""
Ganhos de filtros variam de 1 até valores muito pequenos (0.001, por exemplo). Por isso usa-se a escala logarítmica:

$$|H|_{dB} = 20\log_{10}|H|.$$

| ganho $\lvert H\rvert$ | em dB | leitura |
|---|---|---|
| 1 | 0 dB | passa inteiro |
| 0.707 | −3 dB | metade da potência |
| 0.5 | −6 dB | metade da amplitude |
| 0.1 | −20 dB | 10× menor |
| 0.01 | −40 dB | 100× menor |
| 0.001 | −60 dB | 1000× menor |

Regra de bolso: **cada −20 dB divide a amplitude por 10**. Na média móvel, $|H(\pi/2)| = [[mm1:.3f]]$, ou seja, $[[mm1_db:.2f]]$ dB.
"""),
    ("4. Como um enunciado especifica um filtro", r"""
Nenhum filtro real tem degrau perfeito. Por isso o enunciado define **faixas** com **tolerâncias**:

- **Faixa de passagem:** o ganho deve ficar perto de 1. O desvio máximo é dado por $R_p$ (em dB) ou por $\delta_1$ (linear): $1-\delta_1 \le |H| \le 1+\delta_1$.
- **Faixa de rejeição:** o ganho deve ficar perto de 0. O mínimo de atenuação é $A_s$ (em dB) ou $\delta_2$ (linear): $|H| \le \delta_2$.
- **Faixa de transição:** entre as duas, sem exigência. A largura dela, $\Delta\omega$, é o que mais pesa no **custo** do filtro.

Conversões que aparecem em quase toda questão:

$$R_p = -20\log_{10}\frac{1-\delta_1}{1+\delta_1}, \qquad A_s = -20\log_{10}\delta_2.$$

Exemplo: $\delta_2 = 0.005$ dá $A_s = -20\log_{10}0.005 \approx 46$ dB.

Os quatro tipos básicos são: **passa-baixas**, **passa-altas**, **passa-faixa** e **rejeita-faixa**. Os nomes dizem que faixa fica.
"""),
    ("5. O filtro ideal e por que ele não existe", r"""
O passa-baixas **ideal** tem ganho 1 até um corte $\omega_c$ e 0 depois. A resposta ao impulso dele (a DTFT inversa do degrau) é:

$$h_d(n) = \frac{\sin(\omega_c n)}{\pi n}.$$

Ela tem dois problemas:

1. é **infinita**, porque não acaba nunca;
2. é **não causal**, porque tem valores em $n < 0$.

A solução é **atrasar** a resposta em $\alpha$ amostras e **cortar** num comprimento $M$:

$$h_d(n) = \frac{\sin[\omega_c(n-\alpha)]}{\pi(n-\alpha)}, \qquad 0 \le n \le M-1,\quad \alpha = \frac{M-1}{2}.$$

Os outros tipos saem do passa-baixas por soma e subtração:

| Filtro | Resposta ideal |
|---|---|
| Passa-altas | $\delta(n-\alpha) - h_{LP}(n)$ (tudo menos o passa-baixas) |
| Passa-faixa | $h_{LP,2}(n) - h_{LP,1}(n)$ (diferença de dois passa-baixas) |
| Rejeita-faixa | $\delta(n-\alpha) - h_{PF}(n)$ (tudo menos o passa-faixa) |
"""),
    ("6. Cortar estraga: o papel da janela", r"""
Cortar $h_d(n)$ de repente equivale a multiplicá-la por uma janela **retangular**. Isso cria **ondulações** perto das bordas (o fenômeno de Gibbs). Com a janela retangular, a rejeição fica em torno de apenas 21 dB (Tabela 7.1 do livro), por maior que seja $M$.

A saída é cortar **suavemente**: multiplicar por uma janela $w(n)$ que vai a zero devagar nas pontas.

$$h(n) = h_d(n)\,w(n).$$

O preço é uma transição mais larga. Cada janela tem o seu equilíbrio (Tabela 7.1, conforme a pesquisa de referência):

| Janela | $w(n)$ | Transição $\Delta\omega$ | $A_s$ que alcança |
|---|---|---|---|
| Hann | $0.5 - 0.5\cos\frac{2\pi n}{M-1}$ | $6.2\pi/M$ | ≈ 44 dB |
| Hamming | $0.54 - 0.46\cos\frac{2\pi n}{M-1}$ | $6.6\pi/M$ | ≈ 53 dB |
| Blackman | $0.42 - 0.5\cos\frac{2\pi n}{M-1} + 0.08\cos\frac{4\pi n}{M-1}$ | $11\pi/M$ | ≈ 74 dB |

As duas regras de ouro:

- **a janela escolhe a atenuação** ($A_s$ não melhora aumentando $M$);
- **o comprimento escolhe a transição** ($\Delta\omega \approx k\pi/M$, logo $M = \lceil k\pi/\Delta\omega\rceil + 1$).

A **janela de Kaiser** junta as duas coisas num parâmetro ajustável $\beta$: em vez de escolher na tabela, calcula-se $\beta$ a partir do $A_s$ desejado.
"""),
    ("7. Fase linear: por que h(n) simétrico", r"""
Se $h(n)$ é **simétrico** em torno do centro, $h(n) = h(M-1-n)$, então

$$H(e^{j\omega}) = H_r(\omega)\,e^{-j\omega\alpha}.$$

$H_r(\omega)$ é real (a "amplitude"), e o resto é um **atraso puro de $\alpha$ amostras, igual para todas as frequências**. O filtro não deforma o sinal: só o atrasa. Essa é a grande vantagem dos FIR.

Dois tipos aparecem nas questões:

- **Tipo 1 ($M$ ímpar):** serve para qualquer filtro.
- **Tipo 2 ($M$ par):** tem $H_r(\pi) = 0$ **obrigatoriamente**. Não serve para passa-altas nem rejeita-faixa, que precisam deixar passar $\omega = \pi$.

Na prática: se o filtro precisa de ganho em $\omega = \pi$, **$M$ tem de ser ímpar**.
"""),
    ("8. A receita de uma questão FIR", r"""
1. **Ler as faixas** e anotar as bordas $\omega_p$, $\omega_s$ e as tolerâncias.
2. **Largura de transição:** $\Delta\omega$ = a menor distância entre uma borda de passagem e a de rejeição vizinha.
3. **Escolher a janela** pela atenuação: a primeira da tabela com $A_s$ suficiente (ou calcular $\beta$, se for Kaiser).
4. **Calcular $M$:** $M = \lceil k\pi/\Delta\omega \rceil + 1$; acertar a paridade se precisar de ganho em $\pi$.
5. **Cortes:** $\omega_c = (\omega_p + \omega_s)/2$, no meio de cada transição.
6. **Montar** $h(n) = h_d(n)\,w(n)$.
7. **Medir:** calcular $|H(e^{j\omega})|$ e conferir $R_p$ e $A_s$. Se a questão pedir o $M$ **mínimo**, testar os valores menores.
"""),
    ("9. Exemplo resolvido do zero: P7.9", r"""
**Enunciado (resumo):** passa-faixa com janela de Hann. Rejeição até $0.2\pi$, passagem de $0.35\pi$ a $0.55\pi$, rejeição a partir de $0.75\pi$. $R_p = 0.25$ dB e $A_s = 40$ dB.

**Passo 1, faixas:** rejeição $[0, 0.2\pi]$, passagem $[0.35\pi, 0.55\pi]$, rejeição $[0.75\pi, \pi]$.

**Passo 2, transição:** as duas transições medem $[[f9_t1:g]]\pi$ e $[[f9_t2:g]]\pi$. A janela faz a mesma transição nos dois lados, então vale a menor:

$$\Delta\omega = [[f9_dw:g]]\pi.$$

**Passo 3, janela:** é imposta (Hann). Conferindo: Hann alcança ≈ 44 dB ≥ 40 dB. Atende.

**Passo 4, comprimento:**

$$M = \left\lceil \frac{6.2\pi}{[[f9_dw:g]]\pi} \right\rceil + 1 = \lceil [[f9_ratio:.2f]] \rceil + 1 = [[f9_M]].$$

É ímpar (Tipo 1), com atraso $\alpha = [[f9_alpha:g]]$.

**Passo 5, cortes:** $\omega_{c1} = \frac{0.2\pi + 0.35\pi}{2} = [[f9_wc1:g]]\pi$ e $\omega_{c2} = \frac{0.55\pi + 0.75\pi}{2} = [[f9_wc2:g]]\pi$.

**Passo 6, montar:**

$$h(n) = \left[\frac{\sin([[f9_wc2:g]]\pi(n-[[f9_alpha:g]])) - \sin([[f9_wc1:g]]\pi(n-[[f9_alpha:g]]))}{\pi(n-[[f9_alpha:g]])}\right]\cdot\left[0.5 - 0.5\cos\frac{2\pi n}{[[f9_Mm1]]}\right], \quad 0 \le n \le [[f9_Mm1]].$$

**Passo 7, medir:** $R_p = [[f9_rp:.3f]]$ dB ≤ 0.25 e $A_s = [[f9_as_:.2f]]$ dB ≥ 40. **Atende.**

**Resposta:** Hann, $M = [[f9_M]]$, cortes em $[[f9_wc1:g]]\pi$ e $[[f9_wc2:g]]\pi$.
"""),
    ("10. E os IIR? A ideia em uma página", r"""
Um filtro **IIR** tem realimentação: a saída depende também de saídas anteriores. A resposta ao impulso é infinita, mas o filtro tem poucos coeficientes.

No Capítulo 8, o projeto começa por um **protótipo analógico** $H_a(s)$, descrito pela magnitude ao quadrado:

- **Butterworth** (o mais plano possível, sem ondulação): $|H_a(j\Omega)|^2 = \dfrac{1}{1 + (\Omega/\Omega_c)^{2N}}$
- **Chebyshev-I** (ondula na passagem, cai mais rápido): $|H_a(j\Omega)|^2 = \dfrac{1}{1 + \epsilon^2 T_N^2(\Omega/\Omega_p)}$

Aqui a frequência $\Omega$ está em **rad/s** (analógica). Se o enunciado der hertz, converta: $\Omega = 2\pi F$.

O que se calcula é a **ordem $N$**, que é o número de polos. Quanto maior $N$, mais abrupta a queda.

**Receita Butterworth:**

1. Ordem: $N = \left\lceil \dfrac{\log_{10}\left[(10^{R_p/10}-1)/(10^{A_s/10}-1)\right]}{2\log_{10}(\Omega_p/\Omega_s)} \right\rceil$
2. Corte: $\Omega_c = \Omega_p/(10^{R_p/10}-1)^{1/(2N)}$
3. Polos: $N$ pontos num semicírculo de raio $\Omega_c$, no lado esquerdo do plano $s$ (o que garante estabilidade)
4. Função de sistema: $H_a(s) = \Omega_c^N / \prod_k (s - p_k)$

**Exemplo rápido, P8.12** ($\Omega_p = 500$ rad/s, $R_p = 0.25$ dB, $\Omega_s = 2000$ rad/s, $A_s = 50$ dB):

$$N = \lceil [[i12_N_real:.2f]] \rceil = [[i12_N]], \qquad \Omega_c = \frac{500}{([[i12_num:.5f]])^{1/[[i12_N2]]}} = [[i12_Omega_c:.2f]]\ \text{rad/s}.$$

Verificação: em $\Omega_s$ a atenuação é $[[i12_att_ws:.2f]]$ dB ≥ 50; com $N = [[i12_Nm1]]$ seriam só $[[i12_att_lower:.2f]]$ dB. Então $N = [[i12_N]]$ é o mínimo.
"""),
    ("11. Checklist para qualquer questão", r"""
1. Que **tipo** de filtro? (passa-baixas, passa-altas, passa-faixa, rejeita-faixa)
2. **FIR ou IIR?** Qual método o enunciado impõe?
3. Frequências em **π** (digital), **rad/s** ou **Hz** (analógica)? Converter se precisar.
4. Tolerâncias em **dB** ou **lineares**? Converter com as fórmulas do item 4.
5. FIR: **$\Delta\omega$**, janela, **$M$** e **paridade**. IIR: **ordem $N$** e corte.
6. Pede o **mínimo**? Então **verifique** os valores menores. A fórmula é só uma estimativa.
7. Conferir a resposta **medindo** $R_p$ e $A_s$ nas bordas.
"""),
    ("12. Erros que derrubam a nota", r"""
- **Usar a transição maior** quando há duas diferentes: vale sempre a **menor**.
- **Esquecer a paridade** em passa-altas e rejeita-faixa: $M$ par zera o ganho em $\omega = \pi$.
- **Forçar $M$ ímpar sem necessidade** em passa-faixa: pode deixar de ser o mínimo (P7.14).
- **Confiar na fórmula de Kaiser para o mínimo:** no P7.13 a estimativa ($M = 55$) **não atende**; o mínimo é 57.
- **Esquecer de converter Hz em rad/s** nos IIR.
- **Arredondar a ordem para baixo:** $N$ sempre para **cima**.
- **Medir numa grade que pula as bordas:** o pior ponto costuma estar justamente na borda (P7.10).
"""),
)
