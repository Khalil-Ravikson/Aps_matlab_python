# DOCUMENTO DE TRANSFERÊNCIA ACADÊMICA E ESPECIFICAÇÃO DE IMPLEMENTAÇÃO EM PYTHON

**Avaliação Universitária**: Engenharia da Computação – Processamento Digital de Sinais (PDS / DSP)  
**Referência Bibliográfica Principal**: INGLE, Vinay K.; PROAKIS, John G. *Digital Signal Processing Using MATLAB*. 3rd ed. Stamford: Cengage Learning, 2012.  
**Escopo da Implementação**: 12 Exercícios Práticos dos Capítulos 7 (Filtros FIR) e 8 (Filtros IIR Analógicos Protótipos).

---

## 1. IDENTIFICAÇÃO DA AVALIAÇÃO E REFERÊNCIA BIBLIOGRÁFICA

- **Instituição/Curso**: Engenharia da Computação — Disciplina de Processamento Digital de Sinais.
- **Objetivo do Documento**: Servir como especificação técnica matemática e computacional para o Agente de IA encarregado da implementação automatizada dos scripts Python (SciPy, NumPy, Matplotlib).
- **Livro-Texto**: *Digital Signal Processing Using MATLAB – 3rd Edition* (Vinay K. Ingle & John G. Proakis).
- **Mapeamento de Conteúdo**:
  - Capítulo 7 (*FIR Filter Design*): Páginas 321–380. Exercícios: P7.9, P7.10, P7.11, P7.12, P7.13, P7.14, P7.15, P7.16.
  - Capítulo 8 (*IIR Filter Design*): Páginas 385–466. Exercícios: P8.12, P8.13, P8.14, P8.15.

---

## 2. INVENTÁRIO COMPLETO DOS 12 EXERCÍCIOS

### 2.1. Capítulo 7 – Projeto de Filtros FIR

1. **Exercício P7.9 (Página 378)**
   - *Enunciado*: "Design a linear-phase bandpass filter using the Hann window design technique. The specifications are: lower stopband edge: $0.2\pi$, $A_s = 40\text{ dB}$; upper stopband edge: $0.75\pi$; lower passband edge: $0.35\pi$, $R_p = 0.25\text{ dB}$; upper passband edge: $0.55\pi$. Plot the impulse response and the magnitude response (in dB) of the designed filter. Do not use the `fir1` function."
   - *Tipo*: Passa-Faixa (Bandpass).
   - *Janela*: Hann.

2. **Exercício P7.10 (Páginas 378–379)**
   - *Enunciado*: "Design a bandstop filter using the Hamming window design technique. The specifications are: lower stopband edge: $0.4\pi$, $A_s = 50\text{ dB}$; upper stopband edge: $0.6\pi$; lower passband edge: $0.3\pi$, $R_p = 0.2\text{ dB}$; upper passband edge: $0.7\pi$. Plot the impulse response and the magnitude response (in dB) of the designed filter. Do not use the `fir1` function."
   - *Tipo*: Rejeita-Faixa (Bandstop).
   - *Janela*: Hamming.

3. **Exercício P7.11 (Página 379)**
   - *Enunciado*: "Design a bandpass filter using the Hamming window design technique. The specifications are: lower stopband edge: $0.3\pi$, $A_s = 50\text{ dB}$; upper stopband edge: $0.6\pi$; lower passband edge: $0.4\pi$, $R_p = 0.5\text{ dB}$; upper passband edge: $0.5\pi$. Plot the impulse response and the magnitude response (in dB) of the designed filter. Do not use the `fir1` function."
   - *Tipo*: Passa-Faixa (Bandpass).
   - *Janela*: Hamming.

4. **Exercício P7.12 (Página 379)**
   - *Enunciado*: "Design a highpass filter using one of the fixed window functions. The specifications are: stopband edge: $0.4\pi$, $A_s = 50\text{ dB}$; passband edge: $0.6\pi$, $R_p = 0.004\text{ dB}$. Plot the zoomed magnitude response (in dB) of the designed filter in the passband to verify the passband ripple $R_p$. Do not use the `fir1` function."
   - *Tipo*: Passa-Alta (Highpass).
   - *Janela*: Blackman (selecionada para atender $R_p = 0.004\text{ dB}$).

5. **Exercício P7.13 (Página 379)**
   - *Enunciado*: "Using the Kaiser window method, design a linear-phase FIR digital filter that meets the following specifications: $0.975 \le |H(e^{j\omega})| \le 1.025, 0 \le \omega \le 0.25\pi$; $0 \le |H(e^{j\omega})| \le 0.005, 0.35\pi \le \omega \le 0.65\pi$; $0.975 \le |H(e^{j\omega})| \le 1.025, 0.75\pi \le \omega \le \pi$. Determine the minimum length impulse response $h(n)$ of such a filter. Provide a plot containing subplots of the amplitude response and the magnitude response in dB. Do not use the fir1 function."
   - *Tipo*: Rejeita-Faixa (Bandstop).
   - *Janela*: Kaiser ($\delta_1 = 0.025, \delta_2 = 0.005$).

6. **Exercício P7.14 (Páginas 379–380)**
   - *Enunciado*: "We wish to use the Kaiser window method to design a linear-phase FIR digital filter that meets the following specifications: $0 \le |H(e^{j\omega})| \le 0.01, 0 \le \omega \le 0.25\pi$; $0.95 \le |H(e^{j\omega})| \le 1.05, 0.35\pi \le \omega \le 0.65\pi$; $0 \le |H(e^{j\omega})| \le 0.01, 0.75\pi \le \omega \le \pi$. Determine the minimum length impulse response $h(n)$ of such a filter. Provide a plot containing subplots of the amplitude response and the magnitude response in dB. Do not use the fir1 function."
   - *Tipo*: Passa-Faixa (Bandpass).
   - *Janela*: Kaiser ($\delta_1 = 0.05, \delta_2 = 0.01$).

7. **Exercício P7.15 (Página 380)**
   - *Enunciado*: "Design the staircase filter of Example 7.26 using the Kaiser window approach. The specifications are: Band-1: $0 \le \omega \le 0.3\pi$, Ideal gain = 1, $\delta_1 = 0.01$; Band-2: $0.4\pi \le \omega \le 0.7\pi$, Ideal gain = 0.5, $\delta_2 = 0.005$; Band-3: $0.8\pi \le \omega \le \pi$, Ideal gain = 0, $\delta_3 = 0.001$. Compare the filter length of this design with that of Example 7.26. Provide a plot of the magnitude response in dB. Do not use the fir1 function."
   - *Tipo*: Multibanda em Degraus (Staircase Filter).
   - *Janela*: Kaiser ($\delta_3 = 0.001 \implies A_s = 60\text{ dB}$).

8. **Exercício P7.16 (Página 380)**
   - *Enunciado*: "Design a bandpass filter using a fixed window design technique that has the minimum length and that satisfies the following specifications: lower stopband edge = $0.3\pi$, $A_s = 40\text{ dB}$, upper stopband edge = $0.6\pi$; lower passband edge = $0.4\pi$, $R_p = 0.5\text{ dB}$, upper passband edge = $0.5\pi$. Provide a plot of the log-magnitude response in dB and stem plot of the impulse response."
   - *Tipo*: Passa-Faixa de Menor Ordem Mínima.
   - *Janela*: Hann ($A_s = 44\text{ dB} \ge 40\text{ dB}$, $\Delta\omega = 6.2\pi/M \implies M = 63$).

---

### 2.2. Capítulo 8 – Projeto de Filtros IIR Analógicos Protótipos

9. **Exercício P8.12 (Página 465)**
   - *Enunciado*: "Design an analog Butterworth lowpass filter that has a 0.25 dB or better ripple at 500 rad/sec and at least 50 dB of attenuation at 2000 rad/sec. Determine the system function in a rational function form. Plot the magnitude response, the log-magnitude response in dB, the phase response, and the impulse response of the filter."
   - *Tipo*: Protótipo Butterworth Analógico Passa-Baixas.
   - *Formato de Transferência*: Forma Racional $H_a(s) = B(s)/A(s)$.

10. **Exercício P8.13 (Página 465)**
    - *Enunciado*: "Design an analog Butterworth lowpass filter that has a 0.5 dB or better ripple at 10 kHz and at least 45 dB of attenuation at 20 kHz. Determine the system function in a cascade form. Plot the magnitude response, the log-magnitude response in dB, the group-delay, and the impulse response of the filter."
    - *Tipo*: Protótipo Butterworth Analógico Passa-Baixas em Hz.
    - *Formato de Transferência*: Forma em Cascata (Biquads em $s$).

11. **Exercício P8.14 (Página 465)**
    - *Enunciado*: "Design a lowpass analog Chebyshev-I filter with an acceptable ripple of 1 dB for $|\Omega| \le 10$ and an attenuation of 50 dB or greater beyond $|\Omega| = 15\text{ rad/sec}$. Determine the system function in a rational function form. Plot the magnitude response, the log-magnitude response in dB, the group-delay, and the impulse response of the filter."
    - *Tipo*: Protótipo Chebyshev-I Analógico Passa-Baixas.
    - *Formato de Transferência*: Forma Racional $H_a(s) = B(s)/A(s)$.

12. **Exercício P8.15 (Páginas 465–466)**
    - *Enunciado*: "Design a lowpass analog Chebyshev-I filter with the following characteristics: A passband ripple of 0.5 dB, passband cutoff frequency of 4 kHz, and stopband attenuation of 45 dB or greater beyond 20 kHz. Determine the system function in a cascade form. Plot the magnitude response, the log-magnitude response in dB, the phase response, and the impulse response of the filter."
    - *Tipo*: Protótipo Chebyshev-I Analógico Passa-Baixas em Hz.
    - *Formato de Transferência*: Forma em Cascata (Biquads em $s$).

---

## 3. FUNDAMENTAÇÃO TEÓRICA E MAPA DE EQUAÇÕES

### 3.1. Filtros FIR de Fase Linear (Capítulo 7)

#### A. Conversão de Especificações Lineares para Decibéis
- $R_p = -20 \log_{10} \left( \frac{1 - \delta_1}{1 + \delta_1} \right) \text{ dB}$
- $A_s = -20 \log_{10}(\delta_2) \text{ dB}$

#### B. Tipos de Filtros Simétricos (Fase Linear)
- **Tipo 1 ($M$ ímpar, $h(n)$ simétrico)**: $\alpha = (M-1)/2$. Aplicável a todas as seletividades (LPF, HPF, BPF, BSF).
- **Tipo 2 ($M$ par, $h(n)$ simétrico)**: Zero fixo obrigatório em $\omega = \pi$ ($H_r(\pi) = 0$). **Proibido para HPF e BSF**.

#### C. Janelas Fixas (Tabela 7.1, Pág. 325)
- **Hann**: $w(n) = 0.5 - 0.5\cos\left(\frac{2\pi n}{M-1}\right), \quad \Delta\omega = \frac{6.2\pi}{M}, \quad A_s = 44\text{ dB}$
- **Hamming**: $w(n) = 0.54 - 0.46\cos\left(\frac{2\pi n}{M-1}\right), \quad \Delta\omega = \frac{6.6\pi}{M}, \quad A_s = 53\text{ dB}$
- **Blackman**: $w(n) = 0.42 - 0.5\cos\left(\frac{2\pi n}{M-1}\right) + 0.08\cos\left(\frac{4\pi n}{M-1}\right), \quad \Delta\omega = \frac{11\pi}{M}, \quad A_s = 74\text{ dB}$

#### D. Método da Janela de Kaiser (Páginas 338–342)
- $\delta = \min(\delta_1, \delta_2) \implies A_s = -20 \log_{10}(\delta)$
- $M \approx \left\lceil \frac{A_s - 7.95}{2.285 \Delta\omega} \right\rceil + 1$
- $\beta = \begin{cases} 0.1102(A_s - 8.7), & A_s \ge 50 \text{ dB} \\ 0.5842(A_s - 21)^{0.4} + 0.07886(A_s - 21), & 21 < A_s < 50 \text{ dB} \\ 0, & A_s \le 21 \text{ dB} \end{cases}$
- $w_{kaiser}(n, \beta) = \frac{I_0\left( \beta \sqrt{1 - \left( \frac{2n}{M-1} - 1 \right)^2} \right)}{I_0(\beta)}, \quad 0 \le n \le M-1$

#### E. Respostas ao Impulso Ideais $h_d(n)$ com Atraso $\alpha = (M-1)/2$
- **Passa-Baixas (LPF)**: $h_d(n) = \frac{\sin[\omega_c(n-\alpha)]}{\pi(n-\alpha)}$ para $n \neq \alpha$, e $\frac{\omega_c}{\pi}$ para $n = \alpha$.
- **Passa-Altas (HPF)**: $h_d(n) = \delta(n-\alpha) - h_{LPF}(n)$.
- **Passa-Faixa (BPF)**: $h_d(n) = h_{LPF2}(n) - h_{LPF1}(n)$.
- **Rejeita-Faixa (BSF)**: $h_d(n) = \delta(n-\alpha) - h_{BPF}(n)$.
- **Multibanda (Staircase)**: $h_d(n) = G_1 h_{LPF1}(n) + G_2 [h_{LPF2}(n) - h_{LPF1}(n)] + \dots$

---

### 3.2. Filtros IIR Analógicos Protótipos (Capítulo 8)

#### A. Protótipo Butterworth (Maximamente Plano)
- $|H_a(j\Omega)|^2 = \frac{1}{1 + (\Omega / \Omega_c)^{2N}}$
- Ordem $N = \left\lceil \frac{\log_{10}\left( \frac{10^{R_p/10} - 1}{10^{A_s/10} - 1} \right)}{2 \log_{10}(\Omega_p / \Omega_s)} \right\rceil$
- Frequência de corte no ponto de passagem: $\Omega_c = \frac{\Omega_p}{(10^{R_p/10} - 1)^{1/(2N)}}$
- Polos no semiplano esquerdo (LHP): $p_k = \Omega_c \exp\left( j \left[ \frac{\pi}{2} + \frac{(2k+1)\pi}{2N} \right] \right), \quad k=0, \dots, N-1$.

#### B. Protótipo Chebyshev Tipo I (Equiripple na Passagem)
- $|H_a(j\Omega)|^2 = \frac{1}{1 + \epsilon^2 T_N^2(\Omega / \Omega_p)}$, onde $\epsilon = \sqrt{10^{R_p/10} - 1}$.
- Ordem $N = \left\lceil \frac{\text{acosh}(g)}{\text{acosh}(\Omega_r)} \right\rceil$, onde $\Omega_r = \frac{\Omega_s}{\Omega_p}$ e $g = \frac{\sqrt{10^{A_s/10} - 1}}{\epsilon}$.
- Polos elípticos no LHP: $p_k = \sigma_k + j\Omega_k$
  - $\alpha = \frac{1}{\epsilon} + \sqrt{1 + \frac{1}{\epsilon^2}}$
  - $a = \frac{1}{2}(\alpha^{1/N} - \alpha^{-1/N}), \quad b = \frac{1}{2}(\alpha^{1/N} + \alpha^{-1/N})$
  - $\sigma_k = -a \Omega_p \sin\left[ \frac{(2k+1)\pi}{2N} \right], \quad \Omega_k = b \Omega_p \cos\left[ \frac{(2k+1)\pi}{2N} \right]$

---

## 4. RESOLUÇÃO DETALHADA E ESPECIFICAÇÃO DE CADA EXERCÍCIO

---

### RESOLUÇÃO P7.9
- **Parâmetros**: $\omega_{1s} = 0.20\pi, \omega_{1p} = 0.35\pi, \omega_{2p} = 0.55\pi, \omega_{2s} = 0.75\pi, A_s = 40\text{ dB}, R_p = 0.25\text{ dB}$.
- **Cálculo da Ordem**:
  - $\Delta\omega_1 = 0.15\pi, \Delta\omega_2 = 0.20\pi \implies \Delta\omega = 0.15\pi$.
  - Janela de Hann ($A_s = 44\text{ dB} \ge 40\text{ dB}$): $M = \lceil 6.2\pi / 0.15\pi \rceil + 1 = 43$ (Tipo 1, $\alpha = 21$).
- **Frequências de Corte Ideais**: $\omega_{c1} = 0.275\pi, \omega_{c2} = 0.650\pi$.
- **Resultados de Validação Esperados**:
  - Atenuação medida na rejeição: $44.00\text{ dB}$.
  - Ondulação medida na passagem: $0.1019\text{ dB}$.

---

### RESOLUÇÃO P7.10
- **Parâmetros**: $\omega_{1p} = 0.30\pi, \omega_{1s} = 0.40\pi, \omega_{2s} = 0.60\pi, \omega_{2p} = 0.70\pi, A_s = 50\text{ dB}, R_p = 0.20\text{ dB}$.
- **Cálculo da Ordem**:
  - $\Delta\omega = 0.10\pi$.
  - Janela de Hamming ($A_s = 53\text{ dB} \ge 50\text{ dB}$): $M = \lceil 6.6\pi / 0.10\pi \rceil + 1 = 67$ (Tipo 1, ímpar obrigatório para BSF, $\alpha = 33$).
- **Frequências de Corte Ideais**: $\omega_{c1} = 0.350\pi, \omega_{c2} = 0.650\pi$.
- **Resultados de Validação Esperados**:
  - Atenuação medida na rejeição: $51.11\text{ dB}$.
  - Ondulação medida na passagem: $0.0417\text{ dB}$.

---

### RESOLUÇÃO P7.11
- **Parâmetros**: $\omega_{1s} = 0.30\pi, \omega_{1p} = 0.40\pi, \omega_{2p} = 0.50\pi, \omega_{2s} = 0.60\pi, A_s = 50\text{ dB}, R_p = 0.50\text{ dB}$.
- **Cálculo da Ordem**:
  - $\Delta\omega = 0.10\pi$.
  - Janela de Hamming: $M = \lceil 6.6\pi / 0.10\pi \rceil + 1 = 67$ (Tipo 1, $\alpha = 33$).
- **Frequências de Corte Ideais**: $\omega_{c1} = 0.350\pi, \omega_{c2} = 0.550\pi$.
- **Resultados de Validação Esperados**:
  - Atenuação medida na rejeição: $51.46\text{ dB}$.
  - Ondulação medida na passagem: $0.0453\text{ dB}$.

---

### RESOLUÇÃO P7.12
- **Parâmetros**: $\omega_s = 0.40\pi, \omega_p = 0.60\pi, A_s = 50\text{ dB}, R_p = 0.004\text{ dB}$.
- **Seleção Crítica da Janela**:
  - Hamming atinge $A_s = 53\text{ dB}$, porém produz $R_p \approx 0.04\text{ dB} > 0.004\text{ dB}$ (inaceitável).
  - Adota-se a **Janela de Blackman** ($A_s = 74\text{ dB}$), cuja ondulação de passagem é $< 0.003\text{ dB}$.
- **Cálculo da Ordem**:
  - $\Delta\omega = 0.20\pi \implies M = \lceil 11.0\pi / 0.20\pi \rceil + 1 = 57$ (Tipo 1, $\alpha = 28$).
  - Corte ideal: $\omega_c = 0.50\pi$.
- **Resultados de Validação Esperados**:
  - Atenuação medida na rejeição: $75.25\text{ dB}$.
  - Ondulação medida na passagem: $0.002942\text{ dB} \le 0.004\text{ dB}$.

---

### RESOLUÇÃO P7.13
- **Especificações Absolutas**: Rejeita-Faixa. $\delta_1 = 0.025, \delta_2 = 0.005$.
- **Cálculos de Kaiser**:
  - $\delta = \min(0.025, 0.005) = 0.005 \implies A_s = -20\log_{10}(0.005) = 46.0206\text{ dB}$.
  - $\Delta\omega = 0.35\pi - 0.25\pi = 0.10\pi$.
  - Parâmetro $\beta = 0.5842(46.02 - 21)^{0.4} + 0.07886(46.02 - 21) = 4.0909$.
  - Ordem $M = \lceil (46.02 - 7.95)/(2.285 \cdot 0.10\pi) \rceil + 1 = 54 \to 55$ (ímpar obrigatório para BSF, $\alpha = 27$).
  - Cuts: $\omega_{c1} = 0.30\pi, \omega_{c2} = 0.70\pi$.
- **Resultados Esperados**:
  - $M_{mínimo} = 55, \beta = 4.0909$.

---

### RESOLUÇÃO P7.14
- **Especificações Absolutas**: Passa-Faixa. $\delta_1 = 0.05, \delta_2 = 0.01$.
- **Cálculos de Kaiser**:
  - $\delta = 0.01 \implies A_s = -20\log_{10}(0.01) = 40.0000\text{ dB}$.
  - $\Delta\omega = 0.10\pi$.
  - Parâmetro $\beta = 0.5842(40 - 21)^{0.4} + 0.07886(40 - 21) = 3.3953$.
  - Ordem $M = \lceil (40 - 7.95)/(2.285 \cdot 0.10\pi) \rceil + 1 = 46 \to 47$ (Tipo 1, $\alpha = 23$).
  - Cuts: $\omega_{c1} = 0.30\pi, \omega_{c2} = 0.70\pi$.
- **Resultados Esperados**:
  - $M_{mínimo} = 47, \beta = 3.3953$.

---

### RESOLUÇÃO P7.15
- **Especificações Multibanda ("Staircase")**:
  - Banda 1: $[0, 0.3\pi]$, ganho 1.0, $\delta_1 = 0.01$.
  - Banda 2: $[0.4\pi, 0.7\pi]$, ganho 0.5, $\delta_2 = 0.005$.
  - Banda 3: $[0.8\pi, \pi]$, ganho 0.0, $\delta_3 = 0.001$.
- **Cálculos de Kaiser**:
  - $\delta_{mínimo} = 0.001 \implies A_s = -20\log_{10}(0.001) = 60.0000\text{ dB}$.
  - $\Delta\omega = 0.10\pi$.
  - Parâmetro $\beta = 0.1102(60 - 8.7) = 5.6533$.
  - Ordem $M = \lceil (60 - 7.95)/(2.285 \cdot 0.10\pi) \rceil + 1 = 74 \to 75$ (Tipo 1, $\alpha = 37$).
  - Resposta Ideal: $h_d(n) = 0.5 h_{LP1}(n) + 0.5 h_{LP2}(n)$ com cortes em $0.35\pi$ e $0.75\pi$.
- **Comparação Acadêmica com Exemplo 7.26**:
  - Janela de Kaiser: $M = 75$.
  - Algoritmo de Parks-McClellan (Equiripple, Exemplo 7.26): $M = 49$.
  - *Conclusão*: O método Equiripple reduz a ordem do filtro em aproximadamente $35\%$ em relação à Janela de Kaiser.

---

### RESOLUÇÃO P7.16
- **Especificações**: Passa-Faixa de Menor Ordem Mínima. $\omega_{1s}=0.3\pi, \omega_{1p}=0.4\pi, \omega_{2p}=0.5\pi, \omega_{2s}=0.6\pi, A_s=40\text{ dB}, R_p=0.5\text{ dB}$.
- **Comparação de Janelas Fixas Candidatas para $\Delta\omega = 0.10\pi$ e $A_s = 40\text{ dB}$**:
  - Janela de Hann ($A_s = 44\text{ dB} \ge 40\text{ dB}$): $M = \lceil 6.2\pi / 0.10\pi \rceil + 1 = 63$.
  - Janela de Hamming ($A_s = 53\text{ dB}$): $M = \lceil 6.6\pi / 0.10\pi \rceil + 1 = 67$.
  - Janela de Blackman ($A_s = 74\text{ dB}$): $M = \lceil 11.0\pi / 0.10\pi \rceil + 1 = 111$.
- **Seleção Final**: **Janela de Hann com $M = 63$** (proporciona o menor comprimento absoluto).

---

### RESOLUÇÃO P8.12
- **Especificações Analógicas**: Butterworth Passa-Baixas. $\Omega_p = 500\text{ rad/s}, R_p = 0.25\text{ dB}, \Omega_s = 2000\text{ rad/s}, A_s = 50\text{ dB}$.
- **Cálculo da Ordem e Frequência de Corte**:
  - $N = \left\lceil \frac{\log_{10}\left( \frac{10^{0.025}-1}{10^5-1} \right)}{2 \log_{10}(500/2000)} \right\rceil = \lceil 5.17 \rceil = 6$.
  - $\Omega_c = \frac{500}{(10^{0.025}-1)^{1/12}} = 632.7668\text{ rad/s}$.
- **Forma Racional $H_a(s) = B(s)/A(s)$**:
  - Numerador: $B(s) = \Omega_c^6 = 6.41892345 \times 10^{16}$.
  - Polinômio do Denominador $A(s) = s^6 + a_5 s^5 + a_4 s^4 + a_3 s^3 + a_2 s^2 + a_1 s + a_0$:
    - $a_5 = 2.444823 \times 10^3$
    - $a_4 = 2.988580 \times 10^6$
    - $a_3 = 2.316084 \times 10^9$
    - $a_2 = 1.196609 \times 10^{12}$
    - $a_1 = 3.919424 \times 10^{14}$
    - $a_0 = 6.418923 \times 10^{16}$

---

### RESOLUÇÃO P8.13
- **Especificações Analógicas**: Butterworth Passa-Baixas em Hz. $F_p = 10\text{ kHz} (\Omega_p = 20000\pi\text{ rad/s}), R_p = 0.5\text{ dB}, F_s = 20\text{ kHz} (\Omega_s = 40000\pi\text{ rad/s}), A_s = 45\text{ dB}$.
- **Cálculo da Ordem e Frequência de Corte**:
  - $N = \left\lceil \frac{\log_{10}\left( \frac{10^{0.05}-1}{10^{4.5}-1} \right)}{2 \log_{10}(1/2)} \right\rceil = \lceil 8.99 \rceil = 9$.
  - $\Omega_c = \frac{20000\pi}{(10^{0.05}-1)^{1/18}} = 70621.0233\text{ rad/s}$.
- **Forma em Cascata (4 Biquads + 1 Seção de 1ª Ordem)**:
  - $H_a(s) = \frac{\Omega_c}{s + \Omega_c} \cdot \prod_{k=1}^4 \frac{\Omega_c^2}{s^2 + 2\Omega_c \cos(\theta_k) s + \Omega_c^2}$
  - Ângulos $\theta_k = \frac{(2k-1)\pi}{18}$ para $k=1,2,3,4$.
  - Seção de 1ª Ordem: Polo real em $s = -70621.02$.

---

### RESOLUÇÃO P8.14
- **Especificações Analógicas**: Chebyshev-I Passa-Baixas. $\Omega_p = 10\text{ rad/s}, R_p = 1.0\text{ dB}, \Omega_s = 15\text{ rad/s}, A_s = 50\text{ dB}$.
- **Cálculo da Ordem e Fator de Ondulação**:
  - $\epsilon = \sqrt{10^{0.1} - 1} = 0.508847$.
  - $g = \frac{\sqrt{10^5 - 1}}{0.508847} = 621.456$.
  - $\Omega_r = 15/10 = 1.5$.
  - $N = \left\lceil \frac{\text{acosh}(621.456)}{\text{acosh}(1.5)} \right\rceil = \left\lceil \frac{7.1252}{0.9624} \right\rceil = \lceil 7.403 \rceil = 8$.
- **Forma Racional $H_a(s) = B(s)/A(s)$**:
  - Numerador: $B(s) = \frac{a_0}{\sqrt{1 + \epsilon^2}} = 1.535333 \times 10^6$.
  - Coeficiente constante do denominador: $a_0 = 1.722672 \times 10^6$.

---

### RESOLUÇÃO P8.15
- **Especificações Analógicas**: Chebyshev-I Passa-Baixas em Hz. $F_p = 4\text{ kHz} (\Omega_p = 8000\pi\text{ rad/s}), R_p = 0.5\text{ dB}, F_s = 20\text{ kHz} (\Omega_s = 40000\pi\text{ rad/s}), A_s = 45\text{ dB}$.
- **Cálculo da Ordem**:
  - $\epsilon = \sqrt{10^{0.05} - 1} = 0.349311$.
  - $g = \frac{\sqrt{10^{4.5}-1}}{0.349311} = 509.076$.
  - $\Omega_r = 20/4 = 5.0$.
  - $N = \left\lceil \frac{\text{acosh}(509.076)}{\text{acosh}(5.0)} \right\rceil = \left\lceil \frac{6.9257}{2.2924} \right\rceil = \lceil 3.021 \rceil = 4$.
- **Forma em Cascata (2 Biquads de 2ª Ordem)**:
  - Polos complexos conjugados obtidos:
    - Par 1: $p_{1,2} = -4407.10 \pm j 25541.22$
    - Par 2: $p_{3,4} = -10639.69 \pm j 10579.52$
  - Biquad 1: $H_1(s) = \frac{6.7178 \times 10^8}{s^2 + 8814.21 s + 6.7178 \times 10^8}$
  - Biquad 2: $H_2(s) = \frac{2.2513 \times 10^8}{s^2 + 21279.38 s + 2.2513 \times 10^8}$
  - Ganho de Escala Global: Multiplicar por $\frac{1}{\sqrt{1 + \epsilon^2}} = 0.94406$.

---

## 5. REQUISITOS E DIRETRIZES DE IMPLEMENTAÇÃO PYTHON

Ao implementar os scripts no ambiente Python, o Agente de Implementação deverá respeitar estritamente as seguintes diretrizes:

1. **Vedação de Funções Automáticas de Alto Nível para FIR**:
   - É **proibido** utilizar `scipy.signal.fir1` ou `scipy.signal.firls`.
   - A resposta ao impulso ideal $h_d(n)$ deve ser construída explicitamente via NumPy (`np.sin`, `np.where`).
   - As janelas (Hann, Hamming, Blackman, Kaiser) devem ser sintetizadas analiticamente ou via `np.kaiser` / `scipy.special.i0`.
2. **Uso de SciPy para Protótipos Analógicos IIR**:
   - Utilizar `scipy.signal.buttord`, `scipy.signal.butter`, `scipy.signal.cheb1ord`, `scipy.signal.cheby1` com a flag `analog=True`.
   - Utilizar `scipy.signal.freqs` para avaliação em frequência analógica e `scipy.signal.impulse` para resposta ao impulso.
3. **Padrão Gráfico Matplotlib**:
   - Para FIR: Subplots contendo `plt.stem(n, h)` para a resposta ao impulso e `plt.plot(w/np.pi, 20*np.log10(np.abs(H)))` para magnitude em dB. Linhas horizontais `plt.axhline(-As)` e verticais `plt.axvline(w_edge/np.pi)` devem marcar os limites.
   - Para IIR: Magnitude $|H(j\Omega)|$, Log-Magnitude (dB), Fase (graus ou radianos), Atraso de Grupo (obtido via $-\frac{d\phi(\Omega)}{d\Omega}$) e Resposta ao Impulso no tempo $h(t)$.

---

## 6. MAPA DE REFERÊNCIA DE PÁGINAS DO LIVRO

- **Tabela de Janelas Fixas (Hann, Hamming, Blackman)**: Tabela 7.1, Página 325.
- **Fórmulas de Kaiser ($eta, M$)**: Seção 7.5, Páginas 338–342.
- **Projeto de Protótipos Butterworth Analógicos**: Seção 8.2.1, Páginas 401–406.
- **Projeto de Protótipos Chebyshev-I Analógicos**: Seção 8.2.2, Páginas 406–412.
- **Exercícios P7.9 a P7.16**: Páginas 378–380.
- **Exercícios P8.12 a P8.15**: Páginas 465–466.

---
*Fim do Documento de Transferência Acadêmica.*
