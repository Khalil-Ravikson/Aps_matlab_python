# Plano de Implementação — DSP Avaliação 2

**Referência**: INGLE, V. K.; PROAKIS, J. G. *Digital Signal Processing Using MATLAB*, 3ª ed., 2012.
**Fonte de especificação**: `docs/documento_transferencia_academica.md` (doravante **DTA**).
**Data**: 2026-10-30 · **Status**: todas as etapas concluídas (0–10); 12 exercícios implementados e validados

---

## 0. Resumo executivo

- O diretório `docs/` contém **um único documento** (DTA), que cobre os 12 exercícios (P7.9–P7.16, P8.12–P8.15).
- Todas as afirmações numéricas do DTA foram **reproduzidas de forma independente** (NumPy 2.5.2 / SciPy 1.18.0, script descartável fora do projeto) antes deste plano.
- **Os resultados do IIR batem com o DTA**, exceto por **um erro de fórmula** na forma em cascata do P8.13.
- **No FIR há divergências que mudam respostas**: o comprimento "mínimo" do DTA para P7.13 (**M = 55**) **não atende à especificação** quando medido; P7.14, P7.15 e P7.16 admitem comprimentos menores que os do DTA. Alguns valores "medidos" do DTA (P7.10, P7.11) não se reproduzem.
- Conclusão: o DTA serve como **fonte das especificações (enunciados) e do método**, mas **seus números de resultado não podem ser usados como oráculo de teste**. Os testes validarão contra a **especificação do enunciado**.
- Pilotos propostos: **P7.9** (FIR) e **P8.12** (IIR) — os dois exercícios sem ambiguidades bloqueantes.

---

## 1. Inventário dos 12 exercícios e parâmetros extraídos

Frequências digitais em múltiplos de π rad/amostra. ω_s = borda de rejeição, ω_p = borda de passagem.

### 1.1 Capítulo 7 — FIR (método da janela)

| Ex. | Tipo | Método exigido | Especificações (do enunciado) | Saídas pedidas |
|---|---|---|---|---|
| P7.9 | Passa-faixa | Janela **Hann** | ω₁ₛ=0.2, ω₁ₚ=0.35, ω₂ₚ=0.55, ω₂ₛ=0.75; Rp=0.25 dB; As=40 dB | h(n); \|H\| dB |
| P7.10 | Rejeita-faixa | Janela **Hamming** | ω₁ₚ=0.3, ω₁ₛ=0.4, ω₂ₛ=0.6, ω₂ₚ=0.7; Rp=0.2 dB; As=50 dB | h(n); \|H\| dB |
| P7.11 | Passa-faixa | Janela **Hamming** | ω₁ₛ=0.3, ω₁ₚ=0.4, ω₂ₚ=0.5, ω₂ₛ=0.6; Rp=0.5 dB; As=50 dB | h(n); \|H\| dB |
| P7.12 | Passa-alta | Uma janela **fixa** (escolha justificada) | ωₛ=0.4, ωₚ=0.6; Rp=0.004 dB; As=50 dB | \|H\| dB com *zoom* na passagem |
| P7.13 | Rejeita-faixa | **Kaiser**, M mínimo | \|H\|∈[0.975,1.025] em [0,0.25]; \|H\|≤0.005 em [0.35,0.65]; \|H\|∈[0.975,1.025] em [0.75,1] | M mínimo; Hr(ω) e \|H\| dB |
| P7.14 | Passa-faixa | **Kaiser**, M mínimo | \|H\|≤0.01 em [0,0.25]; \|H\|∈[0.95,1.05] em [0.35,0.65]; \|H\|≤0.01 em [0.75,1] | M mínimo; Hr(ω) e \|H\| dB |
| P7.15 | Multibanda (escada) | **Kaiser** | B1 [0,0.3] G=1 δ=0.01; B2 [0.4,0.7] G=0.5 δ=0.005; B3 [0.8,1] G=0 δ=0.001 | M; comparar com Ex. 7.26; \|H\| dB |
| P7.16 | Passa-faixa | Janela **fixa** de **menor comprimento** | Mesmas bordas de P7.11; Rp=0.5 dB; As=40 dB | \|H\| dB; h(n) (stem) |

Restrição comum (P7.9–P7.15): "**Do not use `fir1`**". O equivalente em SciPy é `scipy.signal.firwin` (o DTA cita `scipy.signal.fir1`, que não existe). Ele **não** será usado no código de solução, só como **verificação cruzada nos testes**.

### 1.2 Capítulo 8 — protótipos analógicos IIR

| Ex. | Aproximação | Especificações | Forma pedida | Gráficos pedidos |
|---|---|---|---|---|
| P8.12 | Butterworth PB | Ωₚ=500 rad/s, Rp=0.25 dB; Ωₛ=2000 rad/s, As=50 dB | **Racional** B(s)/A(s) | \|H\|, dB, **fase**, h(t) |
| P8.13 | Butterworth PB | Fₚ=10 kHz, Rp=0.5 dB; Fₛ=20 kHz, As=45 dB | **Cascata** | \|H\|, dB, **atraso de grupo**, h(t) |
| P8.14 | Chebyshev-I PB | Ωₚ=10 rad/s, Rp=1 dB; Ωₛ=15 rad/s, As=50 dB | **Racional** | \|H\|, dB, **atraso de grupo**, h(t) |
| P8.15 | Chebyshev-I PB | Fₚ=4 kHz, Rp=0.5 dB; Fₛ=20 kHz, As=45 dB | **Cascata** | \|H\|, dB, **fase**, h(t) |

---

## 2. Métodos matemáticos (com referência ao DTA)

### 2.1 FIR — projeto por janela (DTA §3.1)

1. **Conversão de especificações** (DTA §3.1.A): Rp = −20 log₁₀[(1−δ₁)/(1+δ₁)], As = −20 log₁₀ δ₂.
2. **Largura de transição**: Δω = min(ωₚ−ωₛ) entre todas as transições; **corte ideal** ω_c = (ωₚ+ωₛ)/2 em cada transição.
3. **Comprimento** (janelas fixas, Tabela 7.1): M = ⌈k·π/Δω⌉ + 1, com k = 6.2 (Hann), 6.6 (Hamming), 11 (Blackman); A_s da janela: 44 / 53 / 74 dB.
4. **Kaiser** (DTA §3.1.D): A_s = −20 log₁₀ min(δᵢ); β por partes; M ≈ ⌈(A_s − 7.95)/(2.285 Δω)⌉ + 1.
5. **Resposta ideal com atraso α=(M−1)/2** (DTA §3.1.E): h_LP(n) = sin[ω_c(n−α)]/[π(n−α)]; PA = δ(n−α) − h_LP; PF = h_LP2 − h_LP1; RF = δ(n−α) − h_PF; escada = Σ Gₖ·(h_LP,k − h_LP,k−1).
6. **h(n) = h_d(n)·w(n)**.
7. **Tipo de fase linear** (DTA §3.1.B): M ímpar → Tipo 1; M par → Tipo 2 (zero obrigatório em ω=π, proibido em PA e RF).
8. **Resposta de amplitude** Hr(ω) (P7.13/P7.14): Hr(ω) = Re{H(e^{jω}) e^{jωα}}, real e com sinal.

### 2.2 IIR — protótipos analógicos (DTA §3.2)

- **Butterworth**: N = ⌈log₁₀[(10^{Rp/10}−1)/(10^{As/10}−1)] / (2 log₁₀(Ωₚ/Ωₛ))⌉; Ω_c = Ωₚ/(10^{Rp/10}−1)^{1/(2N)} (atende a **passagem** com exatidão, convenção do livro); polos pₖ = Ω_c·exp{j[π/2 + (2k+1)π/(2N)]}.
- **Chebyshev-I**: ε = √(10^{Rp/10}−1); N = ⌈acosh(g)/acosh(Ωₛ/Ωₚ)⌉, g = √(10^{As/10}−1)/ε; polos por (a, b) da elipse; ganho DC = 1/√(1+ε²) para N par, 1 para N ímpar.
- **Atraso de grupo**: o DTA sugere −dφ/dΩ numérico. Proponho a **forma analítica a partir dos polos**, τ(Ω) = Σₖ −Re(pₖ) / [Re(pₖ)² + (Ω − Im pₖ)²] (filtros só-polos), com a derivada numérica da fase desenrolada mantida como **validação cruzada**. Justificativa: é exata, sem ruído de diferenciação e fácil de defender em sala.

---

## 3. Mapeamento para NumPy / SciPy

| Necessidade | Implementação | Observação |
|---|---|---|
| h_d(n) ideal | NumPy explícito (`np.sin`, `np.where`) | Exigência do enunciado |
| Hann, Hamming, Blackman | Fórmulas da Tabela 7.1 em NumPy | Teste contra `scipy.signal.windows.*(M, sym=True)` |
| Kaiser | `scipy.special.i0` + fórmula do DTA | Teste contra `np.kaiser` |
| Resposta em frequência FIR | `scipy.signal.freqz` | Grade do livro (`freqz_m`): 501 pontos em [0, π] |
| Verificação cruzada FIR | `scipy.signal.firwin` | **Somente em testes** |
| Comparação P7.15 vs Ex. 7.26 | `scipy.signal.remez` (opcional) | Depende da decisão D6 |
| Ordem/corte Butterworth | Fórmulas do livro + `signal.buttord(analog=True)` | Verificado: o SciPy 1.18 dá o mesmo Ω_c (critério da passagem) |
| Projeto Butterworth/Cheb-I | `signal.butter` / `signal.cheby1` (`analog=True`, `output='zpk'`) | Polos também calculados pelas fórmulas do livro e comparados |
| Forma racional | `signal.zpk2tf` | Ok para P8.12/P8.14 (coeficientes até ~10¹⁷) |
| Forma em cascata | Agrupamento de pares conjugados em biquads em s (código próprio) | `zpk2sos` é para sistemas discretos; para s o agrupamento é feito explicitamente |
| Resposta em frequência analógica | `signal.freqs_zpk` | Evita o mau condicionamento de `freqs(b, a)` |
| Resposta ao impulso | `signal.impulse` sobre `signal.lti` em **zpk/ss** | No P8.13, a₀ ≈ 4.4×10⁴³ e a forma tf é numericamente frágil |

---

## 4. Verificação independente do DTA (pré-implementação)

Grade de medição de Rp/As: 501 pontos em [0, π] (como a `freqz_m` do livro), dB normalizado pelo máximo. Desvios absolutos em grade de 4001 pontos.

### 4.1 FIR

| Ex. | Afirmação do DTA | Recalculado | Situação |
|---|---|---|---|
| P7.9 | M=43; As=44.00; Rp=0.1019 | M=43; As=44.05; Rp=0.1030 | ✅ (diferença de grade) |
| P7.10 | M=67; As=**51.11**; Rp=0.0417 | M=67; As=**50.04**; Rp=0.0435 | ⚠️ Atende com margem de 0.04 dB. **Causa da divergência identificada**: o valor do DTA se reproduz exatamente numa grade de 2049 pontos (passo π/2048), que **não contém as bordas** 0.4π/0.6π e por isso superestima As. Com bordas na grade (501 ou 20001 pontos): 50.04 dB |
| P7.11 | M=67; As=51.46; Rp=0.0453 | As=51.20; Rp=0.0488 | ⚠️ Mesma causa (grade de 2049 pontos sem as bordas); atende com margem de 1.2 dB |
| P7.12 | Blackman M=57; As=75.25; Rp=0.002942 | As=75.28; Rp=0.002954 | ✅ |
| P7.12 | "Hamming dá Rp ≈ 0.04 dB" | Com o M da tabela (35): Rp = **0.040** dB; com M = 41…95: 0.027–0.032 dB | ✅ DTA correto para M = 35 (*corrigido em 2026-10-30: a versão anterior deste plano classificava o valor como errado por ter medido outros M*). A conclusão vale para qualquer M |
| P7.13 | β=4.0909; **M=55** mínimo | β=4.0909 ✅; com M=55: \|H\|ₘₐₓ na rejeição = **0.00593 > 0.005** | ❌ **M=55 não atende.** O menor M ímpar que atende é **57** |
| P7.13 | "⌈x⌉+1 = 54 → 55" | x=53.03 → ⌈x⌉+1 = **55** direto | ⚠️ Aritmética intermediária inconsistente |
| P7.14 | β=3.3953; M=47 (Tipo 1) | β ✅; **M=46 (Tipo 2) atende** (δ₂: 0.0098; δ₁: 0.0129) | ⚠️ Forçar M ímpar não é necessário em passa-faixa |
| P7.15 | β=5.6533; M=75 | β ✅; **M=72 já atende** às 3 bandas | ⚠️ A fórmula superestima e forçar M ímpar é desnecessário (G=0 em π) |
| P7.15 | Ex. 7.26 (Parks-McClellan): M=49 | **Não verificável** sem o livro | ❓ Ver D6 |
| P7.16 | Hann M=63 (pela fórmula) é o menor | Medido: **Hann M=62** e **Hamming M=61** já atendem; Blackman: 82 | ❌/⚠️ A resposta depende do critério (D5) |

### 4.2 IIR

| Ex. | Afirmação do DTA | Recalculado | Situação |
|---|---|---|---|
| P8.12 | N=6; Ω_c=632.7668; a₅…a₀ | Idênticos; ganho em Ωₚ = −0.25 dB, em Ωₛ = −59.97 dB | ✅ |
| P8.13 | N=9; Ω_c=70621.0233 | Idênticos; −0.50 dB / −45.05 dB | ✅ |
| P8.13 | Biquads com θₖ=(2k−1)π/18 | Ângulos reais dos polos (a partir do eixo real negativo): **20°, 40°, 60°, 80°** → **θₖ = kπ/9** | ❌ **Erro de fórmula**: (2k−1)π/(2N) vale para N **par**; com N=9 (ímpar) há polo real e θₖ = kπ/N |
| P8.14 | N=8; B=1.535333×10⁶; a₀=1.722672×10⁶ | Idênticos; −1 dB em Ωₚ; −54.99 dB em Ωₛ | ✅ |
| P8.15 | N=4; polos e biquads; ganho 0.94406 | Idênticos | ✅ |

---

## 5. Ambiguidades e decisões pendentes

Pela regra "não implementar com informação incompleta", os itens marcados **[BLOQUEIA]** precisam de decisão antes de implementar o exercício correspondente.

| ID | Questão | Afeta | Proposta |
|---|---|---|---|
| **D1** ✅ | Definição da janela de Hann: zeros nas extremidades, w(n)=0.5−0.5cos(2πn/(M−1)) como no DTA e no `hann` do MATLAB, ou sem zeros, como o `hanning` do MATLAB? No P7.9 dá As 44.05 contra 43.99 dB. | P7.9, P7.16 | ✅ **Decidido**: Hann com zeros nas extremidades (DTA) |
| **D2** | Significado de "comprimento mínimo": estimativa pela fórmula ou **menor M verificado** que atende à especificação? | P7.13, P7.14, P7.15, P7.16 **[BLOQUEIA]** | ✅ **Decidido (2026-10-30)**: reportar os dois; resposta final = **M verificado** por busca iterativa |
| **D3** | Fórmula de M do Kaiser: o DTA usa ⌈x⌉+1; o código dos exemplos do livro pode usar `ceil(x+1)+1` (não confirmei sem o livro) | P7.13–P7.15 | Usar a fórmula do DTA como estimativa; a busca verificada (D2) torna a diferença irrelevante para a resposta final |
| **D4** | Permitir M par (Tipo 2) em passa-faixa e escada? | P7.14, P7.15 **[BLOQUEIA]** | ✅ **Decidido**: M par permitido em PF e escada; M ímpar obrigatório em PA/RF |
| **D5** | P7.16, critério de "menor comprimento": pela fórmula da Tabela 7.1 (Hann 63) ou pela medição (Hamming 61)? | P7.16 **[BLOQUEIA]** | ✅ **Decidido**: as duas análises; resposta final pela medição (**Hamming M=61**). Justificativa: o Δω da Tabela 7.1 vale para a atenuação plena da janela (53 dB na Hamming); como o enunciado exige só 40 dB, a resposta cruza −40 dB antes do fim da transição tabelada |
| **D6** | Valor de M do Exemplo 7.26 | P7.15 | ✅ **Informado (pp. 367–368)**: estimativa inicial M=51; `firpm` com M=51 → 62.07 dB, M=50 → 60.03 dB, M=49 → 60.61 dB; final **M=49**. Será conferido com `signal.remez` na etapa 7 |
| **D7** | A_s do Kaiser no filtro em escada: o DTA usa min δ = 0.001 → 60 dB. Como os degraus têm altura 0.5, o *ripple* induzido escala com o salto e 60 dB é conservador | P7.15 | Manter o critério do DTA (método do livro) e mencionar a observação na análise |
| **D8** | Grade de frequência para medir Rp/As | Todos FIR | ✅ **Regra**: a grade de avaliação **sempre inclui as frequências de borda exatas** (grade do livro com 501 pontos + grade densa ≥ 4001, ambas com as bordas). Grades de potência de 2 (ex.: `worN=2048`) **não** servem para verificar conformidade: pulam as bordas e superestimam As (é a origem das divergências de P7.10/P7.11) |
| **D9** | Unidades dos gráficos IIR: Hz nos exercícios em kHz (P8.13/P8.15) e rad/s nos demais; fase em unidades de π rad (como o livro) | IIR | Adotar esse padrão |
| **D10** | Números de página do DTA | Rastreabilidade | Não verificáveis sem o livro; mantidos como citados, marcados "conforme DTA" |

---

## 6. Arquitetura proposta

Os diretórios `src/`, `exercises/`, `notebooks/`, `results/` e `tests/` já existem (vazios) e serão aproveitados.

```
dsp-avaliacao-2/
├── pyproject.toml            # metadados, dependências, config do pytest
├── requirements.txt          # para o Colab (pip install -r)
├── .python-version           # 3.12.8 (pyenv)
├── README.md                 # como rodar local / Colab
├── docs/
│   ├── documento_transferencia_academica.md   # fonte (não editar)
│   ├── plano_implementacao.md                 # este documento
│   └── errata_dta.md                          # divergências encontradas (seção 4) e resoluções
├── src/dsp_av2/
│   ├── __init__.py
│   ├── specs.py              # dataclasses imutáveis com as especificações literais de cada exercício + referência (DTA §, página)
│   ├── fir/
│   │   ├── ideal.py          # h_d(n): lowpass, highpass, bandpass, bandstop, multiband
│   │   ├── windows.py        # hann, hamming, blackman, kaiser; kaiser_beta, kaiser_length
│   │   ├── design.py         # window_design(), estimate_length(), find_min_length()
│   │   └── analysis.py       # freqz_m (grade do livro), amplitude_response (Hr), measure_rp_as, band_deviations
│   ├── iir/
│   │   ├── prototypes.py     # ordem/Ω_c/ε/polos pelas fórmulas do livro + wrapper SciPy
│   │   ├── forms.py          # forma racional; cascata analógica (biquads em s)
│   │   └── analysis.py       # freqs_zpk, group_delay analítico, impulse (zpk/ss)
│   ├── validation.py         # verifica o resultado contra a especificação → relatório (passa/falha + margens)
│   ├── plotting.py           # só visualização (nenhum cálculo de projeto)
│   └── io.py                 # grava results/<ex>/metrics.json e figuras
├── exercises/
│   ├── p7_09.py … p7_16.py   # cada um: SPEC → design() → validate() → plot() → save()
│   └── p8_12.py … p8_15.py
├── notebooks/
│   ├── p7_09.ipynb …         # narrativa acadêmica: equações → código → resultados; 1ª célula de setup Colab
├── results/
│   └── p7_09/ {impulse.png, magnitude_db.png, metrics.json}
└── tests/
    ├── test_fir_windows.py   # janelas vs scipy.signal.windows / np.kaiser
    ├── test_fir_ideal.py     # simetria, h_d vs firwin(window='boxcar')
    ├── test_fir_analysis.py  # Hr real, medição de Rp/As em casos conhecidos
    ├── test_iir_prototypes.py# fórmulas do livro vs buttord/butter/cheb1ord/cheby1
    ├── test_iir_analysis.py  # atraso de grupo analítico vs numérico
    └── test_p7_09.py …       # por exercício: atende à ESPECIFICAÇÃO (não aos números do DTA)
```

### Princípios

1. **Separação cálculo / validação / visualização.** `fir/` e `iir/` só calculam. `validation.py` só compara com a especificação. `plotting.py` só desenha. Os scripts de `exercises/` orquestram.
2. **Rastreabilidade.** Cada função de projeto tem docstring com a equação (LaTeX) e a referência (DTA §x, Ingle & Proakis eq./tabela). Cada `SPEC` cita o enunciado. `metrics.json` registra versões de NumPy/SciPy.
3. **Oráculo de teste = especificação.** Os testes por exercício verificam Rp ≤ Rp_spec, As ≥ As_spec (ou desvios ≤ δᵢ), tipo de fase linear, simetria de h(n) e M igual ao mínimo verificado. Comparações com valores do DTA entram só onde a seção 4 os confirmou.
4. **Implementação própria + verificação cruzada com SciPy.** O código de solução segue o livro (fórmulas explícitas), e os testes confirmam contra as rotinas de alto nível do SciPy.
5. **Reprodutibilidade.** Dependências fixadas, sem aleatoriedade, figuras regeneráveis com um comando (`python -m exercises.p7_09` ou `make`-like script).

### Execução

- **Local (pyenv)**: `pyenv local 3.12.8` → `python -m venv .venv` → `pip install -e ".[dev]"` → `pytest`.
- **Colab**: 1ª célula do notebook detecta o Colab (`"google.colab" in sys.modules`), instala `requirements.txt` e adiciona `src/` ao `sys.path`. Isso pressupõe o projeto no Colab via `git clone` (requer repositório remoto, ainda inexistente) ou upload de `.zip`/Google Drive. A decidir na etapa de infraestrutura.
- Matplotlib ainda **não está instalado** no ambiente pyenv (NumPy 2.5.2 e SciPy 1.18.0 estão).

---

## 7. Plano incremental

Cada etapa termina com testes passando e revisão sua antes da seguinte.

| Etapa | Entrega | Critério de aceite |
|---|---|---|
| **0. Infraestrutura** | `pyproject.toml`, `requirements.txt`, `.python-version`, pacote vazio, `pytest` rodando | `pytest` verde (0 testes), import funciona local |
| **1. Núcleo FIR** | `fir/ideal.py`, `fir/windows.py`, `fir/analysis.py`, `validation.py` + testes unitários | Janelas = SciPy (tol 1e-12); h_d simétrica; Hr real |
| **2. Piloto FIR — P7.9** | `exercises/p7_09.py`, `tests/test_p7_09.py`, notebook, figuras, `metrics.json` | M=43, Tipo 1, ω_c=0.275π/0.65π; Rp ≤ 0.25 dB; As ≥ 40 dB; h(n) simétrica; firwin(43, hann) idêntico |
| **3. Núcleo IIR** | `iir/prototypes.py`, `iir/forms.py`, `iir/analysis.py` + testes | Fórmulas = SciPy; atraso analítico = numérico |
| **4. Piloto IIR — P8.12** | Script, teste, notebook, figuras (\|H\|, dB, fase, h(t)) | N=6; Ω_c=632.7668; coeficientes de A(s) conforme seção 4.2; −0.25 dB em Ωₚ; ≤ −50 dB em Ωₛ |
| **⏸ Checkpoint** | Revisão dos pilotos com você (estrutura, gráficos, linguagem) | Aprovação explícita |
| **5. Janelas fixas** | P7.10, P7.11, P7.12 | Especificações atendidas; P7.12 com tabela comparando janelas fixas |
| **6. Kaiser** | `find_min_length`; P7.13, P7.14 | Depende de D2/D4; M estimado e M verificado reportados |
| **7. Escada** | P7.15 | Depende de D6/D7 |
| **8. Menor janela fixa** | P7.16 | Depende de D5; tabela Hann/Hamming/Blackman (fórmula × medido) |
| **9. IIR restantes** | P8.13 (com θₖ corrigido), P8.14, P8.15 | Cascata recomposta = forma direta; especificações atendidas |
| **10. Consolidação** | `errata_dta.md`, README, tabela-resumo dos 12 resultados | Tudo regenerável; `pytest` verde |

---

## 8. Próxima ação

Etapas 0 → 2 (infraestrutura, núcleo FIR e piloto P7.9) assim que você aprovar este plano. Para o P7.9 só falta confirmar **D1** (definição da Hann). Os demais itens bloqueantes (D2, D4, D5, D6) podem ser decididos até a etapa 6.

---

## 9. Registro de execução

- **2026-10-30, etapas 0–2.** `pyproject.toml`, `.venv`, pacote `dsp_av2`, núcleo FIR e P7.9 prontos. 79 testes passando.
  Achado durante a implementação: no P7.12, ⌈11π/0.2π⌉ + 1 = **56**, não 57 como diz o DTA. O valor final 57 continua correto, mas pela regra de paridade (passa-altas exige M ímpar), não pela aritmética. Registrar na errata.
- **2026-10-30, etapas 3–4.** Núcleo IIR (`iir/prototypes.py`, `iir/forms.py`, `iir/analysis.py`) e P8.12 prontos. 112 testes passando.
  Decisões técnicas: (i) h(t) é simulada com o filtro normalizado por Ω₀ = max|p_k|, h(t) = Ω₀·h_n(Ω₀t), porque a simulação direta em zpk gera estados complexos e a forma tf do P8.13 é mal condicionada; o resultado confere com a inversa de Laplace por resíduos (erro relativo ~10⁻¹⁴). (ii) Atraso de grupo analítico pelos polos, conferido com −dφ/dΩ numérico. (iii) Na validação analógica, uma tolerância de 10⁻⁹ dB absorve o ruído de ponto flutuante em R_p, que é atingido com igualdade por construção. Também foram testadas as cascatas de P8.13 (θₖ = kπ/9, confirmando a errata) e P8.15 (biquads e ganho 0.94406 do DTA).
- **2026-10-30, P7.10.** M = 67 (Tipo 1), Rp = 0.044 dB, As = 50.04 dB; 119 testes passando. Achados: (i) o pior ponto da rejeição é a própria borda 0.4π/0.6π (fim da transição), não um lóbulo lateral (≈ −52 dB); o mesmo acontece num passa-baixas Hamming equivalente, então a causa é a aproximação de 6.6π/M da Tabela 7.1; (ii) M = 67 também é o menor ímpar que atende (M = 65 → 44.95 dB); (iii) h(α ± ímpar) = 0, porque a rejeição está centrada em π/2.
- **2026-10-30, P7.11–P8.15 e consolidação.** Todos os exercícios implementados; 181 testes; 12 notebooks executados sem erro. Resultados e divergências em `README.md` e `docs/errata_dta.md`. Achados adicionais: P7.11 também atende com M = 66; no P7.12 a Blackman M = 55 também atende (folga de 7×10⁻⁵ dB) e a Hamming nunca atende Rp; o `remez` reproduz o Ex. 7.26 (M = 49); no P7.15, ajustar o A_s de Kaiser leva no máximo a M ≈ 68, ainda ~40% acima do equiripple; no P8.14, um Butterworth equivalente precisaria de N = 16.
