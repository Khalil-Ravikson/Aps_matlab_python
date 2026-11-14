# Errata do Documento de Transferência Acadêmica (DTA)

**Documento corrigido:** [`documento_transferencia_academica.md`](documento_transferencia_academica.md). Ele é a fonte da pesquisa e é mantido **sem alterações**, por rastreabilidade.
**Data:** 2026-10-30
**Critério:** cada item compara a afirmação do DTA com um valor recalculado no projeto. A evidência é um teste automatizado (`tests/`) ou uma célula de notebook (`notebooks/`).

Legenda: ❌ erro que muda a resposta · ⚠️ erro de valor ou de raciocínio que não muda a resposta · ✅ confirmado.

## 1. Erros que mudam a resposta

| # | Ex. | Afirmação do DTA | Correção | Evidência |
|---|---|---|---|---|
| E1 | P7.13 | "$M_{mínimo} = 55$" | **M = 57.** Com M = 55, $\max\lvert H\rvert$ na rejeição é 0.0059 > 0.005, e a especificação é violada. A fórmula de Kaiser é só uma estimativa | `tests/test_p7_13.py::test_kaiser_estimate_fails_the_stopband_tolerance`, `::test_minimum_length_is_57` |
| E2 | P7.14 | "$M = 46 \to 47$ (Tipo 1)" | **M = 46 (Tipo 2).** O passa-faixa não exige M ímpar: o zero do Tipo 2 em ω = π cai na rejeição | `tests/test_p7_14.py::test_minimum_length_is_46_type_2` |
| E3 | P7.15 | "$M = 74 \to 75$" | **M = 72.** M par é admissível (ganho 0 em π), e a busca medida encontra 72 | `tests/test_p7_15.py::test_minimum_length_is_72` |
| E4 | P7.15 | "O método Equiripple reduz a ordem … em aproximadamente 35%" | Com M = 72, Kaiser é **47% mais longo** que o equiripple (72/49). Os 35% do DTA são $1 - 49/75$, uma redução calculada a partir do Kaiser e com o M errado | `notebooks/p7_15.ipynb`, §4 |
| E5 | P7.16 | "Janela de Hann com M = 63 (menor comprimento absoluto)" | Pelo critério medido (decisão D5): **Hamming, M = 61**. Hann alcança no mínimo 62. O M = 63 vale só pelo critério da Tabela 7.1 | `tests/test_p7_16.py::test_measured_minimum_lengths` |
| E6 | P8.13 | "Ângulos $\theta_k = (2k-1)\pi/18$" | **$\theta_k = k\pi/9$** (20°, 40°, 60°, 80°). A fórmula do DTA vale para N par, e aqui N = 9 | `tests/test_p8_13.py::test_cascade_structure_and_errata`, `tests/test_iir_prototypes.py::test_p8_13_cascade_angles_are_k_pi_over_9` |

## 2. Erros de valor ou de raciocínio (a resposta final não muda)

| # | Ex. | Afirmação do DTA | Correção | Evidência |
|---|---|---|---|---|
| V1 | P7.10 | "Atenuação medida 51.11 dB; ondulação 0.0417 dB" | **50.04 dB e 0.044 dB.** Os valores do DTA vêm de uma grade de 2049 pontos que não contém as bordas 0.4π/0.6π. O pior ponto é justamente a borda | `tests/test_fir_analysis.py::test_grid_without_edges_overestimates_attenuation`, `tests/test_p7_10.py` |
| V2 | P7.11 | "51.46 dB; 0.0453 dB" | **51.20 dB e 0.049 dB** (mesma causa de V1) | `notebooks/p7_11.ipynb`, §3.2 |
| V3 | P7.12 | "$M = \lceil 11\pi/0.2\pi \rceil + 1 = 57$" | A conta dá **56**. O 57 vem da regra de paridade (passa-altas exige M ímpar) | `tests/test_p7_12.py::test_design_parameters` |
| V4 | P7.13 | "$M = \lceil \ldots \rceil + 1 = 54 \to 55$" | $\lceil 53.03 \rceil + 1 = 55$ diretamente. O passo intermediário "54" é inconsistente | `tests/test_fir_windows.py::test_kaiser_length_estimate` |
| V5 | P7.9 | "Atenuação 44.00 dB; ondulação 0.1019 dB" | 44.05 dB e 0.103 dB. A diferença está na 3ª casa decimal e vem da grade | `tests/test_p7_09.py` |
| V6 | §5 do DTA | "É proibido utilizar `scipy.signal.fir1`" | Essa função não existe no SciPy. O equivalente ao `fir1` do MATLAB é `scipy.signal.firwin`, que não é usado na solução, só como verificação nos testes | — |

## 3. Afirmações confirmadas

- **P7.9–P7.12:** janelas, Δω, cortes e comprimentos pela Tabela 7.1. Para a Hamming no P7.12, "Rp ≈ 0.04 dB" está correto para o M da tabela (35).
- **P7.12:** Blackman com M = 57; Rp = 0.00296 dB; As = 75.25 dB.
- **P7.13–P7.15:** A_s e β de Kaiser (4.0909; 3.3953; 5.6533); cortes; resposta ideal em degraus.
- **P7.15:** Exemplo 7.26 com **M = 49**, reproduzido de forma independente com `scipy.signal.remez`.
- **P8.12:** N = 6; Ω_c = 632.7668 rad/s; todos os coeficientes de A(s).
- **P8.13:** N = 9; Ω_c = 70621.0233 rad/s.
- **P8.14:** ε = 0.508847; N = 8; B = 1.535333×10⁶; a₀ = 1.722672×10⁶.
- **P8.15:** ε = 0.349311; N = 4; polos; biquads; ganho 0.94406.

## 4. Itens não verificáveis sem o livro

- Números de página citados no DTA (decisão D10).
- Valores do Exemplo 7.26 (pp. 367–368): informados pelo usuário e consistentes com a reprodução por `remez`, que dá 60.48, 59.94 e 61.94 dB contra os 60.61, 60.03 e 62.07 dB citados para M = 49, 50 e 51.
