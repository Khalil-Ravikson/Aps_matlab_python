# DSP Avaliação 2

Resolução computacional dos exercícios **P7.9–P7.16** (FIR) e **P8.12–P8.15** (IIR) de
Ingle & Proakis, *Digital Signal Processing Using MATLAB*, 3ª ed.

- Especificações: [`docs/documento_transferencia_academica.md`](docs/documento_transferencia_academica.md)
- Plano, decisões e verificação independente: [`docs/plano_implementacao.md`](docs/plano_implementacao.md)

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/dsp_av2/specs.py` | Dados literais dos enunciados |
| `src/dsp_av2/fir/` | Cálculo FIR: resposta ideal, janelas (fixas e Kaiser), projeto, busca do M mínimo, análise |
| `src/dsp_av2/iir/` | Cálculo IIR: protótipos (fórmulas do livro), formas racional/cascata, análise |
| `src/dsp_av2/validation.py` | Confronto com a especificação do enunciado |
| `src/dsp_av2/plotting.py` | Somente gráficos |
| `exercises/pX_YY.py` | Um script por exercício: `design()` → `validate()` → `plot()` |
| `notebooks/` | Narrativa acadêmica por exercício (gerados por `tools/build_notebooks.py`) |
| `results/pX_YY/` | Figuras e `metrics.json` regeneráveis |
| `tests/` | Testes unitários e por exercício |
| `tools/` | Geração/execução de notebooks e regeneração completa |
| `docs/errata_dta.md` | Divergências encontradas no DTA, com evidência |

## Uso local (Linux + pyenv)

```bash
pyenv local 3.12.8
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m exercises.p7_09
```

## Google Colab

Envie o projeto (zip ou Google Drive) para `/content/dsp-avaliacao-2` e abra um notebook de
`notebooks/`. A primeira célula detecta o Colab, instala `requirements.txt` e ajusta o `sys.path`.

## Resultados

Todos os 12 exercícios atendem à especificação do enunciado (verificação em `tests/`; 181 testes).

| Ex. | Filtro | Método | Resultado | Observação |
|---|---|---|---|---|
| P7.9 | FIR passa-faixa | Hann | M = 43; Rp = 0.103 dB; As = 44.05 dB | — |
| P7.10 | FIR rejeita-faixa | Hamming | M = 67; Rp = 0.044 dB; As = 50.04 dB | folga de 0.04 dB; o pior ponto é a borda |
| P7.11 | FIR passa-faixa | Hamming | M = 67; Rp = 0.049 dB; As = 51.20 dB | M = 66 também atende |
| P7.12 | FIR passa-altas | Blackman (escolhida) | M = 57; Rp = 0.00296 dB; As = 75.25 dB | Hamming falha em Rp para qualquer M |
| P7.13 | FIR rejeita-faixa | Kaiser, β = 4.0909 | **M mínimo = 57** | a estimativa de 55 (DTA) viola δ₂ |
| P7.14 | FIR passa-faixa | Kaiser, β = 3.3953 | **M mínimo = 46** (Tipo 2) | DTA: 47 (forçava ímpar) |
| P7.15 | FIR em escada | Kaiser, β = 5.6533 | **M mínimo = 72** | Ex. 7.26 (Parks-McClellan): 49, reproduzido com `remez` |
| P7.16 | FIR passa-faixa | menor janela fixa | **Hamming, M = 61** | pela Tabela 7.1 seria Hann 63 |
| P8.12 | Butterworth | racional | N = 6; Ωc = 632.77 rad/s | coeficientes = DTA |
| P8.13 | Butterworth (Hz) | cascata | N = 9; Fc = 11.24 kHz; θₖ = kπ/9 | DTA errava θₖ |
| P8.14 | Chebyshev-I | racional | N = 8; ε = 0.5088; B = 1.535×10⁶ | — |
| P8.15 | Chebyshev-I (Hz) | cascata | N = 4; 2 biquads; G = 0.94406 | — |

Divergências em relação ao DTA: [`docs/errata_dta.md`](docs/errata_dta.md).

## Regenerar tudo

```bash
tools/run_all.sh
```

O script roda os testes, regrava `results/` e reexecuta todos os notebooks. Para um único notebook: `tools/run_notebook.sh p7_13`.
