# DSP Filter Lab: projeto de filtros digitais FIR e IIR em Python

Resolução computacional e comentada de 12 exercícios de projeto de filtros digitais do livro:

> INGLE, V. K.; PROAKIS, J. G. *Digital Signal Processing Using MATLAB*. 3. ed. Stamford: Cengage Learning, 2012.

| Capítulo | Exercícios | Tema |
|---|---|---|
| 7 | P7.9 a P7.16 | Filtros FIR pelo método da janela (Hann, Hamming, Blackman e Kaiser) |
| 8 | P8.12 a P8.15 | Protótipos analógicos IIR (Butterworth e Chebyshev tipo I) |

O trabalho foi feito em Python (NumPy, SciPy e Matplotlib), e não em MATLAB. As fórmulas do livro foram implementadas explicitamente, e cada resultado é verificado medindo a resposta do filtro contra as especificações do enunciado.

O projeto pode ser usado de quatro formas:

1. **Aplicação web (DSP Filter Lab):** cada exercício apresentado como um capítulo de livro didático, com gráficos interativos.
2. **Relatório em PDF:** o mesmo conteúdo da aplicação, com sumário, equações e figuras.
3. **Notebooks Jupyter:** um por exercício, também gerados do mesmo conteúdo.
4. **Scripts de linha de comando:** um por exercício, que imprimem os resultados e gravam as figuras.

---

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como rodar](#como-rodar)
  - [Aplicação web](#1-aplicação-web-dsp-filter-lab)
  - [Testes](#2-testes)
  - [Scripts dos exercícios](#3-scripts-dos-exercícios)
  - [Relatório em PDF](#4-relatório-em-pdf)
  - [Notebooks](#5-notebooks)
  - [Regenerar tudo de uma vez](#6-regenerar-tudo-de-uma-vez)
  - [Google Colab](#7-google-colab)
- [Resultados](#resultados)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Metodologia](#metodologia)
- [Problemas comuns](#problemas-comuns)
- [Documentação complementar](#documentação-complementar)

---

## Requisitos

| Item | Versão | Para quê |
|---|---|---|
| Python | 3.10 ou superior (desenvolvido com 3.12.8) | Tudo |
| Git | qualquer | Clonar o repositório |
| Chromium ou Google Chrome | qualquer versão recente | Apenas para gerar o relatório em PDF |

Não é necessário ter MATLAB, LaTeX nem acesso à internet depois da instalação das dependências.

As bibliotecas Python estão listadas em:

- `requirements.txt`: núcleo de cálculo (NumPy, SciPy, Matplotlib);
- `requirements-app.txt`: núcleo + aplicação web e relatório (Streamlit, Plotly, latex2mathml, Markdown, pypdf);
- `pyproject.toml`: as mesmas dependências, organizadas nos grupos `app` e `dev`.

---

## Instalação

### Linux e macOS

```bash
git clone https://github.com/Khalil-Ravikson/Aps_matlab_python.git
cd Aps_matlab_python

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[app,dev]"
```

Com **pyenv**, o arquivo `.python-version` já seleciona a versão 3.12.8. Basta ter essa versão instalada (`pyenv install 3.12.8`) antes de criar o ambiente virtual.

### Windows (PowerShell)

```powershell
git clone https://github.com/Khalil-Ravikson/Aps_matlab_python.git
cd Aps_matlab_python

python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -e ".[app,dev]"
```

No Windows, troque `.venv/bin/` por `.venv\Scripts\` em todos os comandos deste README.

### Conferir a instalação

```bash
.venv/bin/python -m pytest -q
```

O resultado esperado é `308 passed`.

---

## Como rodar

Todos os comandos são executados **na raiz do projeto**.

### 1. Aplicação web (DSP Filter Lab)

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Abra **http://localhost:8501** no navegador. A barra lateral organiza o conteúdo em:

- **Início:** apresentação do projeto e introdução ao projeto de filtros digitais;
- **Capítulo 7 (FIR):** P7.9 a P7.16;
- **Capítulo 8 (IIR):** P8.12 a P8.15;
- **Análise:** comparação entre FIR e IIR, validação dos resultados, e referências e metodologia.

Cada exercício tem a mesma estrutura de 10 seções:

1. Enunciado original
2. O que está sendo pedido
3. Dados e especificações
4. Fundamentação teórica
5. Equações e variáveis
6. Resolução passo a passo
7. Implementação computacional
8. Gráficos e interpretação
9. Verificação das especificações
10. Conclusão técnica

No fim de cada página há o **modo Explore**, que permite alterar parâmetros (comprimento, janela, β, ordem) e ver o efeito. Ele é separado do resultado oficial e não o altera.

Na resolução, cada número traz um rótulo de origem: dado do enunciado, valor calculado, parâmetro escolhido, resultado obtido ou critério de validação.

Para encerrar a aplicação, use `Ctrl+C` no terminal.

### 2. Testes

```bash
.venv/bin/python -m pytest -q
```

A suíte cobre:

| Grupo | O que garante |
|---|---|
| Núcleo matemático | Janelas, respostas ideais e protótipos conferidos contra implementações independentes do NumPy e do SciPy |
| Exercícios | Cada filtro atende às especificações do enunciado |
| Regressão | Os resultados oficiais dos 12 exercícios continuam iguais aos de `tests/golden/results.json` |
| Conteúdo | Todo número do texto vem do cálculo; as afirmações do texto são verdadeiras; a errata concorda com os cálculos |
| Interface | Todas as páginas da aplicação abrem sem erro e contêm as 10 seções |
| Relatório | O documento é gerado sem marcadores pendentes e com as equações convertidas |
| Análise | As afirmações das seções de comparação e validação conferem com os cálculos |

### 3. Scripts dos exercícios

Cada exercício pode ser executado isoladamente:

```bash
.venv/bin/python -m exercises.p7_13
```

O script imprime os parâmetros calculados e a verificação das especificações, e grava em `results/p7_13/`:

- `overview.png`: os gráficos pedidos pelo enunciado;
- `metrics.json`: os valores calculados, a verificação e as versões das bibliotecas usadas.

Os identificadores disponíveis são: `p7_09`, `p7_10`, `p7_11`, `p7_12`, `p7_13`, `p7_14`, `p7_15`, `p7_16`, `p8_12`, `p8_13`, `p8_14`, `p8_15`.

### 4. Relatório em PDF

Requer o Chromium ou o Google Chrome instalado e acessível no PATH como `chromium`, `chromium-browser` ou `google-chrome`.

```bash
.venv/bin/python -m report.build_report
```

O arquivo é gravado em `report/out/relatorio.pdf` (cerca de 90 páginas; a geração leva de 1 a 2 minutos). Uma versão pronta também está em `docs/relatorio_dsp_filter_lab.pdf`.

Para gravar em outro lugar:

```bash
.venv/bin/python -m report.build_report --out caminho/relatorio.pdf
```

### 5. Notebooks

Os notebooks em `notebooks/` já estão executados e podem ser abertos diretamente no Jupyter, no VS Code ou no GitHub.

Para regenerar e reexecutar (Linux e macOS):

```bash
tools/run_notebook.sh            # todos
tools/run_notebook.sh p7_13      # apenas um
```

Os notebooks são gerados a partir de `content/`. Para alterar um texto, edite o conteúdo e regenere; não edite o `.ipynb` diretamente.

### 6. Regenerar tudo de uma vez

Linux e macOS:

```bash
tools/run_all.sh
```

O script roda os testes, executa os 12 scripts (atualizando `results/`) e reexecuta os 12 notebooks.

### 7. Google Colab

1. Envie o projeto para o Colab, como um `.zip` descompactado em `/content/Aps_matlab_python` ou clonando o repositório:

   ```python
   !git clone https://github.com/Khalil-Ravikson/Aps_matlab_python.git /content/Aps_matlab_python
   ```

2. Abra um notebook de `notebooks/`. Na primeira célula, ajuste `PROJECT_DIR` para `/content/Aps_matlab_python`, se necessário.
3. Execute as células em ordem. A primeira instala as dependências de `requirements.txt`.

A aplicação web e o relatório em PDF foram pensados para uso local, não no Colab.

---

## Resultados

Todos os 12 filtros atendem às especificações do enunciado.

| Exercício | Filtro | Método | Resultado |
|---|---|---|---|
| P7.9 | FIR passa-faixa | Janela de Hann | M = 43; Rp = 0,103 dB; As = 44,05 dB |
| P7.10 | FIR rejeita-faixa | Janela de Hamming | M = 67; Rp = 0,044 dB; As = 50,04 dB |
| P7.11 | FIR passa-faixa | Janela de Hamming | M = 67; Rp = 0,049 dB; As = 51,20 dB |
| P7.12 | FIR passa-altas | Janela de Blackman (escolhida) | M = 57; Rp = 0,00296 dB; As = 75,25 dB |
| P7.13 | FIR rejeita-faixa | Kaiser, comprimento mínimo | M = 57 |
| P7.14 | FIR passa-faixa | Kaiser, comprimento mínimo | M = 46 (Tipo 2) |
| P7.15 | FIR em escada | Kaiser, comparado com Parks-McClellan | Kaiser M = 72; Parks-McClellan M = 49 |
| P7.16 | FIR passa-faixa | Janela fixa de menor comprimento | Hamming, M = 61 |
| P8.12 | Butterworth analógico | Forma racional | N = 6; Ωc = 632,77 rad/s |
| P8.13 | Butterworth analógico | Forma em cascata | N = 9; Fc = 11,24 kHz |
| P8.14 | Chebyshev-I analógico | Forma racional | N = 8; ε = 0,5088 |
| P8.15 | Chebyshev-I analógico | Forma em cascata | N = 4; dois biquads; G = 0,94406 |

Alguns resultados divergem do documento de pesquisa usado como referência (por exemplo, P7.13, P7.14, P7.15, P7.16 e os ângulos da cascata do P8.13). Cada divergência está registrada, com a evidência correspondente, em `docs/errata_dta.md`.

---

## Estrutura do projeto

```
.
├── app/                    Aplicação web (Streamlit)
│   ├── streamlit_app.py    Ponto de entrada e navegação
│   ├── pages/              Uma página por seção e por exercício
│   ├── components/         Página padronizada de exercício, estilo e gráficos Plotly
│   └── explore/            Modo Explore de cada exercício
├── content/                Texto didático (fonte única para aplicação, PDF e notebooks)
├── src/dsp_av2/            Núcleo matemático
│   ├── specs.py            Enunciados e especificações transcritos
│   ├── fir/                Respostas ideais, janelas, projeto e análise FIR
│   ├── iir/                Protótipos, formas racional e em cascata, análise IIR
│   ├── validation.py       Verificação contra as especificações
│   └── plotting.py         Figuras Matplotlib
├── exercises/              Um script por exercício
├── notebooks/              Um notebook por exercício (já executados)
├── report/                 Gerador do relatório em PDF
├── results/                Figuras e métricas geradas pelos scripts
├── tests/                  Testes automatizados e resultados de referência (golden)
├── tools/                  Geração de notebooks e scripts de regeneração
├── docs/                   Plano, errata, relatório pronto e capturas de tela
├── pyproject.toml          Metadados e dependências
├── requirements.txt        Dependências do núcleo
└── requirements-app.txt    Dependências da aplicação e do relatório
```

O núcleo matemático (`src/dsp_av2`) não depende da aplicação. A aplicação, o relatório e os notebooks apenas leem os resultados calculados pelo núcleo.

---

## Metodologia

1. As especificações foram transcritas literalmente dos enunciados (`src/dsp_av2/specs.py`).
2. Os filtros foram projetados com as fórmulas do livro, implementadas explicitamente. Nos exercícios FIR, o equivalente ao `fir1` do MATLAB (`scipy.signal.firwin`) não é usado na solução; aparece apenas nos testes, como verificação independente.
3. Cada resultado é **medido**: a resposta em frequência é avaliada numa grade densa que inclui as frequências de borda exatas e comparada com cada limite do enunciado.
4. Quando o enunciado pede o comprimento ou a ordem mínimos, a resposta é confirmada por busca: todos os valores menores são testados e falham.
5. Nenhum resultado numérico é digitado no texto didático. Todos são preenchidos a partir do cálculo, e os testes garantem isso.

---

## Problemas comuns

**O Streamlit pede um e-mail na primeira execução.**
É um cadastro opcional. Pressione Enter com o campo vazio e a aplicação inicia.

**A porta 8501 já está em uso.**
Encerre a outra instância do Streamlit ou escolha outra porta:

```bash
.venv/bin/streamlit run app/streamlit_app.py --server.port 8502
```

**Alterei o código e a aplicação não mudou.**
Encerre a aplicação com `Ctrl+C` e inicie novamente.

**`Chromium não encontrado` ao gerar o PDF.**
Instale o Chromium ou o Google Chrome e garanta que o executável esteja no PATH com um dos nomes `chromium`, `chromium-browser` ou `google-chrome`. Os demais recursos (aplicação, testes, scripts e notebooks) funcionam sem ele.

**`ModuleNotFoundError: No module named 'dsp_av2'` ou `'exercises'`.**
Execute os comandos a partir da raiz do projeto e confirme que a instalação foi feita com `pip install -e ".[app,dev]"`.

---

## Documentação complementar

| Arquivo | Conteúdo |
|---|---|
| `docs/documento_transferencia_academica.md` | Pesquisa prévia: enunciados, fórmulas e resoluções de referência |
| `docs/errata_dta.md` | Divergências encontradas nessa pesquisa, com evidência |
| `docs/plano_implementacao.md` | Plano, decisões de projeto e verificação independente |
| `docs/plano_aplicacao.md` | Plano da aplicação web |
| `docs/relatorio_dsp_filter_lab.pdf` | Relatório completo em PDF |
| `docs/capturas_exercicios/`, `docs/previa_pilotos/` | Capturas de tela da aplicação e do relatório |

Os números de página do livro citados no projeto vêm da pesquisa prévia e não foram conferidos com o livro físico.
