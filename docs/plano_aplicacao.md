# Plano — DSP Filter Lab (aplicação web educacional)

**Data:** 2026-11-14 · **Status:** proposta, aguardando aprovação (nenhum código da aplicação escrito)
**Base:** commit `70c8e0c` (12 exercícios, 181 testes passando)

---

## 1. Inventário do repositório atual

| Componente | Conteúdo | Uso na aplicação |
|---|---|---|
| `src/dsp_av2/specs.py` | Enunciados literais e especificações dos 12 exercícios (`FIRSpec`, `FIRToleranceSpec`, `AnalogLowpassSpec`), com fonte | "Enunciado" e "Dados do enunciado", **sem copiar os valores** |
| `src/dsp_av2/fir/` | Respostas ideais, janelas, Kaiser, busca do M mínimo, Hr(ω), Rp/As | Cálculo FIR, reutilizado sem alteração |
| `src/dsp_av2/iir/` | Protótipos (fórmulas do livro), formas racional/cascata, atraso de grupo, h(t) | Cálculo IIR, reutilizado sem alteração |
| `src/dsp_av2/validation.py` | Relatórios com cada verificação: medido × limite × folga | Indicadores "atende / não atende" |
| `src/dsp_av2/plotting.py` | Figuras Matplotlib (estáticas) | Reaproveitado no relatório PDF |
| `exercises/pX_YY.py` | `design()`, `validate()`, `plot()` e funções específicas (`search()`, `evaluate_candidates()`, `remez_search()`, `cascade_form()`, …) | Ponto de entrada de cada página |
| `tools/nb/pX_YY.py` | Texto didático dos notebooks: método, equações, interpretação, conclusões | **Fonte do texto das páginas** (ver §3.3) |
| `results/` | `overview.png`, `metrics.json`, `stdout.txt` | Referência para os testes de regressão (golden) |
| `docs/errata_dta.md` | 6 erros que mudam a resposta e 6 de valor ou raciocínio, com evidência | Seção "Validação" e destaques nas páginas |
| `tests/` | 181 testes | Mantidos; acrescentam-se testes de regressão e da interface |

Ferramentas disponíveis na máquina: Python 3.12.8 (`.venv`) e Chromium 156. **Não** estão instalados: Streamlit, Plotly, LaTeX, pandoc, typst.

---

## 2. Arquitetura proposta

```
app/
├── streamlit_app.py          # entrada: st.navigation com seções na barra lateral
├── pages/
│   ├── apresentacao.py       # 1. Apresentação do projeto
│   ├── introducao.py         # 2. Introdução ao projeto de filtros digitais
│   ├── exercicio.py          # 3–4. Página única, parametrizada pelo exercício (12 rotas)
│   ├── fir_vs_iir.py         # 5. Comparação FIR × IIR
│   ├── validacao.py          # 6. Validação: painel dos 12 + errata + testes
│   └── referencias.py        # 7. Referências e metodologia
├── content/
│   ├── p7_09.py … p8_15.py   # texto didático das 10 seções de cada exercício
│   └── common.py             # textos de introdução, comparação e metodologia
├── components/
│   ├── layout.py             # cabeçalho do exercício, blocos das 10 seções, expansores
│   ├── badges.py             # rótulos "dado do enunciado" / "calculado" / "escolhido" / "resultado" / "critério"
│   ├── tables.py             # tabelas de especificação e de verificação
│   └── figures.py            # figuras Plotly interativas, desenhadas a partir dos arrays do núcleo
├── theme.css                 # identidade visual
└── .streamlit/config.toml    # tema (cores, fonte)
report/
├── build_report.py           # mesmo conteúdo → HTML → PDF (Chromium headless)
└── template.html / report.css
tests/
├── golden/results.json       # números de referência do commit 70c8e0c
├── test_regression.py        # núcleo atual = golden (a interface não altera cálculos)
├── test_app_pages.py         # cada página e cada exercício renderiza sem erro (streamlit.testing.AppTest)
└── test_content.py           # todo número citado no texto vem do cálculo, e todo item da errata aparece
```

### 2.1 Navegação

`st.navigation` com grupos na barra lateral:

- **Início**: Apresentação · Introdução
- **Capítulo 7 — FIR**: P7.9 … P7.16, cada um com rótulo curto (ex.: "P7.13 · Rejeita-faixa · Kaiser")
- **Capítulo 8 — IIR**: P8.12 … P8.15
- **Análise**: FIR × IIR · Validação · Referências e metodologia

Cada exercício tem URL própria (ex.: `/p7_13`) para o professor abrir direto.

### 2.2 Página de exercício: as 10 seções padronizadas

| # | Seção | Origem do conteúdo |
|---|---|---|
| 1 | Enunciado original | `SPEC.statement` (inglês, como no livro), com referência de página |
| 2 | O que está sendo pedido | `content/pX_YY.py`, em linguagem simples |
| 3 | Dados e especificações | Tabela gerada de `SPEC` (rótulo **dado do enunciado**) |
| 4 | Fundamentação teórica | `content/`, com expansores para aprofundamento |
| 5 | Equações e variáveis | `st.latex` mais uma tabela "símbolo · significado · valor" |
| 6 | Resolução passo a passo | Passos numerados, cada valor com rótulo **calculado** ou **escolhido** e vindo da execução |
| 7 | Implementação computacional | Fluxo em linguagem natural ("função X faz Y"). O código fica num expansor opcional, porque o professor não precisa lê-lo |
| 8 | Gráficos e interpretação | Plotly interativo, com zoom e valores ao passar o mouse, e texto de leitura de cada gráfico |
| 9 | Verificação | Tabela medido × limite × folga, com indicador visual (rótulo **critério de validação**) |
| 10 | Conclusão técnica | `content/`, mais um destaque quando houver item da errata |

**Regra central:** nenhum número de resultado é digitado no texto. O conteúdo usa marcadores preenchidos pela execução de `exercises.pX_YY` (ex.: `{M}`, `{As:.2f}`). O teste `test_content.py` garante isso.

### 2.3 Identidade visual

- Visual de livro técnico: tipografia serifada nos títulos, sem-serifa no texto, largura de leitura controlada, cores FIR e IIR distintas e consistentes.
- Cinco rótulos visuais fixos, com cor, ícone e texto (nunca só cor): dado do enunciado · valor calculado · parâmetro escolhido · resultado obtido · critério de validação.
- Indicadores de especificação: "✓ Atende" ou "✗ Não atende", com a folga numérica ao lado.
- Responsivo pelo layout nativo do Streamlit, testado em largura de celular.

---

## 3. Decisões que dependem de você

### 3.1 Gráficos: Plotly na aplicação e Matplotlib no PDF (recomendado)

O Plotly dá zoom e leitura de valores ao passar o mouse, útil para conferir Rp e As na tela. O PDF reusa as figuras Matplotlib já validadas. Os dois desenham **os mesmos arrays** vindos do núcleo.

### 3.2 PDF: HTML + Chromium headless (recomendado)

Não há LaTeX na máquina. A proposta é gerar o relatório em HTML a partir do mesmo conteúdo da aplicação e imprimi-lo em PDF com o Chromium, que já está instalado. As equações saem em MathML (biblioteca `latex2mathml`), sem depender de internet. A alternativa seria instalar uma distribuição TeX (centenas de MB) e manter o conteúdo duplicado.

### 3.3 Uma só fonte de texto

Hoje o texto didático está nos geradores de notebook (`tools/nb/`). Proponho mover esse texto para `app/content/`, para que aplicação, PDF e notebooks usem a mesma fonte. Os notebooks continuam existindo e funcionando, como exigido. A alternativa é deixar os notebooks como estão e aceitar duas cópias do texto.

### 3.4 Interatividade

Proponho que os resultados oficiais sejam fixos (os do projeto) e que cada exercício tenha um expansor **"Explore"** opcional: por exemplo, mudar M ou a janela e ver o efeito, com aviso claro de que é exploração e não a resposta. Isso ajuda a ensinar o raciocínio, por exemplo por que M = 55 falha no P7.13.

---

## 4. Informações que faltam para uma explicação acadêmica completa

| # | Falta | Por quê | O que faço sem ela |
|---|---|---|---|
| F1 | **Acesso ao livro** (PDF ou fotos das páginas 321–466 relevantes) | Você pede para conferir enunciados, Tabela 7.1, Exemplo 7.26 e a errata com o material original. Eu não tenho o livro: hoje tudo vem do DTA e do que você me informou | Marco tudo como "conforme o DTA" ou "informado pelo usuário" e não afirmo ter conferido com o livro |
| F2 | Dados da página de apresentação: nome(s) do(s) autor(es), instituição, curso, disciplina, professor, semestre | Não posso inventar | Deixo campos claramente marcados como "a preencher" |
| F3 | Formato exigido do relatório: ABNT? idioma? número de páginas? capa? | Define a estrutura do PDF | Relatório técnico em português, com capa, sumário, metodologia, os 12 exercícios, errata e referências |
| F4 | Outras referências que a disciplina usa (ex.: Oppenheim & Schafer) | Não posso acrescentar referências por conta própria | Uso só Ingle & Proakis (fonte do DTA) e a documentação oficial das bibliotecas usadas (NumPy, SciPy, Matplotlib, Streamlit), que posso conferir |
| F5 | Pontos não verificados registrados no plano anterior (D3: a fórmula de Kaiser no código do livro; D10: números de página) | Dependem do livro | Continuam marcados como não verificados |

---

## 5. Plano incremental

| Etapa | Entrega | Critério de aceite |
|---|---|---|
| 0 | Dependências (`streamlit`, `plotly`, `latex2mathml`) e `tests/golden/results.json` + `test_regression.py` | Golden gerado do commit atual; 181 + novos testes passando |
| 1 | Esqueleto: navegação, tema, Apresentação, Introdução | App abre localmente; AppTest renderiza todas as rotas |
| 2 | Componentes da página de exercício + **piloto FIR (P7.13)** | As 10 seções completas; números do texto = cálculo |
| 3 | **Piloto IIR (P8.13)** | Idem, com cascata e errata θₖ em destaque |
| ⏸ | Revisão sua dos dois pilotos | Aprovação do padrão visual e didático |
| 4 | Os outros 10 exercícios | Cada um validado e testado ao entrar |
| 5 | FIR × IIR, Validação, Referências e metodologia | Painel dos 12 lido do núcleo; errata integrada |
| 6 | Relatório PDF | PDF gerado do mesmo conteúdo, com figuras e equações legíveis |
| 7 | README, `requirements.txt`, execução completa dos testes e validação dos 12 exercícios | `pytest` verde; app e PDF regeneráveis com um comando |

Pilotos escolhidos: **P7.13** (o caso mais didático, em que a fórmula falha e a busca corrige) e **P8.13** (cascata com correção da errata).
