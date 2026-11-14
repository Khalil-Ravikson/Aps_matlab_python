r"""Conteúdo comum: apresentação, introdução, legenda dos tipos de valor e referências.

Regra: nenhuma informação institucional ou bibliográfica é inventada. Campos
que dependem do usuário ficam explicitamente marcados como ``A_PREENCHER``.
"""

from __future__ import annotations

A_PREENCHER = "⟨a preencher⟩"

PROJECT = {
    "app_name": "DSP Filter Lab",
    "subtitle": "Projeto de filtros digitais FIR e IIR: resolução comentada de 12 exercícios",
    "book": "INGLE, V. K.; PROAKIS, J. G. Digital Signal Processing Using MATLAB. 3. ed. Stamford: Cengage Learning, 2012.",
    "exercises": "P7.9–P7.16 (Capítulo 7) e P8.12–P8.15 (Capítulo 8)",
    # Dados institucionais: fornecidos pelo usuário (não inventar)
    "authors": A_PREENCHER,
    "institution": A_PREENCHER,
    "course": "Engenharia da Computação",  # conforme o DTA §1
    "discipline": "Processamento Digital de Sinais",  # conforme o DTA §1
    "professor": A_PREENCHER,
    "term": A_PREENCHER,
}

PRESENTATION = r"""
Este material apresenta a resolução computacional de **12 exercícios** de projeto de filtros digitais do livro de Ingle & Proakis: oito de filtros **FIR** (Capítulo 7) e quatro de protótipos analógicos **IIR** (Capítulo 8).

Cada exercício segue a mesma estrutura, do enunciado original à conclusão técnica. O objetivo não é só mostrar resultados: é **demonstrar o raciocínio** que leva a cada valor, de onde ele vem e como foi verificado.

**Como usar.** Escolha um exercício na barra lateral. Cada página pode ser lida de cima para baixo, como um capítulo de livro, e as explicações complementares ficam em seções expansíveis. Nenhum conhecimento de programação é necessário: o código-fonte aparece apenas como material opcional.
"""

METHOD_SUMMARY = r"""
**Como os resultados foram obtidos e validados**

1. As especificações foram transcritas literalmente dos enunciados.
2. Os filtros foram projetados com as fórmulas do livro, implementadas explicitamente (sem `fir1`/`firwin` no caso FIR).
3. Cada resultado foi **medido**: a resposta em frequência é avaliada numa grade densa que inclui as frequências de borda e comparada com cada limite do enunciado.
4. Quando o enunciado pede o comprimento ou a ordem **mínimos**, a resposta foi confirmada por busca: os valores menores foram testados e falharam.
5. Rotinas independentes do SciPy conferem os cálculos em testes automatizados.
6. As divergências encontradas no documento de pesquisa de referência estão registradas com evidência na errata.
"""

KIND_LEGEND = r"""
Ao longo das resoluções, cada número traz um rótulo que indica a sua **origem**:
"""

INTRODUCTION = r"""
## O que é um filtro digital

Um filtro digital é um sistema que transforma uma sequência de amostras $x(n)$ em outra, $y(n)$, alterando seletivamente o seu conteúdo em frequência: deixa passar algumas faixas e atenua outras. Para sistemas lineares e invariantes no tempo, o comportamento é descrito pela **resposta ao impulso** $h(n)$ ou, equivalentemente, pela **resposta em frequência**

$$H(e^{j\omega}) = \sum_{n} h(n)\,e^{-j\omega n}, \qquad 0 \le \omega \le \pi,$$

em que $\omega$ é a frequência digital, em radianos por amostra. $\omega = \pi$ corresponde à metade da taxa de amostragem.

## FIR e IIR

| | **FIR** (*Finite Impulse Response*) | **IIR** (*Infinite Impulse Response*) |
|---|---|---|
| Resposta ao impulso | finita: $M$ coeficientes | infinita (realimentação) |
| Fase | pode ser **exatamente linear** | não linear |
| Estabilidade | sempre estável | depende da posição dos polos |
| Ordem para a mesma seletividade | maior | menor |
| Projeto neste trabalho | método da janela (Cap. 7) | protótipos analógicos (Cap. 8) |

## Como se especifica um filtro

Um filtro real não tem degraus abruptos. A especificação define **faixas** e **tolerâncias**:

- **Faixa de passagem**: o ganho deve ficar perto de 1. O desvio máximo é dado pela ondulação $R_p$ (dB) ou pela tolerância linear $\delta_1$.
- **Faixa de rejeição**: o ganho deve ficar perto de 0. O mínimo de atenuação é $A_s$ (dB) ou a tolerância linear $\delta_2$.
- **Faixa de transição**: entre as duas, sem exigência. Sua largura $\Delta\omega$ determina, em grande parte, o custo do filtro.

As formas em decibéis e em valores lineares se relacionam por

$$R_p = -20\log_{10}\frac{1-\delta_1}{1+\delta_1}, \qquad A_s = -20\log_{10}\delta_2.$$

## Do enunciado ao filtro: o fluxo comum a todos os exercícios

1. **Ler a especificação**: bordas, tolerâncias e método imposto.
2. **Escolher a estrutura**: janela e comprimento $M$ (FIR), ou aproximação e ordem $N$ (IIR).
3. **Calcular** pelas fórmulas de projeto.
4. **Medir** a resposta obtida e compará-la com cada limite.
5. **Ajustar**, se a medição mostrar que a estimativa não basta, e concluir.

O passo 4 é essencial: muitas fórmulas de projeto são **empíricas** e dão estimativas. Os exercícios mostram casos em que a estimativa falha e a medição corrige o resultado.
"""

# Referências: somente obras verificáveis. O livro-texto é a referência principal
# (citado no DTA); as bibliotecas de software são citadas pelas publicações oficiais
# indicadas pelos próprios projetos.
REFERENCES = (
    ("INGLE, V. K.; PROAKIS, J. G. *Digital Signal Processing Using MATLAB*. 3. ed. Stamford: Cengage Learning, 2012.",
     "Referência principal: enunciados, métodos e fórmulas (Capítulos 7 e 8)."),
    ("HARRIS, C. R. et al. Array programming with NumPy. *Nature*, v. 585, p. 357–362, 2020.",
     "Biblioteca numérica usada nos cálculos."),
    ("VIRTANEN, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, v. 17, p. 261–272, 2020.",
     "Rotinas de processamento de sinais (verificação independente e análise)."),
    ("HUNTER, J. D. Matplotlib: a 2D graphics environment. *Computing in Science & Engineering*, v. 9, n. 3, p. 90–95, 2007.",
     "Figuras do relatório."),
)

DTA_NOTE = (
    "Documento de Transferência Acadêmica (DTA), `docs/documento_transferencia_academica.md`: "
    "pesquisa prévia que transcreve os enunciados e as fórmulas do livro. As divergências "
    "encontradas estão em `docs/errata_dta.md`. Os números de página citados vêm do DTA e "
    "**não foram conferidos** com o livro físico."
)
