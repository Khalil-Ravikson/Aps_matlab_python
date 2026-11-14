r"""Modelo do conteúdo didático compartilhado (aplicação, relatório PDF e notebooks).

Cada exercício é descrito por um :class:`ExerciseContent` com as 10 seções
padronizadas. O texto é Markdown com LaTeX (``$...$`` e ``$$...$$``) e pode
conter **marcadores** ``[[chave]]`` ou ``[[chave:formato]]``, preenchidos com
valores calculados pelo núcleo (``context()`` de cada módulo). Assim, nenhum
resultado numérico é digitado no texto: ele vem sempre da execução.

Formatos aceitos nos marcadores: qualquer especificação de ``format()``
(ex.: ``.4f``) e ``texN``, notação científica em LaTeX com N algarismos
significativos (ex.: ``[[wc2:tex5]]`` → ``4.9873\times10^{9}``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Tipos de valor: distinguem a origem de cada número (requisito acadêmico).
KINDS = {
    "dado": "Dado do enunciado",
    "calculado": "Valor calculado",
    "escolhido": "Parâmetro escolhido",
    "resultado": "Resultado obtido",
    "criterio": "Critério de validação",
}

_MARK = re.compile(r"\[\[(\w+)(?::([^\]]+))?\]\]")


def tex_sci(x: float, digits: int = 4) -> str:
    """Notação científica em LaTeX: 4.987e9 → ``4.987\\times10^{9}``."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    if -3 <= e <= 4:
        return f"{x:.{max(digits - 1 - e, 0)}f}"
    m = x / 10 ** e
    return rf"{m:.{digits - 1}f}\times10^{{{e}}}"


def fill(text: str, ctx: dict) -> str:
    """Substitui os marcadores ``[[chave:formato]]`` pelos valores de ``ctx``.

    Lança ``KeyError`` se a chave não existir: um texto nunca é exibido com
    marcador sem valor.
    """
    def repl(m: re.Match) -> str:
        key, fmt = m.group(1), m.group(2)
        if key not in ctx:
            raise KeyError(f"marcador sem valor no contexto: [[{key}]]")
        val = ctx[key]
        if fmt and fmt.startswith("tex"):
            return tex_sci(float(val), int(fmt[3:] or 4))
        return format(val, fmt) if fmt else str(val)

    return _MARK.sub(repl, text)


def markers(text: str) -> set[str]:
    return {m.group(1) for m in _MARK.finditer(text)}


@dataclass(frozen=True)
class Value:
    """Um valor citado numa etapa, com origem explícita."""

    symbol: str        # LaTeX, sem $ (ex.: r"\delta")
    value: str         # texto com marcadores (ex.: "[[delta]]")
    kind: str          # chave de KINDS
    meaning: str       # o que o valor representa
    unit: str = ""


@dataclass(frozen=True)
class Variable:
    symbol: str        # LaTeX, sem $
    meaning: str


@dataclass(frozen=True)
class Equation:
    """Equação usada na resolução, com o significado de cada variável."""

    latex: str
    name: str
    explanation: str
    variables: tuple[Variable, ...] = ()
    ref: str = ""      # referência (DTA §, livro)


@dataclass(frozen=True)
class Step:
    """Passo da resolução: raciocínio, conta com valores e resultado."""

    title: str
    reasoning: str                 # por que este passo e de onde vêm os números
    equation: str = ""             # LaTeX com marcadores (conta com números substituídos)
    values: tuple[Value, ...] = ()
    note: str = ""                 # observação destacada (ex.: falha da estimativa)


@dataclass(frozen=True)
class Figure:
    key: str                       # identificador usado pelos construtores de figura
    title: str
    reading: str                   # como ler o gráfico e o que ele mostra


@dataclass(frozen=True)
class Deepening:
    """Explicação complementar (expansível na aplicação)."""

    title: str
    body: str


@dataclass(frozen=True)
class CodeRole:
    function: str                  # nome qualificado
    role: str                      # o que faz, em linguagem natural


@dataclass(frozen=True)
class ExerciseContent:
    id: str                        # "p7_13"
    number: str                    # "P7.13"
    chapter: int                   # 7 ou 8
    title: str                     # título curto
    tagline: str                   # uma linha: tipo · método
    # 1. Enunciado: vem de specs (SPEC.statement / SPEC.source)
    asked: str                     # 2. O que está sendo pedido
    data_note: str                 # 3. Comentário sobre os dados (a tabela vem de specs)
    data_rows: tuple[Value, ...]   # 3. Dados do enunciado, reescritos em símbolos
    theory: str                    # 4. Fundamentação teórica
    deepening: tuple[Deepening, ...] = ()
    equations: tuple[Equation, ...] = ()        # 5
    steps: tuple[Step, ...] = ()                # 6
    implementation: str = ""                    # 7
    code_roles: tuple[CodeRole, ...] = ()
    figures: tuple[Figure, ...] = ()            # 8
    verification: str = ""                      # 9 (a tabela vem de validation)
    conclusion: str = ""                        # 10
    highlights: tuple[Value, ...] = ()          # resumo no topo da página (resultados oficiais)
    errata: tuple[str, ...] = ()                # IDs de docs/errata_dta.md tratados aqui
    result_keys: tuple[str, ...] = field(default=())  # chaves que nunca podem aparecer digitadas

    def texts(self) -> list[str]:
        """Todos os textos com marcadores (para testes de consistência)."""
        out = [self.asked, self.data_note, self.theory, self.implementation, self.verification,
               self.conclusion]
        out += [d.body for d in self.deepening]
        out += [v.value for v in self.data_rows] + [v.symbol for v in self.data_rows] + [v.meaning for v in self.data_rows]
        for e in self.equations:
            out += [e.latex, e.explanation]
        for s in self.steps:
            out += [s.title, s.reasoning, s.equation, s.note]
            out += [v.value for v in s.values] + [v.meaning for v in s.values] + [v.symbol for v in s.values]
        out += [f.reading for f in self.figures]
        out += [v.value for v in self.highlights] + [v.meaning for v in self.highlights]
        out += [v.symbol for v in self.highlights] + [f.title for f in self.figures]
        return out


def table_cell(text: str) -> str:
    r"""Torna um texto seguro para uma célula de tabela Markdown.

    Uma barra vertical ``|`` encerra a célula. Dentro de fórmulas ($...$), ela
    vira ``\vert``; fora delas, o caractere visualmente igual ``∣`` (U+2223).
    """
    parts = text.split("$")
    for i, p in enumerate(parts):
        parts[i] = p.replace("|", r"\vert{}") if i % 2 else p.replace("|", "∣")
    return "$".join(parts).replace("\n", " ")
