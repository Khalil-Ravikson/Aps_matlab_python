"""Notebook gerado a partir do conteúdo compartilhado (content/<id>.py).

Mesmo texto da aplicação e do relatório. As células de código executam o
núcleo e desenham as figuras Matplotlib.
"""

from content import registry
from content.model import KINDS, fill, table_cell

from .common import SETUP, code, md

KIND_ICON = {"dado": "■", "calculado": "∑", "escolhido": "◆", "resultado": "●", "criterio": "✓"}


def _sym(s):
    return table_cell(s[6:-1]) if s.startswith(r"\text{") and s.endswith("}") else table_cell(f"${s}$")


def _table(values, ctx):
    rows = ["| Grandeza | Valor | Origem | Significado |", "|---|---|---|---|"]
    rows += [f"| {_sym(fill(v.symbol, ctx))} | {table_cell(fill(v.value, ctx))} | {KIND_ICON[v.kind]} {KINDS[v.kind]} | "
             f"{table_cell(fill(v.meaning, ctx))} |" for v in values]
    return "\n".join(rows)


def cells(ex_id: str, plot_code: str):
    mod = registry.load(ex_id)
    C, ctx, spec = mod.CONTENT, mod.context(), mod.objects()["spec"]
    S = lambda t: fill(t, ctx)  # noqa: E731
    out = [md(f"# {C.number} — {C.title}\n\n*{C.tagline}*\n\n"
              "> Notebook gerado de `content/" + ex_id + ".py`, a mesma fonte da aplicação DSP Filter Lab "
              "e do relatório PDF. Para alterar o texto, edite o conteúdo e rode `tools/run_notebook.sh " + ex_id + "`."),
           code(SETUP),
           md(f"## 1. Enunciado original\n\n> *{spec.statement}*\n\nFonte: {spec.source}."),
           md("## 2. O que está sendo pedido\n\n" + S(C.asked)),
           md("## 3. Dados e especificações\n\n" + _table(C.data_rows, ctx) + "\n\n" + S(C.data_note)),
           md("## 4. Fundamentação teórica\n\n" + S(C.theory)
              + "".join(f"\n\n#### Aprofundamento: {d.title}\n\n{S(d.body)}" for d in C.deepening))]
    eq = ["## 5. Equações e variáveis"]
    for k, e in enumerate(C.equations, 1):
        eq.append(f"**({k}) {e.name}**" + (f" · *{e.ref}*" if e.ref else "") + f"\n\n$$\n{e.latex}\n$$\n\n{e.explanation}")
        if e.variables:
            eq.append("| Símbolo | Significado |\n|---|---|\n" + "\n".join(
                f"| {table_cell('$' + v.symbol + '$')} | {table_cell(v.meaning)} |" for v in e.variables))
    out.append(md("\n\n".join(eq)))
    out.append(md("## 6. Resolução passo a passo"))
    for k, s in enumerate(C.steps, 1):
        txt = f"### Passo {k} — {S(s.title)}\n\n{S(s.reasoning)}"
        if s.equation:
            txt += f"\n\n$$\n{S(s.equation)}\n$$"
        if s.values:
            txt += "\n\n" + _table(s.values, ctx)
        if s.note:
            txt += "\n\n> ⚠️ " + S(s.note)
        out.append(md(txt))
    out.append(md("## 7. Implementação computacional\n\n" + S(C.implementation) + "\n\n| Função | O que faz |\n|---|---|\n"
                  + "\n".join(f"| `{r.function}` | {table_cell(r.role)} |" for r in C.code_roles)))
    out.append(code(f"from content import {ex_id} as content_mod\nobjs = content_mod.objects()\n" + plot_code))
    out.append(md("## 8. Gráficos e interpretação\n\n" + "\n\n".join(f"### {S(f.title)}\n\n{S(f.reading)}" for f in C.figures)))
    out.append(md("## 9. Verificação das especificações\n\n" + S(C.verification)))
    out.append(code("rep = objs.get('final_report') or objs.get('report')\nprint(rep)"))
    out.append(md("## 10. Conclusão técnica\n\n" + S(C.conclusion)))
    return out
