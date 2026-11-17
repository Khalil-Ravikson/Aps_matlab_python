r"""Conteúdo das seções de análise (Etapa 5): Comparação FIR × IIR, Validação e Metodologia.

Fonte única para a aplicação e o relatório. Cada seção é uma lista de blocos:
texto Markdown (já preenchido com valores calculados) ou figura (chave + dados).
Nenhum número é digitado: todos vêm do núcleo, dos resultados oficiais
(``tests/golden/results.json``) ou de cálculos feitos aqui e cobertos por testes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import signal

from dsp_av2.fir import analysis as fa
from dsp_av2.iir import analysis as ia
from dsp_av2.iir.prototypes import butter_order, cheb1_order
from dsp_av2.specs import P7_9, P8_12, P8_13, P8_14, P8_15

from . import common, registry
from .errata import items as errata_items
from .helpers import md_table
from .model import table_cell

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = ROOT / "tests" / "golden" / "results.json"
IIR_SPECS = (P8_12, P8_13, P8_14, P8_15)
FIR_IDS = ("p7_09", "p7_10", "p7_11", "p7_12", "p7_13", "p7_14", "p7_15", "p7_16")
IIR_IDS = ("p8_12", "p8_13", "p8_14", "p8_15")


@dataclass(frozen=True)
class Block:
    kind: str                 # "md" ou "fig"
    text: str = ""            # Markdown (kind="md") ou leitura do gráfico (kind="fig")
    key: str = ""             # identificador da figura
    title: str = ""           # título da figura
    data: dict = field(default_factory=dict)


# ── Comparação FIR × IIR ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def comparison_data() -> dict:
    # 1) Mesma especificação analógica, três aproximações (ordens mínimas)
    orders = []
    for sp in IIR_SPECS:
        nb = butter_order(sp.wp, sp.ws, sp.rp_db, sp.as_db)[0]
        nc = cheb1_order(sp.wp, sp.ws, sp.rp_db, sp.as_db)[0]
        ne = signal.ellipord(sp.wp, sp.ws, sp.rp_db, sp.as_db, analog=True)[0]
        orders.append((sp.exercise, sp.ws / sp.wp, nb, nc, ne))

    # 2) Especificação do P7.9 realizada por IIR digitais (ilustração complementar)
    spec = P7_9
    w = fa.frequency_grid(spec.edges, fa.DENSE_GRID_POINTS)
    pb = (w >= spec.passbands[0].lo * np.pi - 1e-12) & (w <= spec.passbands[0].hi * np.pi + 1e-12)
    sb = np.zeros_like(w, dtype=bool)
    for b in spec.stopbands:
        sb |= (w >= b.lo * np.pi - 1e-12) & (w <= b.hi * np.pi + 1e-12)
    wp = [spec.passbands[0].lo, spec.passbands[0].hi]
    ws = [spec.stopbands[0].hi, spec.stopbands[1].lo]
    digital = []
    curves_db, curves_gd = {}, {}
    for ftype, name, ordf in (("butter", "Butterworth", signal.buttord), ("cheby1", "Chebyshev-I", signal.cheb1ord),
                              ("ellip", "Elíptico", signal.ellipord)):
        N, Wn = ordf(wp, ws, spec.rp_db, spec.as_db, fs=2)
        sos = signal.iirfilter(N, Wn, rp=spec.rp_db, rs=spec.as_db, btype="bandpass", ftype=ftype, output="sos", fs=2)
        _, H = signal.sosfreqz(sos, worN=w)
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(np.abs(H) / np.abs(H).max())
        b, a = signal.sos2tf(sos)
        _, gd = signal.group_delay((b, a), w=w[pb])
        rp, as_ = -db[pb].min(), -db[sb].max()
        digital.append({"name": name, "order": 2 * N, "sections": len(sos), "mults": 5 * len(sos),
                        "rp": rp, "as": as_, "gd_min": gd.min(), "gd_max": gd.max(),
                        "ok": rp <= spec.rp_db + 1e-6 and as_ >= spec.as_db - 1e-6})
        curves_db[f"IIR {name} (ordem {2 * N})"] = db
        curves_gd[f"IIR {name} (ordem {2 * N})"] = gd

    from exercises import p7_09
    d = p7_09.design()
    H = fa.frequency_response(d.h, w)
    _, gd_fir = signal.group_delay((d.h, [1.0]), w=w[pb])
    fir = {"M": d.M, "mults_sym": (d.M + 1) // 2, "alpha": d.alpha, "gd_min": gd_fir.min(), "gd_max": gd_fir.max(),
           "rp": fa.measure_rp_as(d.h, spec.passbands, spec.stopbands, w)[0],
           "as": fa.measure_rp_as(d.h, spec.passbands, spec.stopbands, w)[1]}
    curves_db = {f"FIR Hann (M = {d.M})": fa.magnitude_db(H)} | curves_db
    curves_gd = {f"FIR Hann (M = {d.M})": gd_fir} | curves_gd

    # 3) Comprimentos dos FIR do Cap. 7 (custo por amostra)
    golden = json.loads(GOLDEN_FILE.read_text())
    fir_rows = []
    for ex_id in FIR_IDS:
        e = registry.BY_ID[ex_id]
        M = golden[e.number]["M"]
        fir_rows.append((e.number, e.short, e.method, M, (M + 1) // 2))
    return {"orders": orders, "digital": digital, "fir": fir, "fir_rows": fir_rows,
            "w_pb": w[pb] / np.pi, "w": w / np.pi, "curves_db": curves_db, "curves_gd": curves_gd}


def comparison_blocks() -> list[Block]:
    c = comparison_data()
    o = c["orders"]
    best = min(c["digital"], key=lambda r: r["mults"])
    fir = c["fir"]
    orders_tab = md_table(["Especificação", "$\\Omega_s/\\Omega_p$", "Butterworth", "Chebyshev-I", "Elíptico (ilustração)"],
                          [(ex, f"{r:g}", nb, nc, ne) for ex, r, nb, nc, ne in o])
    dig_tab = md_table(["Filtro", "Ordem", "Seções de 2ª ordem", "Multiplicações por amostra", "$R_p$ (dB)", "$A_s$ (dB)",
                        "Atraso de grupo na passagem (amostras)", "Atende?"],
                       [(f"FIR Hann (P7.9)", fir["M"] - 1, "—", f"{fir['mults_sym']} (com simetria)", f"{fir['rp']:.3f}",
                         f"{fir['as']:.2f}", f"{fir['gd_min']:.2f} a {fir['gd_max']:.2f} (constante)", "✓ sim")]
                       + [(f"IIR {r['name']}", r["order"], r["sections"], f"≈ {r['mults']}", f"{r['rp']:.3f}", f"{r['as']:.2f}",
                           f"{r['gd_min']:.2f} a {r['gd_max']:.2f}", "✓ sim" if r["ok"] else "✗ não") for r in c["digital"]])
    fir_tab = md_table(["Exercício", "Filtro", "Método", "$M$", "Multiplicações por amostra (com simetria)"], c["fir_rows"])
    p814 = [r for r in o if r[0] == "P8.14"][0]
    return [
        Block("md", r"""
Os dois capítulos resolvem problemas parecidos com ferramentas diferentes. Esta seção compara as duas famílias **com os números dos próprios exercícios**, e acrescenta uma ilustração calculada para tornar a comparação direta.

## Diferenças de princípio

| | FIR (Capítulo 7) | IIR (Capítulo 8) |
|---|---|---|
| Estrutura | só zeros (sem realimentação): $y(n) = \sum_k h(k)\,x(n-k)$ | polos e zeros (com realimentação) |
| Fase | **exatamente linear** quando $h(n)$ é simétrica | não linear |
| Atraso de grupo | constante: $\alpha = (M-1)/2$ amostras | varia com a frequência |
| Estabilidade | sempre estável | exige todos os polos no semiplano esquerdo (analógico) ou dentro do círculo unitário (digital) |
| Custo para a mesma seletividade | maior ($M$ costuma ser grande) | menor (ordem baixa) |
| Projeto nesta avaliação | método da janela e Kaiser | protótipos analógicos Butterworth e Chebyshev-I |

**Observação importante.** Os exercícios do Capítulo 8 projetam **protótipos analógicos**, o primeiro passo do projeto IIR. A conversão para filtro digital (invariância ao impulso, transformação bilinear) vem depois no livro e não faz parte dos exercícios. Por isso a comparação direta com os FIR, na terceira parte abaixo, usa uma **ilustração complementar** calculada com o SciPy, identificada como tal.
"""),
        Block("md", f"""
## 1. Dentro do IIR: Butterworth × Chebyshev-I

Para cada especificação dos exercícios do Capítulo 8, a ordem mínima de cada aproximação. As colunas Butterworth e Chebyshev-I usam as fórmulas do livro; a elíptica (`scipy.signal.ellipord`) entra só como referência.

{orders_tab}

Quanto mais estreita a transição ($\\Omega_s/\\Omega_p$ perto de 1), maior a vantagem do Chebyshev: no P8.14 ($\\Omega_s/\\Omega_p = {p814[1]:g}$) ele precisa de **{p814[3]}** contra **{p814[2]}** do Butterworth. O preço é a ondulação na passagem e um atraso de grupo mais irregular (ver o gráfico do P8.14).
"""),
        Block("fig", "Cada grupo é uma especificação do Capítulo 8. As barras mostram a ordem mínima de cada aproximação; a elíptica é só referência.",
              key="bars_orders", title="Ordem mínima por aproximação",
              data={"categories": [r[0] for r in o],
                    "series": {"Butterworth": [r[2] for r in o], "Chebyshev-I": [r[3] for r in o], "Elíptico (ilustração)": [r[4] for r in o]},
                    "ylabel": "ordem N"}),
        Block("md", f"""
## 2. Dentro do FIR: o custo de cada projeto

Num FIR de fase linear, cada amostra de saída custa cerca de $M$ multiplicações, ou $\\lceil M/2 \\rceil$ aproveitando a simetria dos coeficientes:

{fir_tab}

O maior custo é o do filtro em escada (P7.15) com Kaiser. O próprio P7.15 mostra que o Parks-McClellan atende à mesma especificação com {json.loads(GOLDEN_FILE.read_text())["P7.15"]["remez_M_min"]} coeficientes.
"""),
        Block("md", f"""
## 3. FIR × IIR na mesma especificação (ilustração complementar)

Para comparar as famílias na **mesma** especificação digital, a do P7.9 (passa-faixa, $R_p = {P7_9.rp_db:g}$ dB, $A_s = {P7_9.as_db:g}$ dB), foram projetados três IIR digitais com `scipy.signal.iirfilter` (transformação bilinear). Todos foram medidos com o mesmo critério dos exercícios (grade com as bordas exatas):

{dig_tab}

A contagem de multiplicações dos IIR considera 5 coeficientes por seção de 2ª ordem ($b_0, b_1, b_2, a_1, a_2$), uma estimativa usual. As rotinas do SciPy projetam esses IIR para atingir um dos limites **com igualdade** ($R_p$ ou $A_s$ exatos), por isso alguns valores coincidem com o limite; as diferenças medidas são menores que $10^{{-6}}$ dB, ruído numérico.

**Leitura.** O IIR {best["name"].lower()} atende à mesma especificação com cerca de {best["mults"]} multiplicações por amostra, contra {fir["mults_sym"]} do FIR, mas o seu atraso de grupo varia de {best["gd_min"]:.1f} a {best["gd_max"]:.1f} amostras **dentro da passagem**. O FIR atrasa todas as frequências exatamente {fir["alpha"]:g} amostras. Se a forma de onda importa (áudio, comunicações, medidas), a fase linear do FIR costuma compensar o custo maior.
"""),
        Block("fig", "Magnitude normalizada dos quatro projetos para a especificação do P7.9. Todos ficam acima de −Rp na passagem e abaixo de −As nas rejeições.",
              key="lines_mag", title="Magnitude: FIR de Hann × IIR digitais (especificação do P7.9)",
              data={"x": c["w"], "curves": c["curves_db"], "xlabel": "ω / π", "ylabel": "|H| (dB)", "ylim": (-80, 5),
                    "hlines": [(-P7_9.as_db, f"−As = −{P7_9.as_db:g} dB")], "vlines": list(P7_9.edges)}),
        Block("fig", "Atraso de grupo dentro da faixa de passagem. O FIR é uma reta: atraso constante, fase linear. Os IIR variam com a frequência e distorcem a forma de onda.",
              key="lines_gd", title="Atraso de grupo na passagem (amostras)",
              data={"x": c["w_pb"], "curves": c["curves_gd"], "xlabel": "ω / π", "ylabel": "atraso de grupo (amostras)"}),
        Block("md", """
## Conclusão da comparação

- **FIR:** fase exatamente linear e estabilidade garantida, ao custo de muitos coeficientes. O método da janela é simples, mas não aproveita tolerâncias diferentes entre as faixas (P7.13, P7.15).
- **IIR:** ordens muito menores para a mesma seletividade. O Chebyshev-I reduz a ordem em relação ao Butterworth trocando planura por ondulação, e o preço aparece no atraso de grupo.
- **Escolha:** depende do que a aplicação exige. Fase linear leva ao FIR; custo mínimo com fase não crítica leva ao IIR.
"""),
    ]


# ── Validação ───────────────────────────────────────────────────────────────

def _key_result(ex_id: str, g: dict) -> str:
    if ex_id.startswith("p7"):
        return f"M = {g['M']}"
    return f"N = {g['N']}"


@lru_cache(maxsize=1)
def validation_data() -> dict:
    from tools.make_golden import fir, iir
    import importlib
    golden = json.loads(GOLDEN_FILE.read_text())
    rows = []
    for e in registry.EXERCISES:
        mod = registry.load(e.id)
        o = mod.objects()
        rep = o.get("final_report") or o.get("report")
        ex = importlib.import_module(f"exercises.{e.id}")
        if e.chapter == 7:
            got = fir(ex)
            same = got["M"] == golden[e.number]["M"] and np.allclose(got["h"], golden[e.number]["h"], rtol=1e-9, atol=1e-12)
        else:
            form = "rational" if e.id in ("p8_12", "p8_14") else "cascade"
            got = iir(ex, form)
            same = got["N"] == golden[e.number]["N"] and np.allclose(np.asarray(got["poles"]), np.asarray(golden[e.number]["poles"]),
                                                                      rtol=1e-9, atol=1e-9)
        # Nos IIR, a perda em Ωp é igual ao limite por construção (Ω_c escolhido assim): não é folga.
        spec_checks = [c for c in rep.checks if "deve falhar" not in c.name and not c.name.startswith("Perda")]
        min_rel = min((c.margin / abs(c.limit)) for c in spec_checks if c.limit != 0)
        rows.append({"id": e.id, "number": e.number, "short": e.short, "method": e.method, "chapter": e.chapter,
                     "result": _key_result(e.id, golden[e.number]), "passed": rep.passed, "n_checks": len(rep.checks),
                     "min_rel_margin": min_rel, "golden": same})
    return {"rows": rows, "golden_commit": golden["_meta"]["commit"]}


def validation_blocks() -> list[Block]:
    v = validation_data()
    rows = v["rows"]
    n_ok = sum(r["passed"] for r in rows)
    n_gold = sum(r["golden"] for r in rows)
    tab = md_table(["Exercício", "Filtro", "Método", "Resultado", "Verificações", "Menor folga relativa", "Especificação", "Igual ao oficial"],
                   [(r["number"], r["short"], r["method"], r["result"], r["n_checks"], f"{100 * r['min_rel_margin']:.2f}%",
                     "✓ atende" if r["passed"] else "✗ não atende", "✓" if r["golden"] else "✗") for r in rows])
    err = errata_items()
    etab = md_table(["Item", "Exercício", "O documento de referência afirmava", "Correção"],
                    [(i.id, i.exercise, table_cell(i.claim), table_cell(i.correction)) for i in err.values()])
    n_e = sum(1 for i in err.values() if i.id.startswith("E"))
    n_v = sum(1 for i in err.values() if i.id.startswith("V"))
    tight = sorted(rows, key=lambda r: r["min_rel_margin"])[:3]
    return [
        Block("md", f"""
Esta seção reúne, num só lugar, as evidências de que os resultados estão corretos.

## 1. Situação dos 12 exercícios

Calculada agora, a partir do núcleo matemático:

{tab}

- **Especificação:** {n_ok} de {len(rows)} exercícios atendem a todos os limites do enunciado.
- **Igual ao oficial:** {n_gold} de {len(rows)} reproduzem exatamente os resultados oficiais congelados no commit `{v["golden_commit"]}` (`tests/golden/results.json`).
- **Menor folga relativa:** a folga do critério mais apertado de cada exercício, dividida pelo limite. Nos IIR, a perda em $\\Omega_p$ não entra nessa conta, porque é igual ao limite por construção ($\\Omega_c$ é escolhido assim). As menores são {", ".join(f"{t['number']} ({100 * t['min_rel_margin']:.2f}%)" for t in tight)}: são projetos de comprimento ou ordem mínimos, em que folga pequena é esperada.
"""),
        Block("md", """
## 2. Como cada resultado foi verificado

1. **Contra o enunciado, não contra valores de referência.** Cada critério compara um valor **medido** com o limite do enunciado. Os números do documento de pesquisa só servem de referência depois de confirmados de forma independente.
2. **Grades de frequência com as bordas exatas.** O pior ponto costuma ser a própria borda (P7.10, P7.11). Uma grade que pula as bordas superestima a atenuação: é a origem das divergências V1 e V2.
3. **Minimalidade por busca.** Quando o enunciado pede o mínimo, todos os valores menores são testados e falham (P7.13–P7.16). Nos IIR, o mesmo projeto com ordem $N - 1$ não atende.
4. **Verificação independente.** As janelas e respostas ideais são conferidas com `numpy` e `scipy.signal.firwin`; os protótipos, com `scipy.signal.butter` e `cheby1`; o Exemplo 7.26, com `scipy.signal.remez`; o atraso de grupo analítico, com a derivada numérica da fase; a resposta ao impulso, com a inversa de Laplace por resíduos.
5. **Consistência do material didático.** Todo número do texto vem do cálculo; as afirmações qualitativas ("a estimativa falha", "o pior ponto é a borda") são testadas; cada item da errata concorda com o cálculo e aparece na página do exercício.
"""),
        Block("md", """
## 3. Testes automatizados

Para executar: `python -m pytest -q` na raiz do projeto.

| Grupo | Arquivos | O que garante |
|---|---|---|
| Núcleo matemático | `test_fir_*.py`, `test_iir_*.py` | Janelas, respostas ideais, protótipos, atraso de grupo e $h(t)$ conferidos com implementações independentes |
| Exercícios | `test_p7_*.py`, `test_p8_*.py` | Cada filtro atende ao enunciado; buscas de mínimo; observações complementares |
| Regressão | `test_regression.py` | Os 12 resultados são idênticos aos oficiais congelados |
| Conteúdo | `test_content.py` | Marcadores preenchidos, nenhum resultado digitado, afirmações verdadeiras, errata coerente |
| Interface | `test_app_pages.py` | Todas as páginas abrem sem erro e têm as 10 seções; o modo Explore não altera o resultado oficial |
| Relatório | `test_report.py` | Documento sem marcadores pendentes, equações convertidas e sumário paginado |
"""),
        Block("md", f"""
## 4. Correções ao documento de referência

A pesquisa prévia (DTA) foi conferida item por item. Foram encontrados **{n_e} erros que mudam a resposta** (itens E) e **{n_v} erros de valor ou de raciocínio** (itens V) que não mudam a resposta final. Cada item tem evidência em `docs/errata_dta.md` e é mostrado na página do exercício correspondente.

{etab}
"""),
    ]


# ── Metodologia e referências ───────────────────────────────────────────────

DECISIONS = (
    ("D1", "Janela de Hann com zeros nas extremidades, $w(n) = 0.5 - 0.5\\cos(2\\pi n/(M-1))$", "Definição do DTA e do `hann(M)` do MATLAB"),
    ("D2", "\"Comprimento mínimo\" é o menor $M$ **verificado** por medição, e não o da fórmula", "As fórmulas são empíricas; no P7.13 a estimativa viola a especificação"),
    ("D3", "Estimativa de Kaiser $M = \\lceil (A_s-7.95)/(2.285\\Delta\\omega) \\rceil + 1$", "Fórmula do DTA; a busca verificada torna irrelevante a variante usada no código do livro"),
    ("D4", "$M$ par (Tipo 2) é admitido quando o filtro não precisa de ganho em $\\omega = \\pi$", "O zero obrigatório do Tipo 2 cai na rejeição (P7.14, P7.15)"),
    ("D5", "No P7.16, a janela de menor comprimento é a de menor $M$ medido", "Responde diretamente ao enunciado; a Tabela 7.1 é conservadora"),
    ("D6", "Exemplo 7.26: $M = 49$ (informado das pp. 367–368)", "Reproduzido de forma independente com `scipy.signal.remez`"),
    ("D7", "No filtro em escada, $A_s$ de Kaiser pela menor tolerância (60 dB)", "Critério do livro; o ajuste de $A_s$ é discutido como observação"),
    ("D8", "Grades de frequência sempre com as bordas exatas", "O pior ponto costuma ser a borda"),
    ("D9", "Gráficos IIR em Hz quando o enunciado usa Hz; fase em unidades de $\\pi$", "Coerência com o enunciado e com a `freqs_m` do livro"),
    ("D10", "Números de página citados conforme o DTA", "Não conferidos com o livro físico"),
)


def methodology_blocks() -> list[Block]:
    import platform

    import matplotlib
    import scipy

    dec = md_table(["Decisão", "O que foi decidido", "Justificativa"], DECISIONS)
    refs = "\n".join(f"{k}. {r}  \n   *{n}*" for k, (r, n) in enumerate(common.REFERENCES, 1))
    env = md_table(["Componente", "Versão"], [("Python", platform.python_version()), ("NumPy", np.__version__),
                                              ("SciPy", scipy.__version__), ("Matplotlib", matplotlib.__version__)])
    return [
        Block("md", common.METHOD_SUMMARY),
        Block("md", f"""
## Decisões de projeto

Pontos em que o enunciado ou a referência admitiam mais de uma leitura. Cada decisão foi registrada em `docs/plano_implementacao.md`.

{dec}
"""),
        Block("md", f"""
## Ambiente de cálculo

Os resultados oficiais foram produzidos e verificados com:

{env}

Cada script de exercício grava as versões usadas em `results/<exercício>/metrics.json`.
"""),
        Block("md", f"""
## Referências

{refs}

{common.DTA_NOTE}

## Limitações

- Os enunciados, as fórmulas e as páginas citadas foram transcritos pela pesquisa prévia (DTA). O livro físico não foi consultado diretamente nesta implementação.
- Os campos de identificação marcados com {common.A_PREENCHER} dependem de informações ainda não fornecidas.
- O Capítulo 8 foi resolvido até os protótipos analógicos, como pedem os enunciados; a conversão para filtros digitais não faz parte dos exercícios.
"""),
    ]
