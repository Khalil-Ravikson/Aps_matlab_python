"""Interface: cada página renderiza sem erro e sem marcadores ou Markdown crus."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
EXERCISES = ["p7_09", "p7_10", "p7_11", "p7_12", "p7_13", "p7_14", "p7_15", "p7_16",
             "p8_12", "p8_13", "p8_14", "p8_15"]
PAGES = ["pages/apresentacao.py", "pages/introducao.py"] + [f"pages/{e}.py" for e in EXERCISES]


def _texts(at):
    out = [m.value for m in at.markdown] + [l.value for l in at.latex]
    out += [w.value for w in at.warning] + [c.value for c in at.caption]
    return out


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = AppTest.from_file(str(APP / page), default_timeout=60).run()
    assert not at.exception, at.exception
    for t in _texts(at):
        assert "[[" not in t, t[:120]


def test_navigation_entrypoint_runs():
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("ex", EXERCISES)
def test_exercise_page_has_all_sections(ex):
    from content import registry
    C = registry.load(ex).CONTENT
    expected = [f"PASSO {len(C.steps)}", "Explore"] + (["Correção"] if C.errata else [])
    at = AppTest.from_file(str(APP / f"pages/{ex}.py"), default_timeout=60).run()
    headers = [h.value for h in at.header]
    for i, name in enumerate(["Enunciado original", "O que está sendo pedido", "Dados e especificações",
                              "Fundamentação teórica", "Equações e variáveis", "Resolução passo a passo",
                              "Implementação computacional", "Gráficos e interpretação",
                              "Verificação das especificações", "Conclusão técnica"], start=1):
        assert f"{i}. {name}" in headers
    joined = "\n".join(_texts(at))
    for e in expected:
        assert e in joined


def test_explore_does_not_change_official_result():
    """Mover o controle do Explore não altera o resultado oficial exibido."""
    at = AppTest.from_file(str(APP / "pages/p7_13.py"), default_timeout=60).run()
    official_before = [m.value for m in at.markdown if "hl-value" in m.value]
    at.select_slider(key="x713_M").set_value(61).run()
    assert not at.exception
    official_after = [m.value for m in at.markdown if "hl-value" in m.value]
    assert official_before == official_after
