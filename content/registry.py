"""Catálogo dos 12 exercícios: identificação, capítulo e estado do conteúdo didático."""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    id: str          # "p7_13"
    number: str      # "P7.13"
    chapter: int
    short: str       # rótulo curto para a navegação
    method: str


EXERCISES = (
    Entry("p7_09", "P7.9", 7, "Passa-faixa", "Hann"),
    Entry("p7_10", "P7.10", 7, "Rejeita-faixa", "Hamming"),
    Entry("p7_11", "P7.11", 7, "Passa-faixa", "Hamming"),
    Entry("p7_12", "P7.12", 7, "Passa-altas", "janela fixa"),
    Entry("p7_13", "P7.13", 7, "Rejeita-faixa", "Kaiser"),
    Entry("p7_14", "P7.14", 7, "Passa-faixa", "Kaiser"),
    Entry("p7_15", "P7.15", 7, "Escada", "Kaiser × PM"),
    Entry("p7_16", "P7.16", 7, "Passa-faixa", "menor janela"),
    Entry("p8_12", "P8.12", 8, "Butterworth", "racional"),
    Entry("p8_13", "P8.13", 8, "Butterworth", "cascata"),
    Entry("p8_14", "P8.14", 8, "Chebyshev-I", "racional"),
    Entry("p8_15", "P8.15", 8, "Chebyshev-I", "cascata"),
)

BY_ID = {e.id: e for e in EXERCISES}


def has_content(ex_id: str) -> bool:
    """True se o conteúdo didático completo (content/<id>.py) já foi escrito."""
    return importlib.util.find_spec(f"content.{ex_id}") is not None


def load(ex_id: str):
    return importlib.import_module(f"content.{ex_id}")
