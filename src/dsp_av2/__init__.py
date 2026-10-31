"""Resolução computacional dos exercícios de projeto de filtros FIR/IIR.

Referência: INGLE, V. K.; PROAKIS, J. G. *Digital Signal Processing Using
MATLAB*, 3ª ed., 2012. Especificações matemáticas em
``docs/documento_transferencia_academica.md`` (DTA) e decisões de projeto em
``docs/plano_implementacao.md``.

Organização:
- ``specs``       — dados literais dos enunciados (nenhum cálculo).
- ``fir``         — cálculos de projeto FIR (resposta ideal, janelas, análise).
- ``validation``  — comparação dos resultados com a especificação do enunciado.
- ``plotting``    — apenas visualização.
- ``reporting``   — gravação de métricas e figuras em ``results/``.
"""

__version__ = "0.1.0"
