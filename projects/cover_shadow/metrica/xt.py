"""Perigo de cada linha de passe: ganho de xT na grelha pública de Karun Singh (D15).

Fonte da grelha: Karun Singh, "Introducing Expected Threat (xT)" (2019), ficheiro oficial
https://karun.in/blog/data/open_xt_12x8_v1.json (a mesma que o socceraction carrega em
`ExpectedThreat.load_model`). Cópia em `referencias/open_xt_12x8_v1.json`.

- 8 linhas (largura do campo) x 12 colunas (comprimento); a coluna 0 é junto à baliza de quem
  ataca, a 11 junto à baliza adversária. É simétrica de cima para baixo.
- Os valores são xT de uma zona (probabilidade de golo); o valor de um passe é o ganho
  `xT(destino) - xT(origem)`. Aqui só interessa o ganho positivo (D15).
- Os dados da SkillCorner têm origem no centro do campo e o sentido de ataque troca ao intervalo,
  por isso as coordenadas viram-se para "quem ataca vai sempre para +x" antes de escolher a zona.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from metrica.marcacao import baliza_propria_x

GRELHA = Path(__file__).resolve().parent.parent / "referencias" / "open_xt_12x8_v1.json"


@lru_cache(maxsize=1)
def grelha_xt() -> np.ndarray:
    """Matriz (8, 12): linha = posição na largura, coluna = posição no comprimento."""
    with open(GRELHA, encoding="utf-8") as f:
        return np.array(json.load(f))


def sentido_de_ataque(match: dict, equipa: int, periodo: int) -> int:
    """+1 se a equipa ataca para x positivo nesse período, -1 se ataca para x negativo."""
    return 1 if baliza_propria_x(match, equipa, periodo) < 0 else -1


def xt_da_posicao(x, y, sentido: int, comprimento: float, largura: float) -> np.ndarray:
    """xT da zona onde está cada ponto, vendo o campo do lado de quem ataca."""
    grelha = grelha_xt()
    linhas, colunas = grelha.shape
    coluna = np.floor((np.asarray(x) * sentido + comprimento / 2) / comprimento * colunas)
    linha = np.floor((np.asarray(y) + largura / 2) / largura * linhas)
    return grelha[np.clip(linha, 0, linhas - 1).astype(int), np.clip(coluna, 0, colunas - 1).astype(int)]


def ganho_de_xt(origem, destino, sentido: int, comprimento: float, largura: float) -> np.ndarray:
    """Perigo da linha: ganho de xT de `origem` (x, y) para `destino` (arrays x, y), só o positivo."""
    xt_origem = xt_da_posicao(*origem, sentido, comprimento, largura)
    xt_destino = xt_da_posicao(*destino, sentido, comprimento, largura)
    return np.maximum(xt_destino - xt_origem, 0.0)


# Em linguagem corrente: o campo está dividido em 96 quadrados e cada um tem um número, que diz
# quão provável é marcar golo a partir dali (perto da baliza adversária o número é alto). Um passe
# vale a diferença entre o quadrado onde chega e o quadrado onde sai; passes para trás ou sem ganho
# valem zero. Antes de ver o quadrado, roda-se o campo para que a equipa que ataca vá sempre para
# a direita, porque os dados trocam o sentido de ataque ao intervalo.
