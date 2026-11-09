"""Zona de cruzamento (D27): onde uma linha para a área não é um passe rasteiro plausível.

A zona vem das etiquetas da SkillCorner nos eventos de posse: o portador está num corredor lateral
(`wide_left`, `wide_right`) do terço ofensivo (`attacking_third`). Nessa zona, uma linha para um
colega dentro da área só conta se for passe atrasado ou curto; as outras são "cruzamento": não se
avaliam e contam-se à parte. Linhas de apoio fora da área contam normalmente.
"""

from __future__ import annotations

import numpy as np

from metrica.parametros import Parametros

CORREDORES_LATERAIS = ("wide_left", "wide_right")
TERCO_OFENSIVO = "attacking_third"
AREA_PROFUNDIDADE = 16.5  # m desde a linha de golo
AREA_MEIA_LARGURA = 20.16  # m para cada lado do centro (campo de 68 m: 40,32 m de largura)


def em_zona_de_cruzamento(corredor, terco) -> bool:
    return corredor in CORREDORES_LATERAIS and terco == TERCO_OFENSIVO


def dentro_da_area_adversaria(x, y, sentido: int, comprimento: float) -> np.ndarray:
    """True nos pontos dentro da grande área da baliza que quem ataca (sentido +1/-1) procura."""
    distancia_a_baliza = comprimento / 2 - np.asarray(x, dtype=float) * sentido
    return (distancia_a_baliza <= AREA_PROFUNDIDADE) & (np.abs(np.asarray(y, dtype=float)) <= AREA_MEIA_LARGURA)


def e_cruzamento(origem, colegas_x, colegas_y, sentido: int, comprimento: float, p: Parametros) -> np.ndarray:
    """True nas linhas para dentro da área que não são passe atrasado nem curto (assume a zona de
    cruzamento já verificada)."""
    colegas_x, colegas_y = np.asarray(colegas_x, dtype=float), np.asarray(colegas_y, dtype=float)
    avanco = (colegas_x - origem[0]) * sentido
    atrasado = avanco <= p.avanco_maximo_atrasado
    curto = np.hypot(colegas_x - origem[0], colegas_y - origem[1]) <= p.comprimento_passe_curto
    return dentro_da_area_adversaria(colegas_x, colegas_y, sentido, comprimento) & ~(atrasado | curto)


# Em linguagem corrente: quando quem tem a bola está junto à linha lateral, já perto da baliza
# adversária, os colegas dentro da grande área são normalmente alvo de cruzamento (bola pelo ar),
# e a sombra de cobertura serve para passes rasteiros. Por isso, nessa situação só se avaliam as
# linhas para a área que sejam para trás/ao lado ou curtas; as restantes ficam de fora e são contadas.
