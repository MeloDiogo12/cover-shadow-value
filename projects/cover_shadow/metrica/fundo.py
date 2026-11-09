"""Defesas de fundo (D39): quem pode cortar o passe sem projetar sombra.

A v4 inicial dava `p = 1` a uma linha sem sombra, mesmo com o recetor marcado ou com um defesa sem pose em cima da linha.
O fundo F são os defesas que não estão em C (C = quem projeta sombra no momento): os que marcam, os que não têm pose e o
guarda-redes. Não recebem crédito (D5, D25), mas podem cortar o passe. `p_fundo_l` = probabilidade de o passe da linha l
chegar considerando só F, com o mesmo modelo de tempo de interceção (`metrica.sombra.margem_de_tempo` e a logística de
`metrica.chegada`), **sem cone nem bloqueio pelo corpo**: sem orientação, qualquer um de F pode chegar a qualquer ponto
da linha. Os defesas intercetam de forma independente, como na v3: `p_fundo_l = Π_d (1 − p_{l,d})`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrica.chegada import probabilidade_de_chegar_ao_ponto
from metrica.parametros import PADRAO, Parametros
from metrica.sombra import margem_de_tempo


def p_fundo_das_linhas(origem, recetores: pd.DataFrame, fundo: pd.DataFrame, p: Parametros = PADRAO) -> np.ndarray:
    """(L,): probabilidade de o passe de cada linha chegar contando só os defesas de `fundo` (colunas `x`, `y` e,
    se existirem, `vx`, `vy`). Cada defesa chega à linha com a maior das probabilidades ao longo dos seus pontos.
    Com `fundo` vazio, 1 em todas as linhas."""
    if fundo.empty:
        return np.ones(len(recetores))
    t = np.linspace(0, 1, p.pontos_por_linha)[None, None, :]  # (1 × 1 × P)
    px = origem[0] + t * (recetores["x"].to_numpy()[:, None, None] - origem[0])  # (L × 1 × P)
    py = origem[1] + t * (recetores["y"].to_numpy()[:, None, None] - origem[1])

    def por_defesa(coluna: str) -> np.ndarray:  # (1 × n × 1); sem velocidade, o defesa parte parado
        valores = fundo[coluna].to_numpy(float) if coluna in fundo else np.zeros(len(fundo))
        return valores[None, :, None]

    margem = margem_de_tempo(px, py, origem[0], origem[1], por_defesa("x"), por_defesa("y"), p,
                             por_defesa("vx"), por_defesa("vy"))  # (L × n × P)
    p_defesa = probabilidade_de_chegar_ao_ponto(margem, p.incerteza_tempo).max(axis=2)  # (L × n)
    return np.prod(1 - p_defesa, axis=1)


def defesas_de_fundo(defesas: pd.DataFrame, em_c, guarda_redes: pd.DataFrame | None) -> pd.DataFrame:
    """Defesas de fundo de um frame: os de `defesas` (sem o guarda-redes, D25) que não estão em C (`em_c`: jogadores
    de C) mais, se for dado, o `guarda_redes` do frame. Colunas: `x`, `y`, `vx`, `vy`."""
    colunas = ["x", "y", "vx", "vy"]
    partes = [defesas[~defesas["jogador"].isin(list(em_c))][colunas]]
    if guarda_redes is not None and not guarda_redes.empty:
        partes.append(guarda_redes[colunas])
    return pd.concat(partes, ignore_index=True)


# Em linguagem corrente: a v4 inicial só olhava para as sombras e, sem sombra, o passe chegava sempre. Mas um defesa que
# está a marcar o recetor, ou o guarda-redes, ou alguém de quem não temos a pose, também corta passes. Aqui calcula-se
# quanto: para cada linha, vê-se se esses defesas (todos os que não fazem sombra) chegam a algum ponto antes da bola,
# com a mesma conta de tempos da sombra mas sem cone, porque sem pose não se sabe para onde olham. O resultado
# multiplica-se ao que as sombras já cortam. Como é igual com e sem as sombras, só as sombras geram valor.
