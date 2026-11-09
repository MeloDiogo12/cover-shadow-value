"""Velocidade de cada jogador, a partir das posições (para o tempo de reação do defesa, D16/D21).

Durante o tempo de reação o defesa continua à velocidade que trazia (modelo de Shaw, que segue
Spearman); por isso é preciso uma velocidade em cada frame. Calcula-se pela diferença entre as
posições do frame seguinte e do anterior da tabela (0,2 s para cada lado, 0,4 s no total), o que
alisa o ruído do tracking. Nos extremos de um trecho contínuo usa-se só um dos lados (0,2 s); um
frame isolado fica com velocidade 0.

Como em `calc_player_velocities` de Shaw (Metrica_Velocities.py), um passo com velocidade acima de
`maxima` (12 m/s) é erro do tracking: anula-se **antes de alisar**. Um frame que fica sem nenhum
passo válido (por exemplo, o do próprio salto do tracking) conta como parado, velocidade 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.transform.skillcorner_pose import FPS_ALINHADO, PASSO_TRACKING

PASSO_S = 1 / FPS_ALINHADO  # 0,2 s entre dois frames alinhados
VELOCIDADE_MAXIMA = 12.0  # m/s; acima disto é erro do tracking (Shaw, `maxspeed` em Metrica_Velocities.py)


def _diferenca(t: pd.DataFrame, coluna: str, deslocamento: int) -> pd.Series:
    """Variação de `coluna` para o frame vizinho (`deslocamento` = +1 ou -1 linhas), só se for
    exatamente o frame alinhado seguinte/anterior do mesmo jogador e período."""
    vizinho = t.groupby("jogador")[coluna].shift(-deslocamento)
    frame_vizinho = t.groupby("jogador")["frame"].shift(-deslocamento)
    periodo_vizinho = t.groupby("jogador")["periodo"].shift(-deslocamento)
    continuo = (frame_vizinho - t["frame"] == deslocamento * PASSO_TRACKING) & (periodo_vizinho == t["periodo"])
    return (vizinho - t[coluna]).where(continuo) * deslocamento


def _passo(t: pd.DataFrame, deslocamento: int, maxima: float) -> pd.DataFrame:
    """Velocidade (vx, vy) do passo para o frame vizinho; NaN se não há vizinho contínuo ou se a
    velocidade do passo passa de `maxima` (erro do tracking)."""
    v = pd.DataFrame({e: _diferenca(t, e, deslocamento) / PASSO_S for e in ("x", "y")})
    return v.where(np.hypot(v["x"], v["y"]) <= maxima)


def velocidade_dos_jogadores(tabela: pd.DataFrame, maxima: float = VELOCIDADE_MAXIMA) -> pd.DataFrame:
    """`frame`, `jogador`, `vx`, `vy` (m/s): diferença central sobre 0,4 s; lado único nas pontas;
    0 quando o frame não tem nenhum passo válido (sem vizinho contínuo ou acima de `maxima`)."""
    t = tabela.sort_values(["jogador", "frame"]).reset_index(drop=True)
    seguinte, anterior = _passo(t, +1, maxima), _passo(t, -1, maxima)
    saida = t[["frame", "jogador"]].copy()
    for eixo in ("x", "y"):
        saida[f"v{eixo}"] = pd.concat([seguinte[eixo], anterior[eixo]], axis=1).mean(axis=1)
    sem_passo = seguinte["x"].isna() & anterior["x"].isna()
    saida[["vx", "vy"]] = saida[["vx", "vy"]].where(~sem_passo, 0.0)
    return saida


# Em linguagem corrente: a velocidade de um jogador num instante é quanto andou entre o instante
# anterior e o seguinte (0,4 s), dividido por esse tempo. Olhar para os dois lados em vez de um só
# tira os saltinhos do tracking. Se um dos lados não existe (início ou fim de um trecho), usa-se o
# que existe; se não existe nenhum, assume-se que o jogador está parado. Um passo acima de 12 m/s
# (mais do que qualquer jogador corre) é um salto do tracking e deita-se fora antes de fazer a média.
