"""Fase 6 · escolha de candidatos a lance de exemplo (artigo e vídeos).

Funções puras sobre tabelas já calculadas (v3): quais os frames que cumprem os critérios, que defesas
mais valor têm e se alguma marcação está perto de um limiar. Quem escolhe o lance é o autor: aqui só se
filtra e se descrevem os candidatos.
"""

from __future__ import annotations

import pandas as pd
from metrica.linhas import pode_afetar
from metrica.parametros import Parametros

FASES_DE_CONSTRUCAO = ("build_up", "create")  # 1.ª e 2.ª fase de construção (SkillCorner)
ROTULOS_FASE_POSSE = {"build_up": "1.ª fase de construção", "create": "2.ª fase de construção"}
MARGEM_DISTANCIA = 0.5  # m em volta de 4 m (entrada), 5 m (saída) e da distância de pressão
MARGEM_LADO = 0.3  # m em volta do limite do lado da baliza (-1 m)
MARGEM_TEMPO = 0.2  # s em volta do tempo mínimo de marcação (1 s)


def minuto_do_frame(fases: pd.DataFrame, frame: int) -> int:
    """Minuto de jogo (arredondado para baixo) de um frame, a partir do início da fase de jogo em que cai.
    Serve para ordenar e situar o lance; não é o relógio exato."""
    f = fases[(fases["frame_start"] <= frame) & (fases["frame_end"] >= frame)].iloc[0]
    segundos = f["minute_start"] * 60 + f["second_start"] + (frame - f["frame_start"]) / 10
    return int(segundos // 60)


def perto_de_um_limiar(defesas: pd.DataFrame, p: Parametros) -> pd.Series:
    """Por defesa e frame (mesmo índice que `defesas`): True se está perto de um limite da regra da marcação
    (D14, D28, D29, D33): entrada (`distancia_marcacao`), saída (`distancia_saida_marcacao`), distância de pressão
    ao portador, lado da baliza (-`tolerancia_lado_baliza`) ou tempo mínimo seguido. `defesas`: saída de
    `preparar_defesas`."""
    d = defesas
    perto_do_atacante = d["dist_atacante"] <= p.distancia_saida_marcacao + MARGEM_DISTANCIA
    limiar = (
        ((d["dist_atacante"] - p.distancia_marcacao).abs() < MARGEM_DISTANCIA)
        | ((d["dist_atacante"] - p.distancia_saida_marcacao).abs() < MARGEM_DISTANCIA)
        | ((d["dist_portador"] - p.distancia_pressao).abs() < MARGEM_DISTANCIA)
        | (perto_do_atacante & ((d["vantagem_baliza"] + p.tolerancia_lado_baliza).abs() < MARGEM_LADO))
        | (perto_do_atacante & ((d["persistencia_s"] - p.tempo_marcacao).abs() <= MARGEM_TEMPO + 1e-9))
    )
    return limiar


def defesas_que_afetam_alguma_linha(defesas: pd.DataFrame, origem, colegas_xy, p: Parametros) -> pd.Series:
    """Por defesa (mesmo índice que `defesas`, de um só frame): True se pode afetar alguma linha de passe do
    portador (`origem`) para os `colegas_xy` (D18). Só estes importam para decidir se um limiar conta."""
    afeta = pd.Series(False, index=defesas.index)
    for x, y in colegas_xy:
        afeta |= pd.Series(pode_afetar(defesas, origem, {"x": x, "y": y}, p), index=defesas.index)
    return afeta


def marcacao_estavel(defesas: pd.DataFrame, frames) -> bool:
    """True se o conjunto de defesas a marcar é o mesmo em todos os `frames` (a marcação não liga nem desliga
    durante o clip)."""
    frames = sorted(set(frames))
    d = defesas[defesas["frame"].isin(frames)]
    conjuntos = d[d["a_marcar"]].groupby("frame")["jogador"].apply(frozenset)
    conjuntos = [conjuntos.get(f, frozenset()) for f in frames]
    return len(set(conjuntos)) <= 1


def defesas_com_mais_valor(atribuicoes: pd.DataFrame, frame: int, n: int = 3) -> pd.DataFrame:
    """Os `n` defesas com mais ameaça negada (soma das partes de Shapley) num frame: `defesa`, `negada`."""
    a = atribuicoes[atribuicoes["frame"] == frame]
    soma = a.groupby("defesa", as_index=False)["shapley"].sum().rename(columns={"shapley": "negada"})
    return soma.nlargest(n, "negada")


def frames_concentrados(atribuicoes: pd.DataFrame, max_defesas: int = 3, max_por_linha: int = 2) -> pd.Series:
    """Frames em que a ameaça negada vem de 1 a `max_defesas` defesas e nenhuma linha tem mais de `max_por_linha`
    sombras (poucos cones sobrepostos)."""
    por_frame = atribuicoes.groupby("frame").agg(defesas=("defesa", "nunique"))
    por_linha = atribuicoes.groupby(["frame", "recetor"]).size().groupby("frame").max().rename("max_por_linha")
    t = por_frame.join(por_linha)
    return pd.Series(t[(t["defesas"] <= max_defesas) & (t["max_por_linha"] <= max_por_linha)].index, name="frame")


# Em linguagem corrente: a lista de candidatos a lance de exemplo sai de filtros claros. O adversário
# tem de estar a construir (1.ª ou 2.ª fase), todos os defesas que contam têm body pose, não há
# cruzamento, a ameaça negada vem de poucos defesas com poucos cones sobrepostos, e nenhuma
# marcação está perto de um limite da regra (para o lance não mudar com um ajuste pequeno).
