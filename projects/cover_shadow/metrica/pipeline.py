"""Do jogo à tabela de resultados: junta todos os passos da Cover Shadow Value.

`calcular_jogo(jogo)` lê a tabela da fase 3, aplica a posição do corpo (D12), a marcação (D14),
avalia as linhas de cada frame (`metrica.linhas`) e devolve as tabelas de resultados.
O número do jogo é sempre um parâmetro: nada aqui é específico de um jogo.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd
from entradas import exigir
from preparar_dados import caminho_tabela

from footy.ingest import skillcorner as sc
from footy.metrics.orientacao import angulo_para
from metrica.linhas import avaliar_frame, recetores_com_valor
from metrica.marcacao import caracteristicas_de_marcacao, esta_a_marcar
from metrica.parametros import PADRAO, Parametros
from metrica.velocidade import velocidade_dos_jogadores
from metrica.xt import sentido_de_ataque


class Resultado(NamedTuple):
    linhas: pd.DataFrame  # uma por linha de passe e frame
    atribuicoes: pd.DataFrame  # uma por defesa com sombra sobre a linha (parte de Shapley)
    defesas: pd.DataFrame  # defesas por frame, com marcação e orientação
    frames_sem_portador: int


def zona_das_posses(eventos: pd.DataFrame) -> pd.DataFrame:
    """Corredor e terço onde cada posse começa (etiquetas da SkillCorner), para a zona de cruzamento (D27)."""
    p = eventos[eventos["event_type"] == "player_possession"]
    return pd.DataFrame({"posse_id": p["event_id"].to_numpy(), "corredor": p["channel_start"].to_numpy(),
                         "terco": p["third_start"].to_numpy()})


def carregar_tabela(jogo: int) -> pd.DataFrame:
    """Tabela da fase 3 com a posição do corpo quando há pose (D12) e a zona de cada posse (D27)."""
    t = pd.read_parquet(exigir(caminho_tabela(jogo), f"preparar_dados.py --jogo {jogo}")).merge(zona_das_posses(sc.ler_eventos(jogo)), on="posse_id", how="left")
    t["x_tracking"], t["y_tracking"] = t["x"], t["y"]
    t["x"] = t["x_pose"].where(t["tem_pose"], t["x"])
    t["y"] = t["y_pose"].where(t["tem_pose"], t["y"])
    return t


ORIENTACOES = ("tronco", "bola")


def virar_para_a_bola(defesas: pd.DataFrame) -> pd.Series:
    """Orientação (graus) de um defesa a olhar para o portador. Fica NaN onde o tronco não tem
    orientação (sem pose), para a comparação com o corpo avaliar exatamente as mesmas linhas."""
    # `angulo_para` devolve uma Série com índice 0..n-1; as defesas vêm filtradas (sem guarda-redes),
    # por isso é preciso realinhar ao índice delas, senão cada defesa recebe a orientação de outra linha.
    para_a_bola = angulo_para(defesas["x"], defesas["y"], defesas["portador_x"], defesas["portador_y"])
    para_a_bola = para_a_bola.set_axis(defesas.index)
    return para_a_bola.where(defesas["orient_tronco"].notna())


def preparar_defesas(tabela: pd.DataFrame, match: dict, p: Parametros, orientacao: str = "tronco") -> pd.DataFrame:
    """Defesas por frame (sem o guarda-redes, D25) com `a_marcar` (D14, D28, D29), `orientacao` e velocidade `vx`, `vy`.
    `orientacao="tronco"` é a da métrica (D10); `"bola"` vira o defesa para o portador, para a
    comparação da avaliação (C5): só muda a direção, os defesas sem pose continuam sem orientação."""
    if orientacao not in ORIENTACOES:
        raise ValueError(f"orientacao deve ser uma de {ORIENTACOES}")
    d = caracteristicas_de_marcacao(tabela, match, sem_portador=True)
    d = d[d["posicao"] != "GK"].copy()  # D25: o guarda-redes não projeta sombra nem marca
    d["a_marcar"] = esta_a_marcar(d, p)
    d["orientacao"] = d["orient_tronco"]
    if orientacao == "bola":
        d["orientacao"] = virar_para_a_bola(d)
    return d.merge(velocidade_dos_jogadores(tabela, p.velocidade_maxima_tracking), on=["frame", "jogador"], how="left")


def avaliar_jogo(tabela: pd.DataFrame, defesas: pd.DataFrame, match: dict, p: Parametros) -> Resultado:
    comprimento, largura = sc.dimensoes_campo(match)
    defesas_por_frame = dict(tuple(defesas.groupby("frame")))
    linhas, atribuicoes, sem_portador = [], [], 0
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            sem_portador += 1
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        origem, recetores = recetores_com_valor(jogadores, sentido, comprimento, largura, p)
        l, a = avaliar_frame(origem, recetores, defesas_por_frame[frame], p)
        contexto = {"frame": frame, "posse_id": primeiro["posse_id"], "portador": primeiro["portador"]}
        linhas.append(l.assign(**contexto))
        if not a.empty:
            atribuicoes.append(a.assign(frame=frame))
    return Resultado(pd.concat(linhas, ignore_index=True), pd.concat(atribuicoes, ignore_index=True),
                     defesas, sem_portador)


def calcular_jogo(jogo: int = sc.JOGO_PADRAO, p: Parametros = PADRAO, orientacao: str = "tronco") -> Resultado:
    match = sc.ler_match(jogo)
    tabela = carregar_tabela(jogo)
    return avaliar_jogo(tabela, preparar_defesas(tabela, match, p, orientacao), match, p)


# Em linguagem corrente: este módulo é o maestro. Pega na tabela da fase 3, põe cada jogador onde o
# corpo está, decide quem está a marcar, e para cada instante da posse chama o cálculo das linhas
# de passe. No fim devolve as tabelas: as linhas (com perigo, probabilidade e ameaça negada),
# as partes de cada defesa, e os defesas por instante. O número do jogo é um parâmetro.
