"""Linhas de passe e avaliação de um frame (passos 2, 5, 6, 7 e 8 da métrica).

Uma linha de passe vai do portador para um colega (D7, todos, também os totalmente fechados).
Ficam de fora, com o motivo registado em `motivo`: colega em fora de jogo (D26), cruzamento na zona
lateral do terço ofensivo (D27) e defesas sem pose (D13). Para as restantes, cada linha:
1. Perigo = ganho de xT (`metrica.xt`, D15).
2. Defesas que a podem afetar (D18): os que não estão a marcar (D5) e estão a menos de
   `comprimento_cone + largura_junto_ao_corpo / 2` da linha. Se algum destes não tem orientação
   (sem pose), a linha não se avalia (D13).
3. Para cada um, o cone (`metrica.sombra`) e a probabilidade de chegar à linha, nos pontos dentro dele
   (`metrica.chegada`); o alcance só entra nessa probabilidade (D19). Se a linha passa a menos de
   `raio_bloqueio_corpo` do defesa, dentro do cone, está tapada pelo corpo dele (D30).
4. Ameaça negada pelo conjunto e repartição de Shapley.

`avaliar_frame` trabalha sobre dois DataFrames pequenos (recetores e defesas), por isso testa-se
sem tracking nem pose reais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrica.chegada import (
    ameaca_negada,
    probabilidade_de_o_passe_chegar,
    probabilidade_do_defesa,
    valores_de_shapley,
)
from metrica.cruzamento import e_cruzamento, em_zona_de_cruzamento
from metrica.fora_de_jogo import esta_em_fora_de_jogo
from metrica.parametros import PADRAO, Parametros
from metrica.sombra import dentro_do_cone, margem_de_tempo
from metrica.xt import ganho_de_xt

MOTIVOS = ("fora_de_jogo", "cruzamento", "sem_pose", "mais_de_35m")  # por ordem de precedência; "" = avaliada


def recetores_com_valor(frame: pd.DataFrame, sentido: int, comprimento: float, largura: float,
                        p: Parametros = PADRAO) -> tuple:
    """((x, y) do portador, recetores). Recetores: um colega por linha, com o perigo da linha e as
    marcas `fora_de_jogo` (D26) e `cruzamento` (D27). `frame` = jogadores de um frame alinhado, com
    `x`, `y` do corpo (D12) e, se existirem, `corredor` e `terco` da posse (etiquetas da SkillCorner)."""
    portador = frame[frame["jogador"] == frame["portador"]].iloc[0]
    colegas = frame[(frame["equipa"] == portador["equipa"]) & (frame["jogador"] != portador["jogador"])]
    adversarios = frame[frame["equipa"] != portador["equipa"]]
    origem = (portador["x"], portador["y"])
    x, y = colegas["x"].to_numpy(), colegas["y"].to_numpy()
    recetores = pd.DataFrame({
        "recetor": colegas["jogador"].to_numpy(), "x": x, "y": y,
        "valor_xt": ganho_de_xt(origem, (x, y), sentido, comprimento, largura),
        "fora_de_jogo": esta_em_fora_de_jogo(x, origem[0], adversarios["x"].to_numpy(), sentido),
        "cruzamento": False,
    })
    if em_zona_de_cruzamento(portador.get("corredor"), portador.get("terco")):
        recetores["cruzamento"] = e_cruzamento(origem, x, y, sentido, comprimento, p)
    return origem, recetores


def distancia_a_segmento(px, py, ax, ay, bx, by) -> np.ndarray:
    """Distância de cada ponto (px, py) ao segmento A–B (arrays com a mesma forma)."""
    abx, aby = bx - ax, by - ay
    comprimento2 = np.maximum(abx**2 + aby**2, 1e-9)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / comprimento2, 0, 1)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def pode_afetar(defesas: pd.DataFrame, origem, recetor, p: Parametros) -> np.ndarray:
    """D18: defesas que não marcam e a uma distância da linha menor que o comprimento do cone mais
    metade da largura junto ao corpo (o ponto mais afastado da sombra). Independente da orientação."""
    d = distancia_a_segmento(defesas["x"].to_numpy(), defesas["y"].to_numpy(),
                             origem[0], origem[1], recetor["x"], recetor["y"])
    return (d < p.comprimento_cone + p.largura_junto_ao_corpo / 2) & ~defesas["a_marcar"].to_numpy()


def pontos_da_linha(origem, recetor, p: Parametros) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 1, p.pontos_por_linha)
    return origem[0] + t * (recetor["x"] - origem[0]), origem[1] + t * (recetor["y"] - origem[1])


def probabilidade_de_bloqueio(distancia, p: Parametros) -> np.ndarray:
    """Probabilidade de o corpo do defesa tapar um ponto da linha à `distancia` dele (D30). Por omissão,
    1 abaixo de `raio_bloqueio_corpo` e 0 acima; com `bloqueio_gradual`, 1 até `bloqueio_pleno` e a descer
    linearmente até 0 a `bloqueio_nulo`."""
    distancia = np.asarray(distancia, dtype=float)
    if p.bloqueio_gradual:
        return np.clip((p.bloqueio_nulo - distancia) / (p.bloqueio_nulo - p.bloqueio_pleno), 0.0, 1.0)
    return (distancia < p.raio_bloqueio_corpo).astype(float)


def probabilidade_na_linha(defesa, origem, px, py, p: Parametros) -> float:
    """Probabilidade de este defesa chegar à linha, contando só os pontos dentro do seu cone (D19).
    Se a linha passa muito perto dele, dentro do cone, o corpo tapa-a (D30): a bola bate-lhe sem ele
    precisar de reagir. A probabilidade é a maior entre o bloqueio e a de chegar a tempo."""
    sombra = dentro_do_cone(px, py, defesa["x"], defesa["y"], defesa["orientacao"], p)
    bloqueio = probabilidade_de_bloqueio(np.hypot(px - defesa["x"], py - defesa["y"]), p)
    p_bloqueio = float(bloqueio[sombra].max()) if sombra.any() else 0.0
    if p_bloqueio >= 1.0:
        return 1.0
    margem = margem_de_tempo(px, py, origem[0], origem[1], defesa["x"], defesa["y"], p,
                            defesa.get("vx", 0.0), defesa.get("vy", 0.0))
    return max(p_bloqueio, probabilidade_do_defesa(margem, sombra, p.incerteza_tempo))


def motivo_de_exclusao(recetor, candidatos: pd.DataFrame, comprimento: float = 0.0, limite: float | None = None) -> str:
    """Porque é que a linha não se avalia ("" se avalia): colega em fora de jogo (D26), cruzamento
    (D27), algum defesa que a pode afetar sem pose (D13) ou linha com mais de `limite` m (D37, v3.1), por
    esta ordem. Sem `limite` (a v3) o comprimento não conta."""
    if recetor.get("fora_de_jogo", False):
        return "fora_de_jogo"
    if recetor.get("cruzamento", False):
        return "cruzamento"
    if candidatos["orientacao"].isna().any():
        return "sem_pose"
    return "mais_de_35m" if limite is not None and comprimento > limite else ""


def avaliar_linha(origem, recetor, defesas: pd.DataFrame, p: Parametros) -> tuple[dict, list[dict]]:
    """Resultado de uma linha e a parte de cada defesa com sombra sobre ela."""
    candidatos = defesas[pode_afetar(defesas, origem, recetor, p)]
    comprimento = float(np.hypot(recetor["x"] - origem[0], recetor["y"] - origem[1]))
    base = {"recetor": recetor["recetor"], "valor_xt": recetor["valor_xt"], "n_podem_afetar": len(candidatos),
            "comprimento": comprimento}
    motivo = motivo_de_exclusao(recetor, candidatos, comprimento, p.comprimento_maximo_linha)
    if motivo:
        return {**base, "excluida": True, "motivo": motivo, "p_chegar": np.nan, "ameaca_negada": np.nan,
                "n_sombras": 0}, []
    px, py = pontos_da_linha(origem, recetor, p)
    probs = np.array([probabilidade_na_linha(d, origem, px, py, p) for _, d in candidatos.iterrows()])
    com_sombra = probs > 0
    quem, probs = candidatos[com_sombra], probs[com_sombra]
    partes = valores_de_shapley(recetor["valor_xt"], probs)
    linha = {**base, "excluida": False, "motivo": "", "p_chegar": probabilidade_de_o_passe_chegar(probs),
             "ameaca_negada": ameaca_negada(recetor["valor_xt"], probs), "n_sombras": len(quem)}
    atribuicoes = [{"recetor": recetor["recetor"], "defesa": j, "p_defesa": q, "shapley": s}
                   for j, q, s in zip(quem["jogador"], probs, partes)]
    return linha, atribuicoes


def avaliar_frame(origem, recetores: pd.DataFrame, defesas: pd.DataFrame, p: Parametros) -> tuple:
    """(linhas, atribuições) de um frame. `defesas`: jogador, x, y, orientacao (graus; NaN sem
    pose) e a_marcar (bool)."""
    linhas, atribuicoes = [], []
    for _, recetor in recetores.iterrows():
        linha, partes = avaliar_linha(origem, recetor, defesas, p)
        linhas.append(linha)
        atribuicoes += partes
    return pd.DataFrame(linhas), pd.DataFrame(atribuicoes, columns=["recetor", "defesa", "p_defesa", "shapley"])


# Em linguagem corrente: do portador saem linhas retas para os colegas. De cada linha pergunta-se:
# quanto perigo traz (xT), que defesas estão perto o bastante para a poderem fechar e não estão
# ocupados a marcar, e se algum desses não tem dados do corpo (então a linha fica de fora, em vez
# de se adivinhar). Para os restantes, vê-se se a sombra de cada um cobre a linha e com que
# probabilidade ele chegaria à bola; a ameaça negada reparte-se entre eles pelo valor de Shapley.
