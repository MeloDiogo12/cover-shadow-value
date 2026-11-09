"""Modelo completo da probabilidade de o passe chegar (D40) e as "moedas" (valor de cada linha).

A v3 só contava os defesas cuja sombra (cone atrás do tronco) cruza a linha. Os defesas **à frente** da linha, que são os
que mais facilmente a cortam, não contavam. Aqui, para cada linha l e cada defesa d (todos os que defendem, o guarda-redes
incluído), os pontos da linha dividem-se em:
- **dentro do cone de d** (só se d projeta sombra: tem pose e não marca): `p_cone` da v3, com bloqueio pelo corpo;
- **fora do cone** (ou todos os pontos, se d não projeta sombra): `p_fora`, tempo de interceção de Spearman/Shaw, sem
  orientação (qualquer direção) nem bloqueio pelo corpo.
`p_fundo_l = Π_d (1 − p_fora_{d,l})` sobre todos os defesas; `p_sombras_l(C) = Π_{d∈C} (1 − p_cone_{d,l})`;
`p_l(C) = p_fundo_l · p_sombras_l(C)` e `p_l(∅) = p_fundo_l`. Só a parte do cone gera crédito.

A "moeda" é o valor esperado de escolher uma linha dado p: `ev(p) = p·a − (1 − p)·b`, com `a` o ganho da linha e `b` o custo
de a perder. xT: `a` = ganho de xT, `b` = 0; linhas: `a` = adversários ultrapassados, `b` = 0; risco: `a` = ganho de xT,
`b` = xT do adversário onde a linha é mais provavelmente cortada. Com `b = 0` é o `p · a` de sempre.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from metrica.chegada import probabilidade_de_chegar_ao_ponto
from metrica.linhas import probabilidade_de_bloqueio
from metrica.parametros import PADRAO, Parametros
from metrica.sombra import dentro_do_cone, margem_de_tempo
from metrica.xt import xt_da_posicao

MOEDAS = ("xt", "linhas", "risco")


class Probabilidades(NamedTuple):
    """Por linha (L) e defesa (n). `p_cone`: 0 onde o defesa não projeta sombra; `p_fora`: pontos fora do cone (todos, se não
    projeta sombra); `p_corte`: a maior probabilidade de o defesa cortar a linha, em qualquer ponto (cone ou fora), e
    `ponto_corte` (L × 2) o ponto onde o defesa com maior `p_corte` a corta com mais probabilidade."""
    p_cone: np.ndarray
    p_fora: np.ndarray
    p_corte: np.ndarray
    ponto_corte: np.ndarray


def pontos_das_linhas(origem, recetores_xy: np.ndarray, pontos: int) -> tuple[np.ndarray, np.ndarray]:
    """(L × P): coordenadas x e y dos `pontos` pontos de cada linha, do portador ao recetor."""
    t = np.linspace(0, 1, pontos)[None, :]
    return origem[0] + t * (recetores_xy[:, 0:1] - origem[0]), origem[1] + t * (recetores_xy[:, 1:2] - origem[1])


def probabilidades_das_linhas(origem, recetores_xy: np.ndarray, defesas: dict, p: Parametros = PADRAO) -> Probabilidades:
    """`p_cone` e `p_fora` de cada defesa em cada linha. `defesas`: arrays (n) `x`, `y`, `vx`, `vy`, `orientacao` (graus; NaN
    sem pose) e `sombra` (bool: projeta sombra, ou seja tem pose e não marca). Igual, ponto a ponto, a `probabilidade_na_linha`
    da v3 para a parte do cone."""
    px, py = pontos_das_linhas(origem, recetores_xy, p.pontos_por_linha)  # (L × P)
    px, py = px[:, None, :], py[:, None, :]
    d = {k: np.asarray(defesas[k], float)[None, :, None] for k in ("x", "y", "vx", "vy", "orientacao")}
    sombra = np.asarray(defesas["sombra"], bool)[None, :, None]
    no_cone = dentro_do_cone(px, py, d["x"], d["y"], d["orientacao"], p) & sombra  # (L × n × P)
    margem = margem_de_tempo(px, py, origem[0], origem[1], d["x"], d["y"], p, d["vx"], d["vy"])
    chega = probabilidade_de_chegar_ao_ponto(margem, p.incerteza_tempo)
    bloqueio = np.where(no_cone, probabilidade_de_bloqueio(np.hypot(px - d["x"], py - d["y"]), p), 0.0)
    p_cone = np.where(no_cone.any(axis=2), np.maximum(bloqueio.max(axis=2), np.where(no_cone, chega, 0.0).max(axis=2)), 0.0)
    p_fora = np.where(~no_cone, chega, 0.0).max(axis=2)
    por_ponto = np.where(no_cone, np.maximum(bloqueio, chega), chega)  # (L × n × P)
    melhor = por_ponto.max(axis=2)
    quem = melhor.argmax(axis=1)
    ponto = por_ponto[np.arange(len(quem)), quem].argmax(axis=1)  # (L,)
    cortes = np.stack([px[:, 0, :][np.arange(len(ponto)), ponto], py[:, 0, :][np.arange(len(ponto)), ponto]], axis=1)
    return Probabilidades(p_cone, p_fora, melhor, cortes)


def p_fundo(p_fora: np.ndarray) -> np.ndarray:
    """(L,): probabilidade de nenhum defesa cortar a linha fora do seu cone (independentes)."""
    return np.prod(1 - p_fora, axis=1)


# ---------------------------------------------------------------- moedas

def valor_esperado(p, ganho, custo=None):
    """`ev(p) = p·ganho − (1 − p)·custo`: o valor esperado de escolher a linha (a moeda). Sem custo, `p·ganho`."""
    p = np.asarray(p, float)
    return p * ganho if custo is None else p * ganho - (1 - p) * custo


def adversarios_ultrapassados(origem, recetores_xy: np.ndarray, adversarios_x: np.ndarray, sentido: int) -> np.ndarray:
    """(L,): nº de adversários de campo (sem guarda-redes) entre o portador e o recetor, no sentido do ataque. 0 para passes
    para trás ou sem avanço. "Entre" = coordenada longitudinal estritamente maior que a do portador e menor que a do recetor."""
    de = origem[0] * sentido
    para = recetores_xy[:, 0] * sentido
    a = np.asarray(adversarios_x, float)[None, :] * sentido
    dentro = (a > de) & (a < para[:, None])
    return np.where(para > de, dentro.sum(axis=1), 0).astype(float)


def custo_da_perda(pontos_corte: np.ndarray, sentido: int, comprimento: float, largura: float) -> np.ndarray:
    """(L,): xT do **adversário** no ponto de corte: a grelha vista da outra equipa (que ataca no sentido contrário)."""
    return xt_da_posicao(pontos_corte[:, 0], pontos_corte[:, 1], -sentido, comprimento, largura)


def ganho_e_custo(moeda: str, ganho_xt: np.ndarray, ultrapassados: np.ndarray, custo_xt: np.ndarray) -> tuple:
    """(a, b) de cada linha para a moeda pedida (ver o topo do módulo)."""
    if moeda == "xt":
        return ganho_xt, None
    if moeda == "linhas":
        return ultrapassados, None
    if moeda == "risco":
        return ganho_xt, custo_xt
    raise ValueError(f"moeda deve ser uma de {MOEDAS}")


# Em linguagem corrente: antes só se olhava para os defesas que fazem sombra à linha. Mas um defesa que está à frente do
# passe, ou ao lado, também o pode cortar, e é esse que mais vezes o corta. Por isso cada defesa tem duas hipóteses de
# cortar uma linha: pela zona da sombra (como antes, com o corpo a tapar o que está muito perto) e por todo o resto da
# linha, sem se saber para onde olha. A parte de fora multiplica-se (sem crédito para ninguém) e só a parte da sombra
# gera valor. A "moeda" diz o que vale uma linha: o ganho de xT, o número de adversários que o passe ultrapassa, ou o
# ganho de xT menos o perigo que o adversário teria se a bola fosse perdida no sítio onde a linha é mais provavelmente
# cortada.
