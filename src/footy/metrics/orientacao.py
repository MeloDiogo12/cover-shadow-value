"""Orientação do corpo a partir das articulações (vista de cima, só x e y).

Três versões, todas em graus no referencial do campo: 0° = sentido +x, a crescer no sentido
anti-horário (90° = +y). Recebem uma tabela com colunas `<articulação>_x` e `<articulação>_y`
(formato de `footy.transform.skillcorner_pose`) e devolvem uma série; NaN onde falta a pose.

- **ombros:** perpendicular ao segmento ombro esquerdo → ombro direito, rodado 90° no sentido
  anti-horário. É a convenção da SkillCorner e a da fase 2 (confirmada pelo nariz à frente).
- **tronco:** a mesma ideia com ombros e ancas, com o mesmo peso para os dois segmentos. As ancas
  mexem menos com os braços, por isso a direção deve tremer menos.
- **cabeça:** do ponto médio entre as orelhas para o nariz, que é para onde a cara aponta.
  Pode diferir do tronco: um defesa pode correr numa direção e olhar para outra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _graus(vx, vy) -> pd.Series:
    """Ângulo do vetor (vx, vy), em graus de 0 a 360; NaN se o vetor for nulo ou faltar."""
    angulo = np.degrees(np.arctan2(vy, vx)) % 360
    return pd.Series(angulo).where(np.hypot(vx, vy) > 0)


def _segmento_unitario(t: pd.DataFrame, de: str, para: str) -> tuple[pd.Series, pd.Series]:
    dx, dy = t[f"{para}_x"] - t[f"{de}_x"], t[f"{para}_y"] - t[f"{de}_y"]
    norma = np.hypot(dx, dy).replace(0, np.nan)
    return dx / norma, dy / norma


def _frente_de(vx, vy) -> tuple[pd.Series, pd.Series]:
    """Roda 90° no sentido anti-horário: de "esquerda → direita" do corpo para "frente"."""
    return -vy, vx


def orientacao_ombros(t: pd.DataFrame) -> pd.Series:
    return _graus(*_frente_de(*_segmento_unitario(t, "lShoulder", "rShoulder"))).set_axis(t.index)


def orientacao_tronco(t: pd.DataFrame) -> pd.Series:
    ox, oy = _segmento_unitario(t, "lShoulder", "rShoulder")
    ax, ay = _segmento_unitario(t, "lHip", "rHip")
    return _graus(*_frente_de(ox + ax, oy + ay)).set_axis(t.index)


def orientacao_cabeca(t: pd.DataFrame) -> pd.Series:
    meio_x = (t["lEar_x"] + t["rEar_x"]) / 2
    meio_y = (t["lEar_y"] + t["rEar_y"]) / 2
    return _graus(t["nose_x"] - meio_x, t["nose_y"] - meio_y).set_axis(t.index)


VERSOES = {"ombros": orientacao_ombros, "tronco": orientacao_tronco, "cabeca": orientacao_cabeca}


def todas_as_orientacoes(t: pd.DataFrame) -> pd.DataFrame:
    """Uma coluna `orient_<versão>` por versão."""
    return pd.DataFrame({f"orient_{nome}": funcao(t) for nome, funcao in VERSOES.items()})


def diferenca_angular(a, b):
    """Diferença entre dois ângulos em graus, de 0 a 180 (350° e 10° diferem 20°, não 340°)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def angulo_para(x, y, alvo_x, alvo_y) -> pd.Series:
    """Direção (graus) de quem está em (x, y) para o alvo."""
    return _graus(np.asarray(alvo_x) - np.asarray(x), np.asarray(alvo_y) - np.asarray(y))


def vetor(angulo_graus) -> tuple[np.ndarray, np.ndarray]:
    """Vetor unitário de um ângulo, para desenhar setas."""
    radianos = np.radians(angulo_graus)
    return np.cos(radianos), np.sin(radianos)


# Em linguagem corrente: visto de cima, um jogador é quase só uma linha de ombros, uma de ancas e
# uma cabeça. Se se souber onde fica o ombro esquerdo e o direito, a frente do peito é a direção
# perpendicular a essa linha (do lado para onde o nariz aponta). As três versões medem a mesma
# coisa de três formas: só os ombros, ombros e ancas juntos (o tronco), e a cabeça (orelhas e
# nariz). O resultado é um ângulo, como num relógio deitado no campo: 0° aponta ao longo do
# comprimento do campo no sentido +x, 90° para a lateral +y. `diferenca_angular` diz quanto duas
# direções discordam, sem o salto falso entre 359° e 0°.
