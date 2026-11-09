"""Testes das três versões da orientação do corpo, com jogadores sintéticos vistos de cima."""

import numpy as np
import pandas as pd
import pytest

from footy.metrics.orientacao import (
    angulo_para,
    diferenca_angular,
    orientacao_cabeca,
    orientacao_ombros,
    orientacao_tronco,
)


def _jogador_virado_para_x_positivo():
    """Quem olha para +x tem o lado esquerdo em +y e o direito em -y."""
    return {
        "lShoulder_x": 0.0, "lShoulder_y": 0.2, "rShoulder_x": 0.0, "rShoulder_y": -0.2,
        "lHip_x": 0.0, "lHip_y": 0.15, "rHip_x": 0.0, "rHip_y": -0.15,
        "lEar_x": 0.0, "lEar_y": 0.08, "rEar_x": 0.0, "rEar_y": -0.08,
        "nose_x": 0.1, "nose_y": 0.0,
    }


def test_virado_para_x_positivo_da_zero_graus_nas_tres_versoes():
    t = pd.DataFrame([_jogador_virado_para_x_positivo()])
    for funcao in (orientacao_ombros, orientacao_tronco, orientacao_cabeca):
        assert funcao(t).iloc[0] == pytest.approx(0.0)


def test_virado_para_y_positivo_da_noventa_graus():
    t = pd.DataFrame([{"lShoulder_x": -0.2, "lShoulder_y": 0.0, "rShoulder_x": 0.2, "rShoulder_y": 0.0}])
    assert orientacao_ombros(t).iloc[0] == pytest.approx(90.0)


def test_tronco_fica_entre_ombros_e_ancas():
    linha = _jogador_virado_para_x_positivo()
    linha.update({"lHip_x": -0.15, "lHip_y": 0.0, "rHip_x": 0.15, "rHip_y": 0.0})  # ancas a 90°
    assert orientacao_tronco(pd.DataFrame([linha])).iloc[0] == pytest.approx(45.0)


def test_sem_pose_da_nan():
    t = pd.DataFrame([{k: np.nan for k in _jogador_virado_para_x_positivo()}])
    assert orientacao_ombros(t).isna().all()


def test_diferenca_angular_da_a_volta_pelo_lado_mais_curto():
    assert diferenca_angular(350, 10) == pytest.approx(20)
    assert diferenca_angular(0, 180) == pytest.approx(180)
    assert angulo_para([0], [0], [0], [5]).iloc[0] == pytest.approx(90)
