"""Orientação alternativa para a comparação C5: defesa a olhar para a bola."""

import numpy as np
import pandas as pd
import pytest
from metrica.pipeline import virar_para_a_bola, zona_das_posses


def test_defesa_olha_para_o_portador_e_sem_pose_fica_sem_orientacao():
    defesas = pd.DataFrame({
        "x": [0.0, 0.0], "y": [0.0, 0.0], "portador_x": [10.0, 0.0], "portador_y": [0.0, 10.0],
        "orient_tronco": [123.0, np.nan],
    })
    orientacao = virar_para_a_bola(defesas)
    assert orientacao.iloc[0] == pytest.approx(0)  # portador em +x
    assert np.isnan(orientacao.iloc[1])  # sem pose: continua excluído, como na métrica


def test_virar_para_a_bola_com_indice_nao_contiguo():
    """Defesas filtradas (sem guarda-redes) têm índice com buracos: a orientação tem de seguir a linha."""
    defesas = pd.DataFrame({
        "x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0], "portador_x": [10.0, 0.0, -10.0],
        "portador_y": [0.0, 10.0, 0.0], "orient_tronco": [1.0, 2.0, 3.0],
    }, index=[5, 9, 12])
    orientacao = virar_para_a_bola(defesas)
    assert orientacao.index.tolist() == [5, 9, 12]
    assert orientacao.loc[5] == pytest.approx(0)
    assert orientacao.loc[9] == pytest.approx(90)
    assert abs(orientacao.loc[12]) == pytest.approx(180)


def test_zona_das_posses_usa_as_etiquetas_do_inicio_da_posse():
    eventos = pd.DataFrame({
        "event_type": ["player_possession", "passing_option", "player_possession"],
        "event_id": ["a", "b", "c"], "channel_start": ["wide_left", "center", "center"],
        "third_start": ["attacking_third", "attacking_third", "middle_third"],
    })
    z = zona_das_posses(eventos)
    assert z["posse_id"].tolist() == ["a", "c"] and z["corredor"].tolist() == ["wide_left", "center"]


def test_guarda_redes_nao_e_defesa_com_sombra_nem_marca():
    """D25: o guarda-redes da equipa que defende sai da tabela de defesas."""
    from metrica.parametros import PADRAO
    from metrica.pipeline import preparar_defesas

    from footy.transform.skillcorner_pose import ARTICULACOES

    match = {"home_team": {"id": 1}, "home_team_side": ["left_to_right", "right_to_left"], "pitch_length": 104}
    linhas = []
    for frame in (10, 12):
        for jogador, equipa, x, y, pos in [(10, 1, -30, 0, "CB"), (11, 1, -50, 0, "GK"), (20, 2, 0, 0, "CM"),
                                           (21, 2, -28, 1, "RW")]:
            linhas.append({"frame": frame, "periodo": 1, "jogador": jogador, "equipa": equipa, "x": x, "y": y,
                           "posicao": pos, "posse_id": "a", "portador": 20, "equipa_com_bola": 2,
                           "defende": equipa != 2})
    tabela = pd.DataFrame(linhas)
    for nome in ARTICULACOES:
        tabela[f"{nome}_x"] = tabela[f"{nome}_y"] = float("nan")
    defesas = preparar_defesas(tabela, match, PADRAO)
    assert set(defesas["jogador"]) == {10}
