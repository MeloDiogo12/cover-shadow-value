"""Testes do alinhamento tracking (10 fps) ↔ pose (25 fps) e da tabela de trabalho."""

import pandas as pd

from footy.transform.skillcorner_pose import (
    frame_tracking_da_pose,
    frames_alinhados_das_posses,
    linhas_da_pose,
    linhas_do_tracking,
    tabela_de_trabalho,
)


def _ombros():
    return {"lShoulder": {"xyz": [0, 1, 1.5], "p90_mae_cm": 9}, "rShoulder": {"xyz": [0, -1, 1.5], "p90_mae_cm": 9}}


def test_so_os_multiplos_de_5_da_pose_caem_num_frame_de_tracking():
    assert frame_tracking_da_pose(250) == 100
    assert frame_tracking_da_pose(255) == 102
    assert frame_tracking_da_pose(252) is None


def test_frames_alinhados_sao_os_pares_dentro_de_cada_posse():
    posses = pd.DataFrame({
        "posse_id": ["a", "b"], "frame_inicio": [10, 15], "frame_fim": [13, 15],
        "portador": [7, 8], "equipa_com_bola": [1, 2],
    })
    momentos = frames_alinhados_das_posses(posses)
    assert momentos["frame"].tolist() == [10, 12]  # a posse "b" (0 s, frame ímpar) fica sem nenhum
    assert set(momentos["posse_id"]) == {"a"}


def test_tabela_junta_pose_e_marca_quem_defende():
    tracking = linhas_do_tracking(iter([
        {"frame": 10, "period": 1, "ball_data": {"x": 0, "y": 0},
         "player_data": [{"player_id": 7, "x": 1, "y": 2, "is_detected": True},
                         {"player_id": 9, "x": 3, "y": 4, "is_detected": False}]},
        {"frame": 11, "period": 1, "ball_data": {"x": 0, "y": 0}, "player_data": []},
    ]), quais={10})
    pose = linhas_da_pose(iter([
        {"frame": 25, "player_data": [{"player_id": 7, "x": 1, "y": 2, "joints": _ombros()},
                                      {"player_id": 9, "x": 3, "y": 4, "joints": None}]},
        {"frame": 26, "player_data": [{"player_id": 7, "x": 9, "y": 9, "joints": _ombros()}]},
    ]), quais={10})
    momentos = pd.DataFrame({"frame": [10], "posse_id": ["a"], "portador": [7], "equipa_com_bola": [1]})
    jogadores = pd.DataFrame({"jogador": [7, 9], "equipa": [1, 2], "posicao": ["CF", "LCB"]})

    tabela = tabela_de_trabalho(tracking, pose, momentos, jogadores)

    assert tabela["jogador"].tolist() == [7, 9]
    assert tabela["tem_pose"].tolist() == [True, False]
    assert tabela["defende"].tolist() == [False, True]
    assert tabela.loc[0, "lShoulder_y"] == 1
    assert pd.isna(tabela.loc[1, "lShoulder_y"])
