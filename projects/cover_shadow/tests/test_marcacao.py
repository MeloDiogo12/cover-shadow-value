"""Testes das características candidatas de marcação, com um mini-jogo sintético."""

from dataclasses import replace

import pandas as pd
import pytest
from metrica.marcacao import (
    atacante_mais_proximo,
    baliza_propria_x,
    caracteristicas_de_marcacao,
    esta_a_marcar,
)
from metrica.parametros import PADRAO, PADRAO_V2, Parametros

from footy.transform.skillcorner_pose import ARTICULACOES

MATCH = {"home_team": {"id": 1}, "home_team_side": ["left_to_right", "right_to_left"], "pitch_length": 104}


def _linha(frame, jogador, equipa, x, y, posse="a"):
    return {"frame": frame, "periodo": 1, "jogador": jogador, "equipa": equipa, "x": x, "y": y,
            "posse_id": posse, "portador": 20, "equipa_com_bola": 2, "defende": equipa != 2}


def _mini_jogo():
    """Defesa 10 (casa, defende x = -52) junto ao atacante 21 em 3 frames; o portador é o 20."""
    linhas = []
    for frame in (10, 12, 14):
        linhas += [_linha(frame, 10, 1, -30, 0), _linha(frame, 20, 2, 0, 0), _linha(frame, 21, 2, -28, 1)]
    jogo = pd.DataFrame(linhas)
    for nome in ARTICULACOES:  # as colunas existem na tabela real; aqui ninguém tem pose
        jogo[f"{nome}_x"] = jogo[f"{nome}_y"] = float("nan")
    return jogo


def test_baliza_propria_troca_ao_intervalo():
    assert baliza_propria_x(MATCH, equipa=1, periodo=1) == -52  # a casa ataca para +x
    assert baliza_propria_x(MATCH, equipa=1, periodo=2) == 52
    assert baliza_propria_x(MATCH, equipa=2, periodo=1) == 52


def test_atacante_mais_proximo_e_distancia():
    perto = atacante_mais_proximo(_mini_jogo())
    assert perto["atacante"].tolist() == [21, 21, 21]
    assert perto["dist_atacante"].iloc[0] == pytest.approx(5 ** 0.5)


def test_lado_da_baliza_e_persistencia():
    c = caracteristicas_de_marcacao(_mini_jogo(), MATCH)
    assert c["lado_da_baliza"].all()  # o defesa (x=-30) está mais perto de x=-52 do que o atacante (x=-28)
    assert c["persistencia_s"].tolist() == [pytest.approx(0.6)] * 3  # 3 frames a 5 por segundo
    assert c["dist_portador"].iloc[0] == pytest.approx(30)
    assert c["orient_ombros"].isna().all()  # sem colunas de pose não há orientação


def _features(dist_atacante, dist_portador=None, lado=True, persistencia=1.2, jogador=10, vantagem=None):
    """Sequência de frames alinhados (0,2 s) de um defesa, para testar a regra da marcação. Com `vantagem`
    (m a favor do defesa face à baliza), a coluna `vantagem_baliza` acompanha `lado_da_baliza`."""
    n = len(dist_atacante)
    c = pd.DataFrame({
        "jogador": jogador, "frame": [10 + 2 * i for i in range(n)], "dist_atacante": dist_atacante,
        "dist_portador": dist_portador if dist_portador is not None else [20.0] * n,
        "lado_da_baliza": lado, "persistencia_s": persistencia,
    })
    if vantagem is not None:
        c["vantagem_baliza"] = vantagem
        c["lado_da_baliza"] = c["vantagem_baliza"] > 0
    return c


def test_regra_da_marcacao():
    """Começa a marcar quem está perto (<= 4 m, D33), do lado da baliza e há pelo menos 1 s."""
    c = pd.concat([
        _features([2.0], jogador=1), _features([4.5], jogador=2),
        _features([2.0], lado=False, jogador=3), _features([2.0], persistencia=0.6, jogador=4),
    ], ignore_index=True)
    assert esta_a_marcar(c, PADRAO).tolist() == [True, False, False, False]


def test_pressao_nao_e_marcacao():
    """D28: a menos da distância de pressão do portador nunca se está a marcar."""
    c = pd.concat([_features([2.0], [3.0], jogador=1), _features([2.0], [6.0], jogador=2)], ignore_index=True)
    assert esta_a_marcar(c, PADRAO).tolist() == [False, True]


def test_marcacao_continua_ate_a_distancia_de_saida():
    """D29, D33: entra a <= 4 m e só sai acima de 5 m (entre 4 e 5 m continua a marcar)."""
    c = _features([2.0, 3.5, 4.8, 5.5, 4.5])
    # no último frame já saiu; voltar a 4,5 m (> 4 m) não é entrar de novo
    assert esta_a_marcar(c, PADRAO).tolist() == [True, True, True, False, False]


def test_sem_entrada_nao_ha_marcacao_mesmo_entre_4_e_5_m():
    assert not esta_a_marcar(_features([4.5, 4.5, 4.5]), PADRAO).any()


def test_marcacao_continua_nao_atravessa_falha_de_frames_nem_pressao():
    c = _features([2.0, 4.5, 4.5, 4.5])
    c.loc[2:, "frame"] += 4  # falta um frame alinhado entre o 2.º e o 3.º
    assert esta_a_marcar(c, PADRAO).tolist() == [True, True, False, False]
    c = _features([2.0, 4.5, 4.5], [20.0, 3.5, 20.0])  # frame do meio a pressionar
    assert esta_a_marcar(c, PADRAO).tolist() == [True, False, False]


def test_marcacao_de_dois_defesas_nao_se_mistura():
    c = pd.concat([_features([2.0, 4.0], jogador=1), _features([4.5, 4.5], jogador=2)], ignore_index=True)
    assert esta_a_marcar(c, PADRAO).tolist() == [True, True, False, False]


def test_quem_esta_em_cima_do_portador_nao_e_classificado_como_a_marcar():
    """D20 e D28: o atacante mais próximo exclui o portador, e colado ao portador é pressão."""
    curta = Parametros(tempo_marcacao=0.5)  # o mini-jogo só tem 3 frames (0,6 s)
    jogo = _mini_jogo().astype({"x": float, "y": float})
    jogo.loc[jogo["jogador"] == 10, ["x", "y"]] = [-0.5, 0.0]  # defesa colado ao portador (20)
    jogo.loc[jogo["jogador"] == 20, ["x", "y"]] = [0.0, 0.0]
    jogo.loc[jogo["jogador"] == 21, ["x", "y"]] = [30.0, 0.0]  # o outro atacante está longe
    com = caracteristicas_de_marcacao(jogo, MATCH)  # sem a regra D20, o mais próximo é o portador
    sem = caracteristicas_de_marcacao(jogo, MATCH, sem_portador=True)
    assert com["atacante"].iloc[0] == 20 and sem["atacante"].iloc[0] == 21
    assert not esta_a_marcar(com, curta).any() and not esta_a_marcar(sem, curta).any()


def test_persistencia_quebra_quando_muda_a_posse():
    jogo = _mini_jogo()
    jogo.loc[jogo["frame"] == 14, "posse_id"] = "b"
    c = caracteristicas_de_marcacao(jogo, MATCH).sort_values("frame")
    assert c["persistencia_s"].tolist() == [pytest.approx(0.4), pytest.approx(0.4), pytest.approx(0.2)]


def test_lado_a_lado_conta_ate_a_tolerancia_de_1_m():
    """D33: o defesa ligeiramente ao lado do atacante (até 1 m mais longe da baliza) ainda marca."""
    c = pd.concat([_features([2.0], jogador=1, vantagem=-0.10), _features([2.0], jogador=2, vantagem=-0.68),
                   _features([2.0], jogador=3, vantagem=-1.5)], ignore_index=True)
    assert esta_a_marcar(c, PADRAO).tolist() == [True, True, False]


def test_sem_tolerancia_o_lado_a_lado_nao_marca():
    sem = replace(PADRAO, tolerancia_lado_baliza=0.0)
    assert not esta_a_marcar(_features([2.0], vantagem=-0.10), sem).any()


def test_marcacao_a_3_9_m_ja_entra_com_a_v3_mas_nao_com_a_v2():
    c = _features([3.9])
    assert esta_a_marcar(c, PADRAO).all() and not esta_a_marcar(c, PADRAO_V2).any()
