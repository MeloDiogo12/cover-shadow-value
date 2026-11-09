"""Velocidade a partir das posições: diferença central sobre 0,4 s, lado único nas pontas."""

import pandas as pd
import pytest
from metrica.velocidade import velocidade_dos_jogadores


def _tabela(frames, xs, jogador=1, periodo=1):
    return pd.DataFrame({"frame": frames, "jogador": jogador, "periodo": periodo, "x": xs, "y": 0.0})


def test_movimento_uniforme_da_a_velocidade_certa():
    # 1 m a cada 0,2 s = 5 m/s, em todos os frames (nas pontas, com o único lado que existe)
    v = velocidade_dos_jogadores(_tabela([10, 12, 14, 16], [0.0, 1.0, 2.0, 3.0]))
    assert v["vx"].tolist() == [pytest.approx(5.0)] * 4
    assert v["vy"].tolist() == [0.0] * 4


def test_alisa_com_os_dois_lados():
    # o frame do meio salta 1 m a mais: a velocidade nele usa (seguinte - anterior) / 0,4 s
    v = velocidade_dos_jogadores(_tabela([10, 12, 14], [0.0, 1.0, 1.4]))
    assert v["vx"].iloc[1] == pytest.approx(1.4 / 0.4)


def test_frame_isolado_ou_com_buraco_fica_parado():
    # buraco entre 12 e 20 (não é o frame alinhado seguinte): sem vizinho contínuo, velocidade 0
    v = velocidade_dos_jogadores(_tabela([12, 20], [0.0, 40.0]))
    assert v["vx"].tolist() == [0.0, 0.0]


def test_jogadores_nao_se_misturam():
    t = pd.concat([_tabela([10, 12], [0.0, 1.0], jogador=1), _tabela([10, 12], [0.0, -2.0], jogador=2)])
    v = velocidade_dos_jogadores(t).set_index("jogador")
    assert v.loc[1, "vx"].iloc[0] == pytest.approx(5.0)
    assert v.loc[2, "vx"].iloc[0] == pytest.approx(-10.0)


def test_salto_do_tracking_acima_de_12_ms_conta_como_parado():
    # o frame do meio é um pico: sobe 10 m e volta (50 m/s), por isso os dois passos à volta são erro
    v = velocidade_dos_jogadores(_tabela([10, 12, 14, 16, 18], [0.0, 1.0, 11.0, 3.0, 4.0]))
    assert v["vx"].iloc[2] == 0.0  # o frame do pico: sem nenhum passo válido, conta como parado
    assert v["vx"].iloc[1] == pytest.approx(5.0)  # só o passo anterior vale; o seguinte é erro
    assert v["vx"].iloc[3] == pytest.approx(5.0)  # só o passo seguinte vale; o anterior é erro
    assert v["vx"].abs().max() <= 12.0


def test_o_limite_vem_do_argumento():
    t = _tabela([10, 12, 14], [0.0, 2.0, 4.0])  # 10 m/s
    assert velocidade_dos_jogadores(t, maxima=12.0)["vx"].iloc[1] == pytest.approx(10.0)
    assert velocidade_dos_jogadores(t, maxima=8.0)["vx"].tolist() == [0.0, 0.0, 0.0]
