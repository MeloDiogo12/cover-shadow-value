"""Grelha de xT: valores da fonte, sentido de ataque e só ganho positivo."""

import pytest
from metrica.xt import ganho_de_xt, grelha_xt, sentido_de_ataque, xt_da_posicao

MATCH = {"home_team": {"id": 1}, "home_team_side": ["left_to_right", "right_to_left"], "pitch_length": 105}


def test_grelha_oficial():
    g = grelha_xt()
    assert g.shape == (8, 12)
    assert g[0, 0] == pytest.approx(0.00638303)  # canto da própria baliza
    assert g[3, 11] == pytest.approx(0.25745362)  # em frente à baliza adversária


def test_sentido_troca_ao_intervalo():
    assert sentido_de_ataque(MATCH, 1, 1) == 1
    assert sentido_de_ataque(MATCH, 1, 2) == -1
    assert sentido_de_ataque(MATCH, 2, 1) == -1


def test_zona_vista_de_quem_ataca():
    # a mesma posição vale o oposto conforme o sentido de ataque
    perto_da_baliza_x_positiva = xt_da_posicao(50, 0, 1, 105, 68)
    assert perto_da_baliza_x_positiva == pytest.approx(xt_da_posicao(-50, 0, -1, 105, 68))
    assert perto_da_baliza_x_positiva > xt_da_posicao(-50, 0, 1, 105, 68)


def test_so_o_ganho_positivo():
    assert ganho_de_xt((-40, 0), (40, 0), 1, 105, 68) > 0
    assert ganho_de_xt((40, 0), (-40, 0), 1, 105, 68) == 0  # passe para trás
    assert ganho_de_xt((10, 5), (10, 5), 1, 105, 68) == 0
