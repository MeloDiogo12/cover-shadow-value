"""Probabilidade de chegar e valor de Shapley."""

import numpy as np
import pytest
from metrica.chegada import (
    ameaca_negada,
    probabilidade_de_chegar_ao_ponto,
    probabilidade_de_o_passe_chegar,
    probabilidade_do_defesa,
    valores_de_shapley,
)
from metrica.parametros import PADRAO


def test_um_defesa_sozinho_e_o_efeito_com_e_sem_ele():
    partes = valores_de_shapley(0.2, [0.8])
    assert partes == pytest.approx([0.2 * 0.8])
    assert partes[0] == pytest.approx(ameaca_negada(0.2, [0.8]) - ameaca_negada(0.2, []))


def test_dois_defesas_iguais_dividem_a_meias():
    partes = valores_de_shapley(0.1, [0.6, 0.6])
    assert partes[0] == pytest.approx(partes[1])
    assert partes.sum() == pytest.approx(0.1 * (1 - 0.4 * 0.4))


def test_as_partes_somam_o_total():
    p = [0.9, 0.5, 0.7, 0.55]
    assert valores_de_shapley(0.3, p).sum() == pytest.approx(ameaca_negada(0.3, p))


def test_defesa_mais_provavel_recebe_mais():
    partes = valores_de_shapley(0.1, [0.9, 0.5])
    assert partes[0] > partes[1] > 0


def test_sem_defesas_nada_e_negado():
    assert valores_de_shapley(0.1, []).size == 0
    assert probabilidade_de_o_passe_chegar([]) == 1
    assert ameaca_negada(0.1, []) == 0


def test_probabilidade_do_defesa_so_conta_pontos_no_cone():
    margem = np.array([2.0, 0.0, 0.5])
    assert probabilidade_do_defesa(margem, np.array([False, False, False]), 0.45) == 0
    assert probabilidade_do_defesa(margem, np.array([False, True, False]), 0.45) == pytest.approx(0.5)
    assert probabilidade_do_defesa(margem, np.array([True, True, True]), 0.45) > 0.99
    assert probabilidade_de_chegar_ao_ponto(0, 0.45) == pytest.approx(0.5)


def test_sombra_que_chega_tarde_da_probabilidade_baixa():
    # D19: o defesa chega 1 s depois da bola; o ponto está no cone mas quase não conta
    tarde = probabilidade_de_chegar_ao_ponto(-1.0, PADRAO.incerteza_tempo)
    cedo = probabilidade_de_chegar_ao_ponto(1.0, PADRAO.incerteza_tempo)
    assert tarde < 0.05
    assert cedo > 0.95
    assert probabilidade_do_defesa(np.array([-1.0]), np.array([True]), PADRAO.incerteza_tempo) < 0.05
    assert ameaca_negada(0.1, [tarde]) < 0.005
