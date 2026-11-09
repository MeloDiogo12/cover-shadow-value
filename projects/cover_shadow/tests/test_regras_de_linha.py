"""Fora de jogo (D26) e zona de cruzamento (D27)."""

import numpy as np
import pytest
from metrica.cruzamento import dentro_da_area_adversaria, e_cruzamento, em_zona_de_cruzamento
from metrica.fora_de_jogo import esta_em_fora_de_jogo, linha_de_fora_de_jogo
from metrica.parametros import PADRAO

# ---------------------------------------------------------------- fora de jogo (ataque para +x)


def test_linha_de_fora_de_jogo_e_o_penultimo_adversario():
    assert linha_de_fora_de_jogo([50, 30, 10, 45], sentido=1) == 45  # guarda-redes a 50, penúltimo a 45
    assert linha_de_fora_de_jogo([-50, -30, -10], sentido=-1) == 30  # visto do lado de quem ataca para -x


def test_colega_adiantado_do_penultimo_adversario_esta_em_fora_de_jogo():
    adversarios = [50, 40, 10]
    fora = esta_em_fora_de_jogo([45, 35, 41], portador_x=20, adversarios_x=adversarios, sentido=1)
    assert fora.tolist() == [True, False, True]  # 45 e 41 passam o penúltimo (40); 35 não


def test_nao_ha_fora_de_jogo_atras_da_bola_nem_no_proprio_meio_campo():
    adversarios = [50, 5, 3]
    assert not esta_em_fora_de_jogo([30], portador_x=35, adversarios_x=adversarios, sentido=1)[0]  # atrás da bola
    assert not esta_em_fora_de_jogo([-2], portador_x=-10, adversarios_x=[-20, -25], sentido=1)[0]  # nosso meio-campo


def test_fora_de_jogo_com_ataque_para_x_negativo():
    assert esta_em_fora_de_jogo([-45], portador_x=-20, adversarios_x=[-50, -40, -10], sentido=-1)[0]


def test_adversario_unico_nao_da_linha():
    assert linha_de_fora_de_jogo([30], 1) == -np.inf


# ---------------------------------------------------------------- cruzamento

COMPRIMENTO = 105.0


def test_zona_de_cruzamento_so_no_corredor_lateral_do_terco_ofensivo():
    assert em_zona_de_cruzamento("wide_left", "attacking_third")
    assert em_zona_de_cruzamento("wide_right", "attacking_third")
    assert not em_zona_de_cruzamento("half_space_left", "attacking_third")
    assert not em_zona_de_cruzamento("wide_left", "middle_third")
    assert not em_zona_de_cruzamento(None, None)


def test_area_adversaria():
    dentro = dentro_da_area_adversaria([40.0, 35.0, 40.0, 52.5], [0.0, 0.0, 21.0, -20.0], 1, COMPRIMENTO)
    assert dentro.tolist() == [True, False, False, True]  # 35 m: a 17,5 m da linha de golo; 21 m de lado: fora
    assert dentro_da_area_adversaria([-45.0], [5.0], -1, COMPRIMENTO)[0]  # ataque para -x


def test_na_zona_so_passe_atrasado_ou_curto_para_a_area_conta():
    origem = (30.0, 30.0)  # corredor lateral direito, terço ofensivo
    x = [45.0, 42.0, 30.0, 45.0]
    y = [0.0, 25.0, 0.0, 30.0]
    cruz = e_cruzamento(origem, x, y, 1, COMPRIMENTO, PADRAO)
    assert cruz.tolist() == [True, False, False, False]
    # 1.º: dentro da área, a 33 m e para a frente (cruzamento); 2.º e 4.º: fora da área, apoio normal;
    # 3.º: fora da área também (a 22,5 m da linha de golo)
    assert not e_cruzamento((46.0, 30.0), [45.0], [5.0], 1, COMPRIMENTO, PADRAO)[0]  # na área, mas para trás


def test_passe_curto_para_a_area_nao_e_cruzamento():
    origem = (40.0, 22.0)
    curto = e_cruzamento(origem, [45.0], [12.0], 1, COMPRIMENTO, PADRAO)  # 11,2 m, dentro da área
    longo = e_cruzamento(origem, [45.0], [5.0], 1, COMPRIMENTO, PADRAO)  # 17,7 m
    assert not curto[0] and longo[0]


def test_limiar_de_passe_curto_vem_dos_parametros():
    from dataclasses import replace

    origem = (40.0, 22.0)
    assert e_cruzamento(origem, [45.0], [12.0], 1, COMPRIMENTO, replace(PADRAO, comprimento_passe_curto=5.0))[0]


def test_atrasado_tem_tolerancia():
    origem = (40.0, 22.0)
    assert not e_cruzamento(origem, [40.8], [-2.0], 1, COMPRIMENTO, PADRAO)[0]  # longo mas avanço de 0,8 m <= 1 m
    assert e_cruzamento(origem, [42.0], [-2.0], 1, COMPRIMENTO, PADRAO)[0]
    assert pytest.approx(1.0) == PADRAO.avanco_maximo_atrasado
