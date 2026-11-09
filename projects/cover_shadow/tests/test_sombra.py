"""Geometria da sombra: pontos dentro/fora do cone, direção (o alcance não corta o cone, D19)."""

import numpy as np
import pytest
from metrica.parametros import PADRAO, Parametros
from metrica.sombra import contorno_do_cone, dentro_do_cone, eixo_da_sombra, margem_de_tempo

# Defesa em (0, 0) virado para +x (orientação 0°): as costas, e a sombra, ficam para -x.


def test_eixo_aponta_para_as_costas():
    ex, ey = eixo_da_sombra(0)
    assert (ex, ey) == (pytest.approx(-1), pytest.approx(0, abs=1e-12))
    ex, ey = eixo_da_sombra(90)  # virado para +y: costas para -y
    assert (ex, ey) == (pytest.approx(0, abs=1e-12), pytest.approx(-1))


def test_pontos_dentro_e_fora_do_cone():
    px = np.array([-5.0, 5.0, -5.0, -11.0, -5.0])
    py = np.array([0.0, 0.0, 2.0, 0.0, 4.0])
    dentro = dentro_do_cone(px, py, 0, 0, 0, PADRAO)
    # atrás e ao centro; à frente (fora); 2 m ao lado a 5 m cabe (1 + 5·tan30° = 3,9 m);
    # além do comprimento (10 m); 4 m ao lado a 5 m não cabe
    assert dentro.tolist() == [True, False, True, False, False]


def test_largura_junto_ao_corpo_nao_e_um_ponto():
    assert dentro_do_cone(-0.1, 0.9, 0, 0, 0, PADRAO)  # 0,9 m ao lado, colado ao corpo
    assert not dentro_do_cone(-0.1, 1.1, 0, 0, 0, PADRAO)  # metade da largura é 1 m


def test_abertura_segue_o_angulo():
    estreito = Parametros(angulo_cone=20)
    assert dentro_do_cone(-8, 3.0, 0, 0, 0, PADRAO)
    assert not dentro_do_cone(-8, 3.0, 0, 0, 0, estreito)  # 1 + 8·tan10° = 2,4 m


def test_cone_roda_com_o_corpo():
    # virado para +y, as costas são -y: ponto em (0, -5) dentro, em (-5, 0) fora
    assert dentro_do_cone(0, -5, 0, 0, 90, PADRAO)
    assert not dentro_do_cone(-5, 0, 0, 0, 90, PADRAO)


def test_margem_de_tempo_e_o_avanço_sobre_a_bola():
    # portador em (-20, 0), defesa em (0, 0): a bola chega a (-5, 0) em 1 s, o defesa em 0,7 + 1 = 1,7 s
    assert margem_de_tempo(-5, 0, -20, 0, 0, 0, PADRAO) == pytest.approx(1.0 - 1.7)
    # portador a 40 m: a bola demora 35/15 s, o defesa chega a tempo
    assert margem_de_tempo(-5, 0, -40, 0, 0, 0, PADRAO) > 0


def test_o_cone_nao_e_cortado_pelo_alcance():
    # D19: o ponto (-5, 0) está no cone mesmo que o defesa lá chegue tarde (margem negativa)
    assert margem_de_tempo(-5, 0, -20, 0, 0, 0, PADRAO) < 0
    assert dentro_do_cone(-5, 0, 0, 0, 0, PADRAO)


def test_na_reacao_o_defesa_continua_a_velocidade_que_trazia():
    # ponto a 10 m do defesa, no eixo x; defesa parado: 0,7 + 10/5 = 2,7 s
    parado = margem_de_tempo(10, 0, 100, 0, 0, 0, PADRAO)
    # a correr 5 m/s para o ponto: na reação avança 3,5 m, depois 6,5 m a 5 m/s = 0,7 + 1,3 = 2,0 s
    para_o_ponto = margem_de_tempo(10, 0, 100, 0, 0, 0, PADRAO, defesa_vx=5.0)
    # a correr 5 m/s ao contrário: na reação afasta-se 3,5 m, depois 13,5 m a 5 m/s = 0,7 + 2,7 = 3,4 s
    ao_contrario = margem_de_tempo(10, 0, 100, 0, 0, 0, PADRAO, defesa_vx=-5.0)
    assert para_o_ponto - parado == pytest.approx(0.7)
    assert parado - ao_contrario == pytest.approx(0.7)
    assert ao_contrario < parado < para_o_ponto


def test_contorno_do_cone_tem_a_forma_do_trapezio():
    p = Parametros(angulo_cone=90.0, comprimento_cone=10.0, largura_junto_ao_corpo=2.0)
    cantos = contorno_do_cone(0, 0, 0, p)  # virado para +x: a sombra fica para -x
    assert cantos[:, 0].tolist() == pytest.approx([0, 0, -10, -10])
    assert sorted(cantos[:2, 1]) == pytest.approx([-1, 1])  # 2 m junto ao corpo
    assert sorted(cantos[2:, 1]) == pytest.approx([-11, 11])  # 1 + 10 · tan 45° para cada lado
