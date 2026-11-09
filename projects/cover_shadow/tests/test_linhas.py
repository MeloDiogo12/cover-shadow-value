"""Avaliação de um frame: linhas excluídas sem pose, marcação fora, um defesa que fecha a linha."""

import pandas as pd
import pytest
from metrica.linhas import avaliar_frame, distancia_a_segmento, pode_afetar
from metrica.parametros import PADRAO, PADRAO_V2, PADRAO_V3

# Portador em (-30, 0); recetor em (0, 0) com valor 0,1. Defesa a meio, ligeiramente fora da linha,
# virado para o portador (orientação 180°): as costas, e a sombra, apontam para o recetor.
ORIGEM = (-30.0, 0.0)
RECETORES = pd.DataFrame({"recetor": [7], "x": [0.0], "y": [0.0], "valor_xt": [0.1]})


def _defesas(**campos):
    base = {"jogador": [1], "x": [-15.0], "y": [0.5], "orientacao": [180.0], "a_marcar": [False]}
    base.update({k: [v] for k, v in campos.items()})
    return pd.DataFrame(base)


def test_defesa_com_sombra_sobre_a_linha_nega_ameaca():
    linhas, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(), PADRAO)
    assert not linhas.loc[0, "excluida"] and linhas.loc[0, "n_sombras"] == 1
    assert 0 < linhas.loc[0, "ameaca_negada"] <= 0.1
    assert partes["shapley"].sum() == pytest.approx(linhas.loc[0, "ameaca_negada"])  # um defesa fica com tudo


def test_linha_excluida_se_um_defesa_que_a_pode_afetar_nao_tem_pose():
    linhas, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(orientacao=float("nan")), PADRAO)
    assert linhas.loc[0, "excluida"] and pd.isna(linhas.loc[0, "ameaca_negada"])
    assert partes.empty


def test_defesa_longe_sem_pose_nao_exclui_a_linha():
    longe = _defesas(y=40.0, orientacao=float("nan"))  # fora de comprimento + alcance (D18)
    linhas, _ = avaliar_frame(ORIGEM, RECETORES, longe, PADRAO)
    assert not linhas.loc[0, "excluida"] and linhas.loc[0, "n_podem_afetar"] == 0
    assert linhas.loc[0, "ameaca_negada"] == 0


def test_quem_marca_fica_de_fora_mesmo_sem_pose():
    marca = _defesas(a_marcar=True, orientacao=float("nan"))
    linhas, partes = avaliar_frame(ORIGEM, RECETORES, marca, PADRAO)
    assert not linhas.loc[0, "excluida"] and partes.empty and linhas.loc[0, "n_podem_afetar"] == 0


def test_defesa_que_chega_tarde_tem_probabilidade_baixa():
    # perto do portador (mas a mais de 1 m da linha, senão taparia com o corpo, D30) e virado para o
    # recetor (0°): as costas apontam para o portador, o cone cobre o início da linha, mas a bola
    # passa antes de o defesa reagir (D19: conta pouco, não é cortado)
    linhas, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(orientacao=0.0, x=-28.0, y=1.5), PADRAO)
    assert linhas.loc[0, "n_sombras"] == 1
    assert partes.loc[0, "p_defesa"] < 0.1


def test_distancia_ao_segmento():
    assert distancia_a_segmento(5, 3, 0, 0, 10, 0) == pytest.approx(3)
    assert distancia_a_segmento(-4, 3, 0, 0, 10, 0) == pytest.approx(5)  # além da ponta A
    # limite de D18: comprimento do cone (10) + metade da largura (1) = 11 m
    assert not pode_afetar(_defesas(y=11.5), ORIGEM, RECETORES.iloc[0], PADRAO)[0]
    assert pode_afetar(_defesas(y=10.5), ORIGEM, RECETORES.iloc[0], PADRAO)[0]


def _recetores(**campos):
    return RECETORES.assign(**campos)


def test_fora_de_jogo_exclui_a_linha_e_conta_a_parte():
    linhas, partes = avaliar_frame(ORIGEM, _recetores(fora_de_jogo=True), _defesas(), PADRAO)
    assert linhas.loc[0, "excluida"] and linhas.loc[0, "motivo"] == "fora_de_jogo"
    assert partes.empty and pd.isna(linhas.loc[0, "ameaca_negada"])


def test_cruzamento_exclui_a_linha_e_conta_a_parte():
    linhas, _ = avaliar_frame(ORIGEM, _recetores(cruzamento=True), _defesas(), PADRAO)
    assert linhas.loc[0, "excluida"] and linhas.loc[0, "motivo"] == "cruzamento"


def test_motivo_por_ordem_de_precedencia_e_vazio_quando_avaliada():
    sem_pose = _defesas(orientacao=float("nan"))
    assert avaliar_frame(ORIGEM, _recetores(fora_de_jogo=True, cruzamento=True), sem_pose, PADRAO)[0].loc[0, "motivo"] == "fora_de_jogo"
    assert avaliar_frame(ORIGEM, _recetores(cruzamento=True), sem_pose, PADRAO)[0].loc[0, "motivo"] == "cruzamento"
    assert avaliar_frame(ORIGEM, RECETORES, sem_pose, PADRAO)[0].loc[0, "motivo"] == "sem_pose"
    assert avaliar_frame(ORIGEM, RECETORES, _defesas(), PADRAO)[0].loc[0, "motivo"] == ""


def test_defesa_colado_ao_portador_tapa_a_linha_pelo_corpo():
    """D30 (raio fixo, definição v2): a 0,5 m da linha e dentro do cone, a bola bate-lhe no corpo, sem contar o tempo de reação."""
    # defesa quase em cima do portador, de costas para o recetor (virado para o portador, 180°)
    linhas, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(x=-29.0, y=0.5, orientacao=180.0), PADRAO_V2)
    assert partes.loc[0, "p_defesa"] == pytest.approx(1.0)
    assert linhas.loc[0, "ameaca_negada"] == pytest.approx(0.1)


def test_bloqueio_pelo_corpo_precisa_de_estar_dentro_do_cone():
    # virado para o lado (270°: de frente para -y), o cone aponta para +y e a linha, a 0,5 m, fica fora dele
    _, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(x=-29.0, y=0.5, orientacao=270.0), PADRAO_V2)
    assert partes.empty or partes.loc[0, "p_defesa"] < 1.0


def test_linha_a_mais_de_um_metro_do_defesa_nao_e_bloqueio():
    _, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(x=-29.0, y=1.5, orientacao=180.0), PADRAO_V2)
    assert partes.loc[0, "p_defesa"] < 1.0


def test_bloqueio_gradual_desce_de_1_a_0_3_m_ate_0_a_1_2_m():
    from metrica.linhas import probabilidade_de_bloqueio

    g = PADRAO  # a v3 já é gradual (D34)
    assert probabilidade_de_bloqueio([0.0, 0.3, 0.75, 1.2, 2.0], g).tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0, 0.0])
    assert probabilidade_de_bloqueio([0.5, 1.0, 1.5], PADRAO_V2).tolist() == [1.0, 0.0, 0.0]  # binário: raio de 1 m


def test_bloqueio_gradual_numa_linha_a_0_75_m_da_cerca_de_meia_probabilidade():
    g = PADRAO
    # defesa a 0,75 m da linha, de costas para o recetor (virado para o portador): bloqueio 0,5
    _, partes = avaliar_frame(ORIGEM, RECETORES, _defesas(x=-29.0, y=0.75, orientacao=180.0), g)
    assert partes.loc[0, "p_defesa"] >= 0.45  # a linha é amostrada de 0,25 em 0,25 m


# ---- v3.1 (D43): a v3 mais a exclusão das linhas com mais de 35 m

def test_v3_1_e_a_v3_nas_linhas_ate_35_m():
    defesas = pd.concat([_defesas(), _defesas(jogador=2, x=-5.0, y=-0.8, orientacao=180.0)], ignore_index=True)
    v3, p3 = avaliar_frame(ORIGEM, RECETORES, defesas, PADRAO_V3)
    v31, p31 = avaliar_frame(ORIGEM, RECETORES, defesas, PADRAO)
    assert not v31.loc[0, "excluida"] and v31.loc[0, "comprimento"] == pytest.approx(30.0)
    colunas = [c for c in v3.columns if c != "motivo"]
    pd.testing.assert_frame_equal(v3[colunas], v31[colunas])
    pd.testing.assert_frame_equal(p3, p31)


def test_v3_1_exclui_a_linha_de_mais_de_35_m_e_a_v3_avalia_a():
    longe = pd.DataFrame({"recetor": [7], "x": [10.0], "y": [0.0], "valor_xt": [0.1]})  # 40 m de ORIGEM
    defesas = _defesas(x=-10.0, y=0.5)
    v3, _ = avaliar_frame(ORIGEM, longe, defesas, PADRAO_V3)
    v31, partes = avaliar_frame(ORIGEM, longe, defesas, PADRAO)
    assert not v3.loc[0, "excluida"] and v3.loc[0, "ameaca_negada"] > 0
    assert v31.loc[0, "excluida"] and v31.loc[0, "motivo"] == "mais_de_35m" and pd.isna(v31.loc[0, "ameaca_negada"])
    assert partes.empty


def test_a_omissao_e_a_v3_1_e_a_v3_continua_sem_limite():
    assert PADRAO.comprimento_maximo_linha == 35.0 and PADRAO_V3.comprimento_maximo_linha is None
    assert PADRAO_V2.comprimento_maximo_linha is None
    assert PADRAO == PADRAO_V3.__class__(**{**PADRAO_V3.__dict__, "comprimento_maximo_linha": 35.0})
