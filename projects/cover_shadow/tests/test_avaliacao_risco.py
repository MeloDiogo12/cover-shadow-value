"""Fase 9c: moeda «risco» na v3.1 (valor negado = (1 − p_chegar) · (g + c), nunca negativo) e índice por fase."""

import numpy as np
import pandas as pd
import pytest
from avaliacao_risco import (
    dentro_da_area_propria,
    distribuicao_do_indice,
    indice_por_fase,
    moeda_risco,
)
from metrica.passe import custo_da_perda


def _tabelas():
    """Frame 10: linha A (2 defesas com p = 0,5; ganho 0, custo 0,2), linha B (1 defesa com p = 0,4; ganho 0,05, custo 0), C excluída."""
    linhas = pd.DataFrame({
        "frame": [10, 10, 10], "recetor": [1, 2, 3], "excluida": [False, False, True], "valor_xt": [0.0, 0.05, 0.2],
        "p_chegar": [0.25, 0.6, np.nan], "ameaca_negada": [0.0, 0.05 * 0.4, np.nan]})
    atribuicoes = pd.DataFrame({"frame": [10, 10, 10], "recetor": [1, 1, 2], "defesa": [7, 8, 7], "p_defesa": [0.5, 0.5, 0.4],
                                "shapley": [0.0, 0.0, 0.02]})
    custos = pd.DataFrame({"frame": 10, "recetor": [1, 2, 3], "custo_xt": [0.2, 0.0, 0.1], "corte_na_area_propria": [True, False, False]})
    return linhas, atribuicoes, custos


def test_moeda_risco_num_exemplo_a_mao():
    linhas, atrib, custos = _tabelas()
    novas, partes = moeda_risco(linhas, atrib, custos)
    # linha A: ganho 0 mas custo 0,2 -> 0,2 · (1 − 0,5 · 0,5) = 0,15, repartido por igual; linha B: 0,05 · 0,4 = 0,02; a excluída fica NaN
    assert novas["ameaca_negada"].iloc[0] == pytest.approx(0.15) and novas["ameaca_negada"].iloc[1] == pytest.approx(0.02)
    assert np.isnan(novas["ameaca_negada"].iloc[2])
    assert partes["shapley"].tolist() == pytest.approx([0.075, 0.075, 0.02])
    assert novas["valor_xt"].tolist() == pytest.approx([0.2, 0.05, 0.3]) and novas["custo"].tolist() == [0.2, 0.0, 0.1]
    # o Shapley soma a ameaça negada de cada linha
    por_linha = partes.groupby("recetor")["shapley"].sum()
    assert por_linha[1] == pytest.approx(0.15) and por_linha[2] == pytest.approx(0.02)


def test_moeda_risco_nunca_e_negativa_e_sem_custo_e_a_do_xt():
    linhas, atrib, custos = _tabelas()
    novas, partes = moeda_risco(linhas, atrib, custos)
    assert (novas["ameaca_negada"].dropna() >= 0).all() and (partes["shapley"] >= 0).all()
    sem_custo = custos.assign(custo_xt=0.0)
    igual, p_igual = moeda_risco(linhas, atrib, sem_custo)
    assert igual["ameaca_negada"].iloc[:2].tolist() == pytest.approx(linhas["valor_xt"].iloc[:2] * (1 - linhas["p_chegar"].iloc[:2]))
    assert p_igual["shapley"].tolist() == pytest.approx([0.0, 0.0, 0.02])


def test_moeda_risco_recusa_linhas_sem_custo():
    linhas, atrib, custos = _tabelas()
    with pytest.raises(ValueError):
        moeda_risco(linhas, atrib, custos.iloc[:2])


def test_corte_dentro_da_area_de_quem_tem_a_bola_vale_muito():
    """Equipa que ataca para +x: o corte junto à sua baliza (x = −50) custa muito (o adversário ficava a pouco da baliza), junto à baliza
    do adversário (x = +50) custa quase nada. O custo nunca é negativo."""
    pontos = np.array([[-50.0, 0.0], [0.0, 0.0], [50.0, 0.0], [-45.0, 10.0]])
    custo = custo_da_perda(pontos, 1, 105.0, 68.0)
    assert (custo >= 0).all() and custo[0] > 10 * custo[2] and custo[0] > custo[1] > custo[2]
    assert custo_da_perda(pontos, -1, 105.0, 68.0)[2] == pytest.approx(custo[0])  # a mesma zona vista de quem ataca para −x


def test_area_propria():
    x, y = np.array([-45.0, -35.0, 40.0, -45.0]), np.array([0.0, 0.0, 0.0, 25.0])
    assert dentro_da_area_propria(x, y, 1, 105.0).tolist() == [True, False, False, False]
    assert dentro_da_area_propria(-x, y, -1, 105.0).tolist() == [True, False, False, False]  # equipa que ataca para −x defende em +x


def _presenca_e_atribuicoes():
    """Fase A: 4 momentos (frames 1 a 4) com os jogadores 1 e 2 a defender; só o 1 nega (frames 1 e 2). Fase B: 2 momentos, só o 2 nega."""
    presenca = pd.DataFrame({"frame": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], "jogador": [1, 2] * 6})
    contexto = pd.DataFrame({"frame": [1, 2, 3, 4, 5, 6], "fase": ["A"] * 4 + ["B"] * 2})
    atrib = pd.DataFrame({"frame": [1, 2, 5, 6], "defesa": [1, 1, 2, 2], "shapley": [3.0, 1.0, 2.0, 2.0]})
    jogadores = pd.DataFrame({"jogador": [1, 2], "nome": ["a", "b"], "posicao": ["CB", "CB"]})
    return atrib, presenca, contexto, jogadores


def test_indice_por_fase_100_e_a_media_da_fase():
    atrib, presenca, contexto, jogadores = _presenca_e_atribuicoes()
    t = indice_por_fase(atrib, presenca, contexto, jogadores, minimo=2).set_index(["fase", "jogador"])
    # fase A: total 4 em 8 pares jogador–momento -> média 0,5; o 1 faz 4/4 = 1 por momento (índice 200), o 2 faz 0
    assert t.loc[("A", 1), "indice"] == pytest.approx(200.0) and t.loc[("A", 2), "indice"] == 0.0
    # fase B: total 4 em 4 pares -> média 1; cada um faz 2 / 2 = 2 e 0
    assert t.loc[("B", 2), "indice"] == pytest.approx(200.0) and t.loc[("B", 1), "indice"] == 0.0
    # os índices ponderados pelos momentos dão sempre 100 de média na fase
    for fase, g in t.groupby(level="fase"):
        assert np.average(g["indice"], weights=g["momentos"]) == pytest.approx(100.0)
    assert t.loc[("A", 1), "percentil"] == 100.0 and t.loc[("A", 2), "percentil"] == 50.0


def test_indice_por_fase_percentil_so_com_momentos_suficientes():
    atrib, presenca, contexto, jogadores = _presenca_e_atribuicoes()
    t = indice_por_fase(atrib, presenca, contexto, jogadores, minimo=3).set_index(["fase", "jogador"])
    assert t.loc[("A", 1), "percentil"] == 100.0 and np.isnan(t.loc[("B", 1), "percentil"])


def test_distribuicao_do_indice():
    atrib, presenca, contexto, jogadores = _presenca_e_atribuicoes()
    d = distribuicao_do_indice(indice_por_fase(atrib, presenca, contexto, jogadores, minimo=2), minimo=3).set_index("fase")
    assert d.loc["A", "jogadores"] == 2 and d.loc["A", "mediana"] == pytest.approx(100.0) and d.loc["A", "maximo"] == pytest.approx(200.0)
    assert d.loc["B", "jogadores"] == 0 and d.loc["B", "fora_por_poucos_momentos"] == 2
