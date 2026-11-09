"""Fase 9d: a moeda risco é a oficial (D44) sem mudar a v3.1; a moeda de um só lance; números em risco por jogo e por defesa."""

import numeros_9d as n9d
import numeros_v31 as n31
import numpy as np
import pandas as pd
import pytest
from avaliacao_risco import moeda_risco, moeda_risco_do_lance
from metrica.parametros import MOEDA_PADRAO, MOEDAS_DISPONIVEIS, PADRAO, PADRAO_V2, PADRAO_V3


def test_a_moeda_oficial_e_o_risco_e_a_v3_1_nao_mudou():
    assert MOEDA_PADRAO == "risco" and MOEDA_PADRAO in MOEDAS_DISPONIVEIS and {"xt", "linhas"} <= set(MOEDAS_DISPONIVEIS)
    # as versões anteriores continuam reproduzíveis: a moeda não é um parâmetro da métrica
    assert PADRAO.comprimento_maximo_linha == 35.0 and PADRAO_V3.comprimento_maximo_linha is None
    assert PADRAO_V2.distancia_marcacao == 3.0 and PADRAO_V2.bloqueio_gradual is False
    assert "moeda" not in PADRAO.__dataclass_fields__


def _tabelas_do_frame():
    """Frame 10, tal como sai de `dados_do_frame`: sem coluna `frame`."""
    linhas = pd.DataFrame({"recetor": [1, 2, 3], "excluida": [False, False, True], "valor_xt": [0.0, 0.05, 0.2],
                           "p_chegar": [0.25, 0.6, np.nan], "ameaca_negada": [0.0, 0.05 * 0.4, np.nan]})
    atribuicoes = pd.DataFrame({"recetor": [1, 1, 2], "defesa": [7, 8, 7], "p_defesa": [0.5, 0.5, 0.4], "shapley": [0.0, 0.0, 0.02]})
    custos = pd.DataFrame({"frame": [10, 10, 10, 11], "recetor": [1, 2, 3, 1], "custo_xt": [0.2, 0.0, 0.1, 9.0],
                           "corte_na_area_propria": [True, False, False, True]})
    return linhas, atribuicoes, custos


def test_moeda_risco_de_um_lance_e_a_mesma_da_tabela_inteira():
    linhas, atribuicoes, custos = _tabelas_do_frame()
    novas, partes = moeda_risco_do_lance(linhas, atribuicoes, custos, 10)
    assert "frame" not in novas.columns and "frame" not in partes.columns
    inteiras, partes_inteiras = moeda_risco(linhas.assign(frame=10), atribuicoes.assign(frame=10), custos[custos["frame"] == 10])
    assert novas["ameaca_negada"].iloc[:2].tolist() == pytest.approx(inteiras["ameaca_negada"].iloc[:2].tolist())
    assert partes["shapley"].tolist() == pytest.approx(partes_inteiras["shapley"].tolist())
    # linha A: g = 0, c = 0,2, duas sombras de 0,5 -> 0,2 · 0,75 = 0,15; o custo de outro frame não entra
    assert novas["ameaca_negada"].iloc[0] == pytest.approx(0.15) and novas["valor_xt"].tolist() == pytest.approx([0.2, 0.05, 0.3])


def _tabelas_dos_jogos():
    """Dois jogos, cada um com um frame de uma linha: valor 0,4 negado por um defesa (jogo 1) e 0,2 só com 0,05 negado (jogo 2)."""
    def linhas(frame, valor, negada):
        return pd.DataFrame({"frame": [frame], "posse_id": [1], "recetor": [9], "excluida": [False], "valor_xt": [valor], "ameaca_negada": [negada]})
    def atrib(frame, negada):
        return pd.DataFrame({"frame": [frame], "recetor": [9], "defesa": [7], "p_defesa": [0.5], "shapley": [negada]})
    return {1: {"risco": (linhas(5, 0.4, 0.3), atrib(5, 0.3))}, 2: {"risco": (linhas(6, 0.2, 0.05), atrib(6, 0.05))}}


def test_totais_por_jogo_e_juntos(monkeypatch):
    monkeypatch.setattr(n9d, "JOGOS", (1, 2))
    t = n9d.totais_da_moeda(_tabelas_dos_jogos())
    assert t["1"]["pct_negada"] == pytest.approx(75.0) and t["2"]["pct_negada"] == pytest.approx(25.0)
    assert t["juntos"]["lances"] == 2 and t["juntos"]["disponivel"] == pytest.approx(0.6) and t["juntos"]["negada"] == pytest.approx(0.35)
    assert t["juntos"]["pct_negada"] == pytest.approx(100 * 0.35 / 0.6)


def test_lance_por_defesa_com_valor_percentagem_e_indice(monkeypatch):
    monkeypatch.setattr(n9d, "JOGOS", (1, 2))
    JG = {1: {"jogadores": pd.DataFrame({"jogador": [7, 8], "nome": ["A. Sete", "B. Oito"]})}}
    ctx = pd.DataFrame({"jogo": [1], "frame": [5], "fase": ["high_block"]})
    indices = pd.DataFrame({"fase": ["high_block", "high_block"], "jogador": [7, 8], "momentos": [150, 40], "indice": [180.0, 20.0],
                            "percentil": [95.0, np.nan]})
    r = n9d.lance_por_defesa(JG, _tabelas_dos_jogos(), ctx, indices, 1, 5)
    assert r["fase"] == "high_block" and r["disponivel"] == pytest.approx(0.4) and r["pct_negada"] == pytest.approx(75.0)
    d = r["por_defesa"]["A. Sete"]
    assert d["negada"] == pytest.approx(0.3) and d["pct_do_disponivel"] == pytest.approx(75.0)
    assert d["indice_na_fase"] == 180.0 and d["percentil_na_fase"] == 95.0 and d["momentos_na_fase"] == 150


def test_fase_e_terco_somam_100_na_parte_do_total(monkeypatch):
    monkeypatch.setattr(n9d, "JOGOS", (1, 2))
    monkeypatch.setattr(n31, "JOGOS", (1, 2))
    T = _tabelas_dos_jogos()
    for j in T:
        T[j]["risco"][0]["motivo"] = ""
    ctx = pd.DataFrame({"jogo": [1, 2], "frame": [5, 6], "fase": ["high_block", "low_block"], "terco": ["middle_third", "defensive_third"]})
    r = n9d.fase_e_terco(T, ctx)
    for por in ("fase", "terco"):
        assert sum(x["pct_da_negada_total"] for x in r[por]) == pytest.approx(100.0)
        assert sum(x["pct_dos_lances"] for x in r[por]) == pytest.approx(100.0)
    assert {x["fase"]: x["pct_negada"] for x in r["fase"]} == pytest.approx({"high_block": 75.0, "low_block": 25.0})
