"""Fase 9b: moeda «linhas quebradas», janela do momento da decisão e comprimento das linhas (v3.1, D43)."""

import numpy as np
import pandas as pd
import pytest
from avaliacao_v31 import (
    fim_das_posses,
    moeda_linhas,
    negada_por_comprimento,
    pct_por_contexto,
    posses_por_tipo_de_fim,
    presenca_na_janela,
    sobem_e_descem,
)


def _tabelas():
    """Frame 10: linha A (2 defesas com p = 0,5 cada; vale 3 adversários), linha B (1 defesa com p = 0,4; vale 2)."""
    linhas = pd.DataFrame({
        "frame": [10, 10, 10], "recetor": [1, 2, 3], "excluida": [False, False, True], "valor_xt": [0.1, 0.05, 0.2],
        "p_chegar": [0.25, 0.6, np.nan], "ameaca_negada": [0.1 * 0.75, 0.05 * 0.4, np.nan]})
    atribuicoes = pd.DataFrame({"frame": [10, 10, 10], "recetor": [1, 1, 2], "defesa": [7, 8, 7], "p_defesa": [0.5, 0.5, 0.4],
                                "shapley": [0.0375, 0.0375, 0.02]})
    ultrapassados = pd.DataFrame({"frame": [10, 10, 10], "recetor": [1, 2, 3], "ultrapassados": [3.0, 2.0, 1.0]})
    return linhas, atribuicoes, ultrapassados


def test_moeda_linhas_num_exemplo_a_mao():
    linhas, atrib, ultra = _tabelas()
    novas, partes = moeda_linhas(linhas, atrib, ultra)
    # linha A: 3 · (1 − 0,5 · 0,5) = 2,25 repartido por igual; linha B: 2 · 0,4 = 0,8 todo para o defesa 7; a excluída continua NaN
    assert novas["ameaca_negada"].iloc[0] == pytest.approx(2.25) and novas["ameaca_negada"].iloc[1] == pytest.approx(0.8)
    assert np.isnan(novas["ameaca_negada"].iloc[2])
    assert partes["shapley"].tolist() == pytest.approx([1.125, 1.125, 0.8])
    assert novas["valor_xt"].tolist() == [3.0, 2.0, 1.0]
    # o Shapley soma a ameaça negada de cada linha
    por_linha = partes.groupby("recetor")["shapley"].sum()
    assert por_linha[1] == pytest.approx(2.25) and por_linha[2] == pytest.approx(0.8)


def test_moeda_linhas_com_valor_unitario_e_a_v3_com_valor_1():
    linhas, atrib, _ = _tabelas()
    um = pd.DataFrame({"frame": 10, "recetor": [1, 2, 3], "ultrapassados": 1.0})
    novas, partes = moeda_linhas(linhas, atrib, um)
    assert novas["ameaca_negada"].iloc[0] == pytest.approx(1 - 0.25) and partes["shapley"].iloc[0] == pytest.approx(0.375)


def test_moeda_linhas_recusa_linhas_sem_numero_de_ultrapassados():
    linhas, atrib, ultra = _tabelas()
    with pytest.raises(ValueError):
        moeda_linhas(linhas, atrib, ultra.iloc[:2])


def test_janela_do_momento_da_decisao():
    presenca = pd.DataFrame({"frame": [70, 80, 90, 100, 100], "posse_id": ["a", "a", "a", "a", "b"], "jogador": [1, 1, 1, 1, 2]})
    fim = pd.DataFrame({"posse_id": ["a", "b"], "frame_end": [100, 100], "end_type": ["pass", "shot"]})
    um = presenca_na_janela(presenca, fim, 1.0)  # 10 frames de tracking: frames >= 90
    assert um["frame"].tolist() == [90, 100, 100] and list(um.columns) == list(presenca.columns)
    dois = presenca_na_janela(presenca, fim, 2.0)  # frames >= 80
    assert dois["frame"].tolist() == [80, 90, 100, 100]


def test_posses_por_tipo_de_fim():
    fim = pd.DataFrame({"posse_id": ["a", "b", "c"], "frame_end": [1, 1, 1], "end_type": ["pass", "pass", "shot"]})
    alinhadas = pd.DataFrame({"posse_id": ["a", "b", "b"]})
    janelas = {1.0: pd.DataFrame({"posse_id": ["b"]})}
    r = posses_por_tipo_de_fim(alinhadas, fim, janelas)
    assert r["posses_alinhadas"] == {"pass": 2} and r["posses_com_frames_1s"] == {"pass": 1}


def test_fim_das_posses_so_das_posses_individuais():
    e = pd.DataFrame({"event_type": ["player_possession", "passing_option"], "event_id": ["8_0", "x"], "frame_end": [5, 6], "end_type": ["pass", None]})
    assert fim_das_posses(e)["posse_id"].tolist() == ["8_0"]


def test_sobem_e_descem_pelo_percentil_dentro_da_posicao():
    atual = pd.DataFrame({"jogo": 1, "jogador": [1, 2, 3, 4], "posicao": "CB", "nome": list("abcd"), "negada": [1.0, 2.0, 3.0, 4.0]})
    janela = atual.assign(negada=[4.0, 3.0, 2.0, 1.0])  # ordem invertida
    r = sobem_e_descem(atual, janela, n=1)
    assert r["sobem"][0]["jogador"] == 1 and r["sobem"][0]["variacao"] == pytest.approx(0.75)
    assert r["descem"][0]["jogador"] == 4 and r["descem"][0]["variacao"] == pytest.approx(-0.75)


def test_pct_por_contexto_soma_os_dois_jogos():
    lances = pd.DataFrame({"jogo": [1, 2], "frame": [1, 1], "n_avaliadas": [2, 1], "disponivel": [10.0, 10.0], "negada": [5.0, 1.0]})
    contexto = pd.DataFrame({"jogo": [1, 2], "frame": [1, 1], "fase": ["high_block", "high_block"], "terco": ["a", "b"]})
    r = pct_por_contexto(lances, contexto, "fase")
    assert r.loc[0, "lances"] == 2 and r.loc[0, "pct_negada"] == pytest.approx(30.0)


def test_negada_por_comprimento_reparte_pelas_bandas():
    linhas = pd.DataFrame({"comprimento": [5.0, 15.0, 32.0, 40.0], "excluida": [False, False, False, True],
                           "valor_xt": [1.0, 1.0, 1.0, 1.0], "ameaca_negada": [0.5, 0.25, 0.25, np.nan]})
    t = negada_por_comprimento(linhas)
    assert t["linhas"].sum() == 3 and t["pct_do_total_negada"].sum() == pytest.approx(100.0)
    assert t.loc[t["de_m"] == 30.0, "negada"].iloc[0] == pytest.approx(0.25)


def test_nos_dados_reais_a_v3_1_e_a_v3_nas_linhas_ate_35_m():
    """Precisa das tabelas calculadas (`avaliacao_v3/` e `avaliacao_v3_1/`); sem elas salta."""
    import avaliacao as av
    jogo = 1925299
    try:
        v3, _ = av.ler_variante(jogo, "base", av.PASTAS["v3"])
        v31, _ = av.ler_variante(jogo, "base", av.PASTAS["v3_1"])
    except FileNotFoundError:
        pytest.skip("sem as tabelas calculadas")
    t = v3.merge(v31, on=["frame", "recetor"], suffixes=("_v3", "_v31"))
    # tabelas recém-calculadas já trazem `comprimento` nas duas versões (o merge sufixa-o); as antigas só na v3.1
    comprimento = "comprimento" if "comprimento" in t else "comprimento_v31"
    ate = t[t[comprimento] <= 35.0]
    assert len(ate) > 0 and (ate["excluida_v3"] == ate["excluida_v31"]).all()
    assert np.allclose(ate["ameaca_negada_v3"].fillna(-1), ate["ameaca_negada_v31"].fillna(-1))
    assert np.allclose(ate["p_chegar_v3"].fillna(-1), ate["p_chegar_v31"].fillna(-1))
    mais = t[t[comprimento] > 35.0]
    assert mais["excluida_v31"].all()  # todas as linhas de mais de 35 m ficam excluídas
