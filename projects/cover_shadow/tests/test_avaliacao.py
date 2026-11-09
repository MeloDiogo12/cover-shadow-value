"""Funções da fase 5: variantes, correlações, passes reais vs linhas disponíveis, intervalos, seleção."""

import numpy as np
import pandas as pd
import pytest
from avaliacao import (
    RAIOS_BLOQUEIO_V3,
    VALORES_TESTADOS,
    adicionar_comprimento,
    com_comparaveis,
    comparar_com_comparaveis,
    comparar_com_disponiveis,
    escolher_lances,
    intervalo_wilson,
    linhas_que_coincidem,
    linhas_que_mudam,
    passes_com_linha,
    spearman,
    spearman_com_intervalo,
    spearman_por_posicao,
    taxa_de_completos_por_faixa,
    taxa_por_faixa_e_comprimento,
    variantes,
    variantes_v3,
)
from metrica.parametros import PADRAO_V2, PADRAO_V3


def test_variantes_mexem_num_parametro_de_cada_vez():
    todas = variantes()
    assert todas["base"] == (PADRAO_V2, "tronco")
    assert todas["bola"] == (PADRAO_V2, "bola")
    parametros, orientacao = todas["angulo_cone=45"]
    assert parametros.angulo_cone == 45 and orientacao == "tronco"
    assert parametros.comprimento_cone == PADRAO_V2.comprimento_cone  # o resto fica como estava
    assert len(todas) == 2 + sum(len(v) for v in VALORES_TESTADOS.values())
    assert all(PADRAO_V2.__dict__[k] not in v for k, v in VALORES_TESTADOS.items())  # sem repetir a base


def test_variantes_v3_so_mexem_no_bloqueio_pelo_corpo():
    todas = variantes_v3()
    assert todas["base"] == (PADRAO_V3, "tronco") and PADRAO_V3.bloqueio_gradual
    assert len(todas) == 2 + len(RAIOS_BLOQUEIO_V3) + 16 and todas["bola"] == (PADRAO_V3, "bola")
    p, _ = todas["raio_bloqueio_corpo=0.7"]
    assert not p.bloqueio_gradual and p.raio_bloqueio_corpo == 0.7
    assert p.distancia_marcacao == PADRAO_V3.distancia_marcacao  # o resto fica como na v3
    assert todas["distancia_marcacao=5"][0].distancia_marcacao == 5.0 and todas["angulo_cone=45"][0].bloqueio_gradual


def test_escolher_lances_aceita_quantidades_diferentes_por_grupo():
    n = 40
    lances = pd.DataFrame({"jogo": 1, "frame": range(n), "posse_id": [f"p{i}" for i in range(n)],
                           "n_avaliadas": 3, "negada": np.linspace(0, 1, n)})
    discordancia = pd.DataFrame({"jogo": 1, "frame": range(n), "discordancia": np.linspace(1, 0, n)})
    escolha = escolher_lances(lances, discordancia, {"negada": 3, "ao_acaso": 3, "corpo_vs_bola": 2})
    assert escolha.groupby("grupo").size().to_dict() == {"negada": 3, "ao_acaso": 3, "corpo_vs_bola": 2}
    assert escolha["posse_id"].is_unique


def test_spearman_ignora_constantes_e_amostras_pequenas():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert np.isnan(spearman([1, 2], [1, 2]))


def test_spearman_por_posicao_pondera_pelo_numero_de_jogadores_e_ignora_grupos_pequenos():
    base = pd.DataFrame({"jogo": 1, "jogador": range(8), "posicao": ["CB"] * 5 + ["LB"] * 3,
                         "negada": [1, 2, 3, 4, 5, 1, 2, 3]})
    variante = base.assign(negada=[1, 2, 3, 4, 5, 3, 2, 1])  # CB igual; LB (3 jogadores) invertido
    r = spearman_por_posicao(base, variante, minimo=4)
    assert r["rho_media"] == pytest.approx(1) and list(r["posicoes"]["posicao"]) == ["CB"]


def _linhas(**colunas):
    base = {"frame": [1, 1], "recetor": [10, 11], "excluida": [False, False], "valor_xt": [0.1, 0.2],
            "ameaca_negada": [0.0, 0.1], "n_sombras": [0, 1]}
    base.update(colunas)
    return pd.DataFrame(base)


def test_linhas_que_mudam_compara_quem_faz_sombra_e_a_ameaca_negada():
    atrib_base = pd.DataFrame({"frame": [1], "recetor": [11], "defesa": [5], "p_defesa": [0.9], "shapley": [0.1]})
    atrib_var = pd.DataFrame({"frame": [1, 1], "recetor": [10, 11], "defesa": [6, 5], "p_defesa": [0.5, 0.9],
                              "shapley": [0.05, 0.1]})
    m = linhas_que_mudam(_linhas(), _linhas(ameaca_negada=[0.05, 0.1], n_sombras=[1, 1]), atrib_base, atrib_var)
    assert m.set_index("recetor")["muda"].to_dict() == {10: True, 11: False}  # só a linha 10 ganha uma sombra
    assert m.set_index("recetor")["dif_negada"].to_dict() == pytest.approx({10: 0.05, 11: 0.0})


def test_linhas_excluidas_nao_entram_na_comparacao():
    excluida = _linhas(excluida=[True, False])
    vazio = pd.DataFrame(columns=["frame", "recetor", "defesa", "p_defesa", "shapley"])
    m = linhas_que_mudam(excluida, _linhas(), vazio, vazio)
    assert list(m["recetor"]) == [11]


def test_passe_real_junta_a_linha_do_ultimo_frame_e_mede_o_percentil():
    linhas = pd.DataFrame({
        "posse_id": ["a"] * 6, "frame": [10, 10, 10, 12, 12, 12], "recetor": [1, 2, 3, 1, 2, 3],
        "excluida": False, "p_chegar": [1.0, 1.0, 1.0, 1.0, 0.2, 0.8],
    })
    passes = pd.DataFrame({"posse_id": ["a"], "portador": [9], "recetor": [2], "completo": [False]})
    t = passes_com_linha(passes, linhas)
    assert len(t) == 1 and t.loc[0, "frame"] == 12  # o último frame da posse
    assert t.loc[0, "sombra"] == pytest.approx(0.8)
    assert t.loc[0, "sombra_disponivel"] == pytest.approx((0 + 0.8 + 0.2) / 3)
    assert t.loc[0, "percentil"] == pytest.approx(1.0)  # as outras duas têm menos sombra


def test_passe_para_linha_excluida_fica_de_fora():
    linhas = pd.DataFrame({"posse_id": ["a"] * 2, "frame": [10, 10], "recetor": [1, 2],
                           "excluida": [True, False], "p_chegar": [np.nan, 1.0]})
    passes = pd.DataFrame({"posse_id": ["a"], "portador": [9], "recetor": [1], "completo": [True]})
    assert passes_com_linha(passes, linhas).empty


def test_comparar_e_taxas_por_faixa():
    t = pd.DataFrame({"sombra": [0.0, 0.2, 0.0, 0.4], "sombra_disponivel": [0.2, 0.2, 0.4, 0.4],
                      "percentil": [0.3, 0.5, 0.2, 0.5], "p_chegar": [1.0, 0.8, 1.0, 0.6],
                      "completo": [True, True, True, False]})
    r = comparar_com_disponiveis(t)
    assert r["n"] == 4 and r["dif"] == pytest.approx(-0.15) and r["dif_inf"] <= r["dif"] <= r["dif_sup"]
    faixas = taxa_de_completos_por_faixa(t).set_index("faixa")
    assert faixas.loc["1 (sem sombra)", "n"] == 2 and faixas.loc["1 (sem sombra)", "taxa"] == 1
    assert faixas["n"].sum() == 4


def test_intervalo_de_wilson():
    p, baixo, alto = intervalo_wilson(80, 100)
    assert p == pytest.approx(0.8) and 0.70 < baixo < 0.8 < alto < 0.88
    assert np.isnan(intervalo_wilson(0, 0)[0])


def test_linhas_que_coincidem_com_as_opcoes_da_skillcorner():
    opcoes = pd.DataFrame({"opcao_id": ["o1", "o2"], "portador": [9, 9], "recetor": [1, 2],
                           "frame_inicio": [10, 10], "frame_fim": [14, 14], "xthreat": [0.05, 0.02]})
    linhas = pd.DataFrame({"frame": [10, 12, 20, 12], "portador": [9, 9, 9, 8], "recetor": [1, 1, 1, 2],
                           "valor_xt": [0.1, 0.3, 0.9, 0.5]})
    t = linhas_que_coincidem(opcoes, linhas)
    assert len(t) == 1 and t.loc[0, "valor_xt"] == pytest.approx(0.2) and t.loc[0, "frames"] == 2


def test_escolha_de_lances_um_por_posse_e_sem_repetir_entre_grupos():
    lances = pd.DataFrame({"jogo": 1, "frame": range(1, 31), "posse_id": [f"p{i // 2}" for i in range(30)],
                           "n_avaliadas": 3, "negada": np.arange(30, dtype=float)})
    discordancia = pd.DataFrame({"jogo": 1, "frame": range(1, 31), "discordancia": np.arange(30)[::-1] * 1.0})
    escolha = escolher_lances(lances, discordancia, n_por_grupo=3)
    assert escolha.groupby("grupo").size().to_dict() == {"negada": 3, "ao_acaso": 3, "corpo_vs_bola": 3}
    assert not escolha["posse_id"].duplicated().any()
    assert escolha[escolha["grupo"] == "negada"]["frame"].iloc[0] == 30  # o de mais ameaça negada


def test_spearman_com_intervalo_contem_a_estimativa():
    a = np.arange(30, dtype=float)
    rho, baixo, alto = spearman_com_intervalo(a, a + np.sin(a), repeticoes=200)
    assert baixo <= rho <= alto and rho > 0.9


def test_posses_ja_revistas_nao_voltam_a_sair():
    lances = pd.DataFrame({"jogo": 1, "frame": range(1, 61), "posse_id": [f"p{i // 2}" for i in range(60)],
                           "n_avaliadas": 3, "negada": np.arange(60, dtype=float)})
    discordancia = pd.DataFrame({"jogo": 1, "frame": range(1, 61), "discordancia": np.arange(60)[::-1] * 1.0})
    antes = escolher_lances(lances, discordancia, n_por_grupo=3)
    depois = escolher_lances(lances, discordancia, n_por_grupo=3, ja_revistas=[(1, p) for p in antes["posse_id"]])
    assert len(depois) == 9 and not set(depois["posse_id"]) & set(antes["posse_id"])


def test_comprimento_da_linha_vem_das_posicoes_da_tabela():
    tabela = pd.DataFrame({"frame": [10, 10, 10], "jogador": [9, 1, 2], "x": [0.0, 3.0, 0.0], "y": [0.0, 4.0, 10.0]})
    linhas = pd.DataFrame({"frame": [10, 10], "portador": [9, 9], "recetor": [1, 2]})
    assert adicionar_comprimento(linhas, tabela)["comprimento"].tolist() == [pytest.approx(5.0), pytest.approx(10.0)]


def _linhas_com_comprimento():
    """Um frame com a linha escolhida (r1) e três outras: r2 parecida, r3 mais longa, r4 com outro ganho."""
    return pd.DataFrame({
        "posse_id": "a", "frame": 10, "recetor": [1, 2, 3, 4], "excluida": False,
        "p_chegar": [0.9, 0.5, 0.1, 0.0], "comprimento": [10.0, 12.0, 25.0, 11.0],
        "valor_xt": [0.05, 0.06, 0.05, 0.30],
    })


def test_comparaveis_controlam_comprimento_e_ganho_de_xt():
    linhas = _linhas_com_comprimento()
    passes = pd.DataFrame({"posse_id": ["a"], "portador": [9], "recetor": [1], "completo": [True]})
    escolhida = passes_com_linha(passes, linhas)
    t = com_comparaveis(escolhida, linhas)
    assert t.loc[0, "n_comparaveis"] == 1  # só a r2: a r3 é mais longa, a r4 tem outro ganho de xT
    assert t.loc[0, "sombra_comparavel"] == pytest.approx(0.5)
    assert t.loc[0, "sombra"] == pytest.approx(0.1) and t.loc[0, "sombra_disponivel"] == pytest.approx((0.1 + 0.5 + 0.9 + 1.0) / 4)


def test_passe_sem_linhas_comparaveis_fica_de_fora_da_comparacao():
    escolhidas = pd.DataFrame({"sombra": [0.1, 0.2], "sombra_disponivel": [0.4, 0.4], "sombra_comparavel": [0.3, np.nan],
                               "n_comparaveis": [2, 0]})
    r = comparar_com_comparaveis(escolhidas)
    assert r["n"] == 1 and r["dif_controlada"] == pytest.approx(-0.2) and r["dif_sem_controlo"] == pytest.approx(-0.3)


def test_taxa_de_completos_separa_por_banda_de_comprimento():
    t = pd.DataFrame({"comprimento": [5.0, 5.0, 15.0, 25.0], "p_chegar": [1.0, 1.0, 1.0, 0.4],
                      "completo": [True, False, True, False]})
    r = taxa_por_faixa_e_comprimento(t)
    curto_sem_sombra = r[(r["comprimento"] == "0 a 10 m") & (r["faixa"] == "1 (sem sombra)")].iloc[0]
    assert curto_sem_sombra["n"] == 2 and curto_sem_sombra["taxa"] == pytest.approx(0.5)
    assert r.groupby("comprimento")["n"].sum().to_dict() == {"0 a 10 m": 2, "10 a 20 m": 1, "20 m ou mais": 1}
