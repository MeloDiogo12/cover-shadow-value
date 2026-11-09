"""Fase 8d: variantes, resumos de C4/C5 e secção 5 da avaliação da v4, e a folha da 4.ª ronda de revisão."""

import zipfile

import avaliacao_v4 as a4
import numpy as np
import pandas as pd
import pytest
import revisao_v4 as r4


def test_as_variantes_são_as_da_v3_mais_o_limite_de_comprimento():
    v = a4.variantes_v4()
    assert next(iter(v)) == "base" and v["base"].limite == 35.0 and v["base"].orientacao == "tronco"
    assert v["bola"].orientacao == "bola"
    assert v["limite=30"].limite == 30.0 and v["limite=40"].limite == 40.0
    assert {"raio_bloqueio_corpo=0.5", "angulo_cone=45", "distancia_marcacao=5", "tempo_marcacao=1.5"} <= set(v)
    assert len(v) == 25  # 23 da v3 (base, 5 raios, 16 dos parâmetros antigos, bola) + 2 limites


def test_a_base_calcula_as_alternativas_da_incerteza():
    assert a4.ALTERNATIVOS_DA_BASE[0] == "ambos" and len(a4.ALTERNATIVOS_DA_BASE) == 1 + a4.N_CONJUNTOS


def test_por_comprimento_reparte_pelas_faixas_e_as_partes_somam_100():
    comp = pd.Series([5.0, 10.0, 12.0, 25.0, 33.0, 40.0])
    t = a4.por_comprimento(comp, pd.Series([1.0, 1, 1, 1, 1, 1]), pd.Series([2.0, 2, 2, 2, 2, 2]))
    assert list(t["linhas"]) == [2, 1, 1, 1, 1]  # 10 m cai na faixa 0–10 (fechada à direita)
    assert t["pct_da_negada"].sum() == pytest.approx(100)
    assert t.loc["30–35 m", "negada"] == 1.0 and t.loc["> 35 m", "negada"] == 1.0


def test_linhas_que_mudam_conta_so_as_linhas_nas_duas_versoes():
    tronco = pd.DataFrame({"frame": [1, 1, 2], "recetor": [7, 8, 7], "quem_sombra": ["3", "", "4,5"]})
    bola = pd.DataFrame({"frame": [1, 1, 3], "recetor": [7, 8, 7], "quem_sombra": ["3", "9", "4"]})
    r = a4.linhas_que_mudam(tronco, bola)
    assert r["linhas"] == 2 and r["linhas_que_mudam"] == 1 and r["pct_muda"] == 50.0
    assert r["linhas_com_sombra_em_alguma"] == 2 and r["pct_muda_entre_as_com_sombra"] == 50.0


def test_peso_de_I_e_D_por_posicao():
    j = pd.DataFrame({"jogador": [1, 2, 3], "posicao": ["CB", "CB", "LB"], "negada": [1.0, 1.0, 2.0],
                      "negada_I": [1.5, 1.5, 3.0], "negada_D": [-0.5, -0.5, -1.0]})
    t = a4.peso_de_I_e_D(j)
    assert t.loc["CB", "negada"] == 2.0 and t.loc["CB", "pct_I"] == 150.0 and t.loc["CB", "pct_D"] == -50.0
    assert t.loc["CB", "pct_I"] + t.loc["CB", "pct_D"] == pytest.approx(100)


def test_negada_por_jogador_usa_a_coluna_pedida_e_conta_zero_nos_frames_sem_valor():
    momentos = pd.DataFrame({"frame": [1, 2], "E0": [0.4, 0.2]})
    atrib = pd.DataFrame({"frame": [1], "defesa": [10], "shapley": [0.1], "shapley_simetrica": [0.1]})
    presenca = pd.DataFrame({"frame": [1, 2, 1, 2], "posse_id": ["a"] * 4, "jogador": [10, 10, 11, 11]})
    jog = pd.DataFrame({"jogador": [10, 11], "equipa": [1, 1], "posicao": ["CB", "CB"]})
    t = a4.negada_por_jogador(momentos, atrib, presenca, jog).set_index("jogador")
    assert t.loc[10, "negada"] == pytest.approx(0.05) and t.loc[11, "negada"] == 0.0
    assert t.loc[10, "disponivel"] == pytest.approx(0.3)


def test_rho_igual_a_um_quando_a_variante_e_a_base():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"jogo": 1, "jogador": range(8), "posicao": ["CB"] * 4 + ["LB"] * 4, "negada": rng.uniform(size=8)})
    r = a4.comparar_com_a_base(base, base)
    assert r["rho_media"] == pytest.approx(1) and r["rho_min"] == pytest.approx(1) and r["n_posicoes"] == 2


def test_tabela_c8_le_o_json_da_8b():
    modelo = {"fora_da_amostra": {"1_para_2": {"treino": 1, "teste": 2, "ao_acaso": {"log_loss": 2.0, "top1": 0.1, "top3": 0.3},
                                               "modelos": {"M0": {"log_loss": 1.5, "top1": 0.2, "top3": 0.5, "n_passes": 9}},
                                               "M2_menos_M0": {"diferenca": -0.01, "ic95_inf": -0.02, "ic95_sup": 0.0}}},
              "coeficientes_conjunto": {"M3": {"sombra_bola": {"coef": 1.0, "erro_padrao": 0.1, "ic95_inf": 0.8, "ic95_sup": 1.2,
                                                               "razao_de_chances": 2.7}}}}
    t = a4.tabela_c8(modelo)
    assert list(t["modelos"]["modelo"]) == ["M0", "ao acaso"] and t["diferencas"].loc[0, "comparacao"] == "M2 − M0"
    assert t["coeficientes_M3"].loc["sombra_bola", "coef"] == 1.0


# ---------------------------------------------------------------- revisão da 4.ª ronda

def test_dissuasao_mais_negativa_tem_posses_distintas_e_exclui_as_dadas():
    m1 = pd.DataFrame({"frame": [1, 2, 3, 4], "posse_id": ["a", "a", "b", "c"], "D": [-0.5, -0.4, -0.3, -0.2],
                       "I": 0.1, "v": 0.0, "n_defesas": 2})
    m2 = pd.DataFrame({"frame": [9], "posse_id": ["z"], "D": [-0.45], "I": 0.1, "v": 0.0, "n_defesas": 1})
    r = r4.momentos_com_dissuasao_mais_negativa({10: m1, 20: m2}, excluir_posses={(10, "b")}, n=3)
    assert list(zip(r["jogo"], r["frame"])) == [(10, 1), (20, 9), (10, 4)]


def test_a_folha_xlsx_cria_se_uma_so_vez_e_nunca_se_reescreve(tmp_path):
    lances = pd.DataFrame({"lance": ["W01", "D01"], "ficheiros": ["W01.png + W01.mp4", "D01.png + D01.mp4"]})
    destino = tmp_path / "revisao.xlsx"
    assert r4.criar_folha_xlsx(lances, destino) is True
    conteudo = destino.read_bytes()
    assert r4.criar_folha_xlsx(lances.iloc[:1], destino) is False and destino.read_bytes() == conteudo
    with zipfile.ZipFile(destino) as z:
        folha = z.read("xl/worksheets/sheet1.xml").decode()
    assert "W01.png + W01.mp4" in folha and "D01" in folha and "sim,em parte,não" in folha
    assert "A ordem dos defesas por valor faz sentido?" in folha and "interceção" in folha
