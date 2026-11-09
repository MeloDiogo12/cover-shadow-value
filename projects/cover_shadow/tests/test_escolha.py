"""Tabela de escolhas do passe (fase 8a): cada função com um exemplo pequeno, sem dados reais."""

import numpy as np
import pandas as pd
import pytest
from escolha import (
    comprimento_do_passe,
    conjunto_de_escolhas,
    construir_opcoes,
    distancia_ao_adversario_mais_proximo,
    e_rasteiro,
    faixas_de_comprimento,
    funil,
    juntar_frame_da_escolha,
    marcar_passes,
    opcoes_da_skillcorner,
    passes_do_jogo,
    recetores_marcados,
    versoes_da_sombra,
)


def eventos_de_exemplo() -> pd.DataFrame:
    return pd.DataFrame({
        "event_type": ["player_possession", "player_possession", "player_possession", "passing_option"],
        "event_id": ["a", "b", "c", "o1"], "end_type": ["pass", "pass", "shot", None],
        "period": [1, 1, 1, 1], "team_id": [10, 10, 10, 10], "frame_end": [100, 201, 300, None],
        "player_id": [1, 1, 1, 2], "player_targeted_id": [2, 3, np.nan, None],
        "high_pass": [False, np.nan, np.nan, None], "pass_outcome": ["successful", "unsuccessful", None, None],
        "start_type": ["recovery", "pass_reception", None, None], "pass_distance": [12.0, np.nan, np.nan, None],
    })


def test_passes_do_jogo_so_tem_passes_e_o_recetor_visado():
    p = passes_do_jogo(eventos_de_exemplo())
    assert p["posse_id"].tolist() == ["a", "b"]
    assert p["recetor_visado"].tolist() == [2, 3]
    assert not p["high_pass"].iloc[0]
    assert p["high_pass"].isna().tolist() == [False, True]  # passe falhado: sem informação sobre o ar


def test_frame_da_escolha_e_o_ultimo_alinhado_da_posse_e_o_intervalo_em_segundos():
    passes = pd.DataFrame({"posse_id": ["a", "b", "c"], "frame_passe": [101, 200, 50]})
    tabela = pd.DataFrame({"posse_id": ["a", "a", "b"], "frame": [98, 100, 200]})
    r = juntar_frame_da_escolha(passes, tabela)
    assert r["frame"].tolist()[:2] == [100, 200] and np.isnan(r["frame"].iloc[2])  # "c" sem frame alinhado
    assert r["intervalo_s"].iloc[0] == pytest.approx(0.1) and r["intervalo_s"].iloc[1] == 0


def test_distancia_ao_adversario_mais_proximo_inclui_o_guarda_redes():
    posicoes = pd.DataFrame({
        "frame": [1] * 4, "jogador": [1, 2, 8, 9], "defende": [False, False, True, True],
        "x": [0.0, 10.0, 3.0, 40.0], "y": [0.0, 0.0, 4.0, 0.0],  # 9 é o guarda-redes
    })
    d = distancia_ao_adversario_mais_proximo(posicoes).set_index("recetor")["dist_adversario"]
    assert d[1] == pytest.approx(5.0) and d[2] == pytest.approx(np.hypot(7, 4))


def test_recetor_marcado_so_conta_defesas_a_marcar():
    defesas = pd.DataFrame({"frame": [1, 1, 2], "atacante": [5, 6, 5], "a_marcar": [True, False, False]})
    m = recetores_marcados(defesas)
    assert m[["frame", "recetor"]].values.tolist() == [[1, 5]]


def test_versoes_da_sombra_e_um_menos_p_chegar_e_fica_vazia_sem_pose():
    tronco = pd.DataFrame({"frame": [1, 1], "posse_id": ["a", "a"], "recetor": [2, 3], "motivo": ["", "sem_pose"],
                           "valor_xt": [0.1, 0.2], "p_chegar": [0.75, np.nan], "n_sombras": [1, 0]})
    bola = tronco.assign(p_chegar=[0.5, np.nan], n_sombras=[2, 0])
    v = versoes_da_sombra(tronco, bola)
    assert v["sombra_tronco"].iloc[0] == pytest.approx(0.25) and v["sombra_bola"].iloc[0] == pytest.approx(0.5)
    assert v["sombra_tronco"].isna().iloc[1] and v["n_sombras_bola"].iloc[0] == 2


def passe_e_sombras():
    """Uma equipa a atacar para +x; passador 1 em (0, 0); opções 2 (à frente), 3 (atrás), 4 (fora de jogo)."""
    passes = pd.DataFrame({"posse_id": ["a"], "frame": [10.0], "passador": [1], "recetor_visado": [2],
                           "sentido": [1], "high_pass": pd.array([False], dtype="boolean"), "intervalo_s": [0.0]})
    sombras = pd.DataFrame({
        "frame": [10] * 3, "posse_id": ["a"] * 3, "recetor": [2, 3, 4], "motivo": ["", "sem_pose", "fora_de_jogo"],
        "valor_xt": [0.01, 0.0, 0.05], "n_sombras_tronco": [1, 0, 0], "n_sombras_bola": [0, 0, 0],
        "sombra_tronco": [0.4, np.nan, np.nan], "sombra_bola": [0.1, np.nan, np.nan],
    })
    posicoes = pd.DataFrame({"frame": [10] * 5, "jogador": [1, 2, 3, 4, 9], "x": [0.0, 30.0, -10.0, 40.0, 33.0],
                             "y": [0.0, 0.0, 0.0, 0.0, 0.0], "defende": [False, False, False, False, True]})
    return passes, sombras, posicoes


def test_construir_opcoes_calcula_as_variaveis_de_cada_linha():
    passes, sombras, posicoes = passe_e_sombras()
    marcados = pd.DataFrame({"frame": [10], "recetor": [2], "recetor_marcado": [True]})
    sc = pd.DataFrame({"posse_id": ["a"], "recetor": [2], "xpass_sc": [0.8], "score_sc": [0.7], "visado_sc": [True]})
    o = construir_opcoes(passes, sombras, posicoes, marcados, sc, (105.0, 68.0)).set_index("recetor")
    assert o["escolhida"].to_dict() == {2: 1, 3: 0, 4: 0}
    assert o.loc[2, "comprimento"] == pytest.approx(30) and o.loc[3, "avanco"] == pytest.approx(-10)
    assert o.loc[2, "recetor_marcado"] == 1 and o.loc[3, "recetor_marcado"] == 0
    assert o.loc[2, "dist_adversario"] == pytest.approx(3.0)
    assert o.loc[2, "ganho_xt"] > 0 > o.loc[3, "ganho_xt"]  # o ganho tem sinal; `valor_xt` vem da métrica
    assert o.loc[2, "xpass_sc"] == 0.8 and np.isnan(o.loc[3, "xpass_sc"])


def test_sentido_negativo_inverte_o_avanco():
    passes, sombras, posicoes = passe_e_sombras()
    o = construir_opcoes(passes.assign(sentido=-1), sombras, posicoes, pd.DataFrame(columns=["frame", "recetor", "recetor_marcado"]),
                         pd.DataFrame(columns=["posse_id", "recetor", "xpass_sc", "score_sc", "visado_sc"]),
                         (105.0, 68.0)).set_index("recetor")
    assert o.loc[2, "avanco"] == pytest.approx(-30) and o.loc[3, "avanco"] == pytest.approx(10)


def test_conjunto_tira_fora_de_jogo_e_cruzamento_mas_guarda_sem_pose():
    passes, sombras, posicoes = passe_e_sombras()
    o = construir_opcoes(passes, sombras, posicoes, pd.DataFrame(columns=["frame", "recetor", "recetor_marcado"]),
                         pd.DataFrame(columns=["posse_id", "recetor", "xpass_sc", "score_sc", "visado_sc"]), (105.0, 68.0))
    assert conjunto_de_escolhas(o)["recetor"].tolist() == [2, 3]


def test_marcar_passes_explica_porque_o_visado_saiu_do_conjunto():
    passes = pd.DataFrame({"posse_id": ["a", "b", "c", "d"], "recetor_visado": [2, 4, 5, 6]})
    opcoes = pd.DataFrame({"posse_id": ["a", "a", "b", "b", "c"], "recetor": [2, 3, 4, 3, 3],
                           "escolhida": [1, 0, 1, 0, 0], "motivo": ["", "sem_pose", "fora_de_jogo", "", ""]})
    p = marcar_passes(passes, opcoes).set_index("posse_id")
    assert p["tem_linhas"].tolist() == [True, True, True, False]
    assert p.loc["a", "visado_no_conjunto"] and p.loc["a", "n_sem_pose"] == 1 and not p.loc["a", "conjunto_completo"]
    assert p.loc["b", "motivo_visado"] == "fora_de_jogo" and not p.loc["b", "visado_no_conjunto"]
    assert p.loc["c", "motivo_visado"] == "ausente" and p.loc["b", "n_opcoes"] == 1
    assert not p.loc["d", "tem_linhas"]


def test_comprimento_do_passe_usa_a_linha_quando_nao_ha_pass_distance():
    passes = pd.DataFrame({"posse_id": ["a", "b"], "comprimento_evento": [12.0, np.nan]})
    opcoes = pd.DataFrame({"posse_id": ["a", "b", "b"], "escolhida": [1, 1, 0], "comprimento": [11.0, 25.0, 3.0]})
    assert comprimento_do_passe(passes, opcoes).tolist() == [12.0, 25.0]


def test_skillcorner_fica_com_a_deteccao_que_cobre_o_passe():
    eventos = pd.DataFrame({
        "event_type": ["passing_option"] * 3, "associated_player_possession_event_id": ["a"] * 3,
        "player_id": [2, 2, 3], "frame_start": [10, 90, 10], "frame_end": [50, 110, 20],
        "xpass_completion": [0.1, 0.9, 0.5], "passing_option_score": [0.2, 0.8, 0.4], "targeted": [False, True, False],
    })
    o = opcoes_da_skillcorner(eventos, pd.DataFrame({"posse_id": ["a"], "frame_passe": [100]})).set_index("recetor")
    assert o.loc[2, "xpass_sc"] == 0.9 and o.loc[3, "xpass_sc"] == 0.5


def passes_marcados() -> pd.DataFrame:
    return pd.DataFrame({
        "high_pass": pd.array([False, True, pd.NA, False], dtype="boolean"),
        "tem_linhas": [True, True, True, True], "intervalo_s": [0.0, 0.0, 0.2, 0.6],
        "visado_no_conjunto": [True, True, True, True], "conjunto_completo": [False, True, True, True],
    })


def test_rasteiro_conhece_tres_criterios():
    p = passes_marcados()
    assert e_rasteiro(p, "conhecidos").tolist() == [True, False, False, True]
    assert e_rasteiro(p, "nao_ar").tolist() == [True, False, True, True]
    assert e_rasteiro(p, "todos").all()
    with pytest.raises(ValueError):
        e_rasteiro(p, "outro")


def test_funil_so_desce_e_aplica_o_limite_do_intervalo():
    f = funil(passes_marcados(), limite_s=0.4, criterio="nao_ar")
    assert f.tolist() == [4, 3, 2, 2, 1]  # rasteiros 0, 2, 3; limite corta o 4.º; só o 3.º tem o conjunto completo
    assert f.is_monotonic_decreasing


def test_faixas_de_comprimento_conta_pelo_ar_e_passes_sem_informacao():
    p = pd.DataFrame({"comprimento_passe": [5.0, 15.0, 45.0, 45.0, 45.0],
                      "high_pass": pd.array([False, False, True, False, pd.NA], dtype="boolean")})
    t = faixas_de_comprimento(p)
    ultima = t.iloc[-1]
    assert ultima["passes"] == 3 and ultima["pelo_ar"] == 1 and ultima["sem_informacao"] == 1
    assert ultima["pct_pelo_ar"] == pytest.approx(50.0) and t.iloc[0]["pct_pelo_ar"] == 0


def test_sombra_so_com_defesas_com_pose_preenche_a_linha_que_a_metrica_exclui():
    from escolha import sombras_so_com_defesas_com_pose

    match = {"home_team": {"id": 1}, "home_team_side": ["left_to_right", "right_to_left"], "pitch_length": 104,
             "pitch_width": 68}
    jogadores = pd.DataFrame({
        "frame": 2, "jogador": [10, 11, 20, 21, 22], "equipa": [1, 1, 2, 2, 2], "x": [0.0, 12.0, 8.0, 15.0, 40.0], "y": 0.0,
        "portador": 10, "equipa_com_bola": 1, "periodo": 1, "defende": [False, False, True, True, True]})
    defesas = pd.DataFrame({"frame": 2, "jogador": [20, 21, 22], "x": [8.0, 15.0, 40.0], "y": 0.0,
                            "orientacao": [180.0, np.nan, 0.0],
                            "a_marcar": False})  # o 21 não tem pose: a métrica exclui a linha (D13)
    r = sombras_so_com_defesas_com_pose(jogadores, defesas, "tronco", match, {2})
    assert r["recetor"].tolist() == [11]
    assert r["sombra_tronco_pose"].iloc[0] == pytest.approx(1.0)  # o 20 está em cima da linha: corpo a tapá-la
