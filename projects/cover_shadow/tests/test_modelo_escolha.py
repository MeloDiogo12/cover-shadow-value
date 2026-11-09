"""Logit condicional da escolha do passe (fase 8b): dados simulados com coeficientes conhecidos e pequenos exemplos."""

import numpy as np
import pandas as pd
import pytest
from modelo_escolha import (
    BASE,
    Ajuste,
    ajustar,
    avaliar_passes,
    avaliar_probabilidades,
    fora_da_amostra,
    intervalo_bootstrap,
    montar,
    passes_com_opcoes_da_skillcorner,
    preparar_dados,
    probabilidades,
    referencia_ao_acaso,
    renormalizar,
    tabela_de_coeficientes,
)


def simular(n_passes=3000, coef=(-0.15, 1.2), n_linhas=8, semente=1, jogo=1) -> pd.DataFrame:
    """Cada passe tem `n_linhas` com duas variáveis; o portador escolhe pela regra do logit condicional."""
    rng = np.random.default_rng(semente)
    X = rng.normal(size=(n_passes, n_linhas, 2)) * [10, 1]
    pontuacao = X @ np.array(coef) + rng.gumbel(size=(n_passes, n_linhas))  # Gumbel = escolha por softmax
    escolhida = np.zeros((n_passes, n_linhas), int)
    escolhida[np.arange(n_passes), pontuacao.argmax(axis=1)] = 1
    return pd.DataFrame({"passe": np.repeat(np.arange(n_passes), n_linhas), "jogo": jogo, "a": X[:, :, 0].ravel(),
                         "b": X[:, :, 1].ravel(), "escolhida": escolhida.ravel(), "recetor": np.tile(np.arange(n_linhas), n_passes)})


def test_o_modelo_recupera_os_coeficientes_conhecidos():
    a = ajustar(simular(), ["a", "b"])
    erro = np.sqrt(np.diag(a.cov))
    assert a.coef == pytest.approx([-0.15, 1.2], abs=4 * erro.max())
    assert erro.max() < 0.05  # com 3000 passes o erro-padrão é pequeno
    assert tabela_de_coeficientes(a).loc["b", "razao_de_chances"] == pytest.approx(np.exp(a.coef[1]))


def test_probabilidades_somam_um_em_cada_passe():
    t = simular(n_passes=50)
    d = montar(t, ["a", "b"])
    p = probabilidades(d, np.array([-0.1, 0.7]))
    assert pd.Series(p).groupby(t["passe"].to_numpy()).sum().to_numpy() == pytest.approx(1.0)


def test_sem_informacao_o_log_loss_e_o_do_acaso_e_top1_e_um_sobre_n():
    t = simular(n_passes=200, n_linhas=5)
    sem_efeito = Ajuste(["a"], np.zeros(1), np.eye(1), 0.0, 200)  # coeficiente nulo: todas as linhas iguais
    assert avaliar_passes(t, sem_efeito)["log_loss"].mean() == pytest.approx(np.log(5))
    assert referencia_ao_acaso(t) == pytest.approx({"log_loss": np.log(5), "top1": 0.2, "top3": 0.6})


def test_top_k_conta_as_linhas_mais_provaveis_que_a_escolhida():
    t = pd.DataFrame({"passe": [0, 0, 0, 0], "a": [3.0, 2.0, 1.0, 0.0], "escolhida": [0, 0, 1, 0]})
    a = Ajuste(["a"], np.array([1.0]), np.eye(1), 0.0, 1)
    r = avaliar_passes(t, a).iloc[0]
    assert not r["top1"] and r["top3"]  # a escolhida é a 3.ª mais provável
    assert r["log_loss"] == pytest.approx(-np.log(np.exp(1) / np.exp([3, 2, 1, 0]).sum()))


def test_fora_da_amostra_treina_num_jogo_e_testa_no_outro():
    d = pd.concat([simular(800, semente=1, jogo=1), simular(800, semente=2, jogo=2)], ignore_index=True)
    d["passe"] = d["jogo"].astype(str) + "_" + d["passe"].astype(str)
    ruido = np.random.default_rng(0).normal(size=(len(d), len(BASE) + 2))  # variáveis sem relação com a escolha
    for i, v in enumerate([*BASE, "sombra_bola", "sombra_tronco"]):
        d[v] = ruido[:, i]
    d["comprimento"], d["avanco"] = d["a"], d["b"]
    r = fora_da_amostra(d)
    assert set(r) == {"1_para_2", "2_para_1"} and r["1_para_2"]["teste"] == 2
    m = r["1_para_2"]["modelos"]
    assert m["M0"]["log_loss"] < r["1_para_2"]["ao_acaso"]["log_loss"]  # as variáveis informam
    assert m["M0"]["n_passes"] == m["M1"]["n_passes"] == m["M2"]["n_passes"] == 800
    assert r["1_para_2"]["M2_menos_M1"]["ic95_inf"] <= r["1_para_2"]["M2_menos_M1"]["diferenca"] <= r["1_para_2"]["M2_menos_M1"]["ic95_sup"]


def test_bootstrap_do_intervalo_cobre_a_media_e_e_reprodutivel():
    dif = np.random.default_rng(3).normal(-0.05, 0.1, size=400)
    r = intervalo_bootstrap(dif, repeticoes=500)
    assert r["ic95_inf"] < r["diferenca"] < r["ic95_sup"] and r["ic95_sup"] < 0
    assert intervalo_bootstrap(dif, repeticoes=500) == r


def opcoes_de_exemplo() -> pd.DataFrame:
    """Passes: a rasteiro completo; b pelo ar; c falhado de 40 m; d com 1 linha sem pose; e com frame tardio."""
    linhas = []
    for posse, high, comp_passe, intervalo, sem_pose in [("a", False, 20, 0.0, 0), ("b", True, 20, 0.0, 0), ("c", None, 40, 0.0, 0),
                                                         ("d", None, 12, 0.1, 1), ("e", False, 15, 0.6, 0)]:
        for recetor in (1, 2, 3):
            linhas.append({"jogo": 1, "posse_id": posse, "recetor": recetor, "escolhida": int(recetor == 1),
                           "high_pass": high, "comprimento_passe": comp_passe, "comprimento": 10.0 * recetor,
                           "intervalo_s": intervalo, "motivo": "sem_pose" if sem_pose and recetor == 2 else "",
                           "sombra_tronco_pose": 0.2, "sombra_bola_pose": 0.3})
    o = pd.DataFrame(linhas)
    o["high_pass"] = pd.array(o["high_pass"], dtype="boolean")
    return o


def test_preparar_dados_aplica_rasteiro_comprimento_intervalo_e_marca_linhas_sem_pose():
    d = preparar_dados(opcoes_de_exemplo())
    assert sorted(d["posse_id"].unique()) == ["a", "d"]  # b pelo ar, c longo, e fora do intervalo
    assert d["comprimento"].max() <= 35 and d["passe"].iloc[0] == "1_a"
    assert d.loc[d["posse_id"] == "d", "linha_sem_pose"].tolist() == [0, 1, 0]
    assert sorted(preparar_dados(opcoes_de_exemplo(), so_conjuntos_completos=True)["posse_id"].unique()) == ["a"]


def test_passe_cuja_linha_escolhida_passa_o_limite_de_comprimento_sai():
    o = opcoes_de_exemplo()
    o.loc[(o["posse_id"] == "a") & (o["recetor"] == 1), "comprimento"] = 36.0
    assert "a" not in preparar_dados(o)["posse_id"].unique()


def test_m3_leva_as_duas_sombras_e_a_reparametrização_não_muda_a_verosimilhança():
    t = simular(n_passes=1500, coef=(-0.15, 1.2), semente=4)
    t["sombra_bola"], t["sombra_tronco"] = t["b"] * 0.2, t["b"] * 0.2 + np.random.default_rng(1).normal(size=len(t))
    t["tronco_menos_bola"] = t["sombra_tronco"] - t["sombra_bola"]
    m3 = ajustar(t, ["a", "sombra_bola", "sombra_tronco"])
    dif = ajustar(t, ["a", "sombra_bola", "tronco_menos_bola"])
    assert m3.log_verosimilhanca == pytest.approx(dif.log_verosimilhanca)
    assert dif.coef[2] == pytest.approx(m3.coef[2])  # o coeficiente da diferença é o do tronco
    assert dif.coef[1] == pytest.approx(m3.coef[1] + m3.coef[2])  # e o da bola absorve o do tronco


def test_renormalizar_dá_distribuições_por_passe_e_o_score_só_conta_pela_ordem_no_top_k():
    t = pd.DataFrame({"passe": [0, 0, 0, 1, 1], "escolhida": [0, 1, 0, 1, 0], "score_sc": [0.9, 0.6, 0.3, 0.8, 0.7]})
    p = renormalizar(t["score_sc"].to_numpy(), t)
    assert pd.Series(p).groupby(t["passe"]).sum().to_numpy() == pytest.approx(1.0)
    r = avaliar_probabilidades(t, t["score_sc"].to_numpy())
    assert list(r["top1"]) == [False, True] and list(r["top3"]) == [True, True]


def test_comparacao_so_usa_passes_com_pelo_menos_duas_opcoes_e_a_escolhida_listada():
    t = pd.DataFrame({"passe": ["a", "a", "b", "c", "c"], "escolhida": [1, 0, 1, 0, 0],
                      "score_sc": [0.9, 0.7, 0.8, 0.9, np.nan]})
    assert list(passes_com_opcoes_da_skillcorner(t)["passe"]) == ["a", "a"]
