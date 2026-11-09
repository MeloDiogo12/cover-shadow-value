"""D42: xPass (variáveis, logístico), peso γ da sombra, v3.1/v4 com o xPass e critérios K1–K5."""

import avaliacao_xpass as ax
import calcular_xpass as cx
import numpy as np
import pandas as pd
import pytest
import xpass as xp
from metrica.v4 import avaliar_momento, p_chegar_por_coligacao, valores_por_coligacao

LONGE = xp.LONGE


# ---------------------------------------------------------------- variáveis

def test_caracteristicas_num_exemplo_a_mao():
    # portador em (0, 0), recetor em (10, 0), a atacar para +x; defesas: (5, 2) junto à linha, (5, 8) longe, (-3, 1) atrás do portador
    adv = np.array([[5.0, 2.0], [5.0, 8.0], [-3.0, 1.0]])
    c = xp.caracteristicas((0.0, 0.0), [[10.0, 0.0]], adv, 1)[0]
    comprimento, avanco, dist_linha, n_perto, dist_recetor, dist_portador = c
    assert comprimento == pytest.approx(10.0) and avanco == pytest.approx(10.0)
    assert dist_linha == pytest.approx(2.0)  # o de (-3, 1) cai fora do segmento e não conta
    assert n_perto == 1
    assert dist_recetor == pytest.approx(np.hypot(5, 2))
    assert dist_portador == pytest.approx(np.hypot(3, 1))


def test_avanco_muda_de_sinal_com_o_sentido_de_ataque():
    adv = np.array([[LONGE, LONGE]])
    assert xp.caracteristicas((0, 0), [[10, 0]], adv, -1)[0, 1] == pytest.approx(-10.0)


def test_sem_defesa_perto_valem_o_teto():
    c = xp.caracteristicas((0, 0), [[10, 0]], np.array([[LONGE, LONGE]]), 1)[0]
    assert c[2] == xp.TETO and c[3] == 0 and c[4] == xp.TETO and c[5] == xp.TETO


def test_varias_linhas_de_uma_vez():
    c = xp.caracteristicas((0, 0), [[10, 0], [0, 20]], np.array([[5.0, 1.0]]), 1)
    assert c.shape == (2, 6) and c[0, 3] == 1 and c[1, 3] == 0  # o defesa só está junto à primeira linha


# ---------------------------------------------------------------- logístico

def test_logistico_recupera_coeficientes_conhecidos():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20000, 3)) * [10, 1, 5]
    verdadeiro = np.array([0.8, -0.05, 0.6, 0.1])
    y = (rng.uniform(size=len(X)) < xp.sigmoide(verdadeiro[0] + X @ verdadeiro[1:])).astype(float)
    assert xp.ajustar_logistico(X, y) == pytest.approx(verdadeiro, abs=0.06)


# ---------------------------------------------------------------- γ

def test_ajuste_de_gamma_com_dados_simulados_recupera_gamma_conhecido():
    rng = np.random.default_rng(2)
    n = 40000
    offset, sombra = rng.normal(1.5, 1.0, n), rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < xp.sigmoide(offset - 1.2 * sombra)).astype(float)
    g = xp.ajustar_gamma(offset, sombra, y)
    assert g.gamma == pytest.approx(-1.2, abs=0.15)
    assert g.ic95[0] < -1.2 < g.ic95[1]


def test_gamma_zero_sem_efeito_da_sombra():
    base = np.array([0.5, -1.0, 2.0])
    assert cx.p_do_xpass(base, 0.0, np.array([0.0, 0.5, 1.0])) == pytest.approx(xp.sigmoide(base))


# ---------------------------------------------------------------- v3.1 e v4 com o xPass

def exemplo(semente=0, n_linhas=6, n_defesas=4):
    rng = np.random.default_rng(semente)
    pc = rng.uniform(0, 1, (n_linhas, n_defesas)) * (rng.uniform(size=(n_linhas, n_defesas)) < 0.5)
    return rng.uniform(0, 0.1, n_linhas), rng.normal(size=n_linhas), rng.normal(0, 1.5, n_linhas), pc


def p_coligacoes(base, gamma, pc):
    return cx.p_do_xpass(base, gamma, 1 - p_chegar_por_coligacao(pc))


@pytest.mark.parametrize("semente", range(4))
def test_com_gamma_zero_nao_ha_intercecao_e_a_v4_so_tem_dissuasao(semente):
    """Com γ = 0 as sombras não cortam nenhum passe (I = 0), mas a escolha do portador (M2) continua a depender delas: a ameaça negada da v4 é só
    dissuasão. Na v3.1, que não tem escolha, a ameaça negada é zero (ver o teste do Shapley da linha)."""
    x, eta, base, pc = exemplo(semente)
    p = p_coligacoes(base, 0.0, pc)
    assert p == pytest.approx(np.broadcast_to(xp.sigmoide(base), p.shape))  # p não depende da coligação
    m = avaliar_momento(x, eta, -1.7, pc, p_coligacoes=p)
    assert m.I == pytest.approx(0, abs=1e-12) and np.abs(m.shapley_I).max() == pytest.approx(0, abs=1e-12)
    assert m.D == pytest.approx(m.v)


def test_com_gamma_zero_o_shapley_da_linha_e_zero():
    assert cx.shapley_da_linha(0.05, 0.7, 0.0, np.array([0.4, 0.2])) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("semente", range(4))
def test_com_gamma_negativo_a_v4_nega_perigo_e_i_mais_d_e_v(semente):
    x, eta, base, pc = exemplo(semente)
    m = avaliar_momento(x, eta, -1.7, pc, p_coligacoes=p_coligacoes(base, -1.5, pc))
    assert m.I + m.D == pytest.approx(m.v)
    assert m.shapley_I.sum() + m.shapley_D.sum() == pytest.approx(m.v)  # eficiência do Shapley
    assert m.p0 == pytest.approx(xp.sigmoide(base)) and np.all(m.pC <= m.p0 + 1e-12)


def test_p_coligacoes_igual_ao_do_fundo_reproduz_o_comportamento_anterior():
    x, eta, _base, pc = exemplo(3)
    fundo = np.random.default_rng(0).uniform(0.5, 1, len(x))
    antes = avaliar_momento(x, eta, -1.7, pc, p_fundo=fundo)
    depois = avaliar_momento(x, eta, -1.7, pc, p_coligacoes=p_chegar_por_coligacao(pc) * fundo[None, :])
    assert depois.v == pytest.approx(antes.v) and depois.shapley_I == pytest.approx(antes.shapley_I)
    assert valores_por_coligacao(x, eta, -1.7, pc, p_coligacoes=p_chegar_por_coligacao(pc)).E == pytest.approx(
        valores_por_coligacao(x, eta, -1.7, pc).E)


@pytest.mark.parametrize("gamma", [-2.0, -0.4])
def test_shapley_da_linha_soma_o_valor_da_linha(gamma):
    pc = np.array([0.5, 0.3, 0.1])
    escala, base = 0.08, 1.2
    total = escala * (cx.p_do_xpass(base, gamma, 0.0) - cx.p_do_xpass(base, gamma, 1 - np.prod(1 - pc)))
    assert cx.shapley_da_linha(escala, base, gamma, pc).sum() == pytest.approx(total)


def test_shapley_da_linha_com_um_so_defesa_leva_tudo():
    pc = np.array([0.6])
    esperado = 0.1 * (cx.p_do_xpass(0.3, -1.0, 0.0) - cx.p_do_xpass(0.3, -1.0, 0.6))
    assert cx.shapley_da_linha(0.1, 0.3, -1.0, pc) == pytest.approx([esperado])


# ---------------------------------------------------------------- critérios

def momentos(v, D, n_defesas, frames=None, n_linhas=3, E0=1.0):
    return pd.DataFrame({"frame": frames if frames is not None else range(len(v)), "v": v, "D": D, "n_defesas": n_defesas,
                         "n_linhas": n_linhas, "E0": E0})


def test_k3_aplica_os_limites_entre_os_momentos_com_defesas_em_c():
    bom = momentos([0.1] * 9 + [-0.1], [0.1] * 10, [1] * 10)  # 10 % de v < 0, 0 % de D < 0
    r = ax.k3({1: bom, 2: bom})
    assert r["cumpre"] and r["pct_v_negativa"] == pytest.approx(10.0)
    mau = momentos([0.1] * 8 + [-0.1] * 2, [0.1] * 10, [1] * 10)  # 20 % > 10 %
    assert not ax.k3({1: mau, 2: mau})["cumpre"]
    d_negativo = momentos([0.1] * 10, [-0.1] * 2 + [0.1] * 8, [1] * 10)  # D total do jogo < 0 se o resto for pequeno
    d_negativo.loc[2:, "D"] = 0.001
    assert not ax.k3({1: d_negativo, 2: bom})["cumpre"]


def test_k3_ignora_momentos_sem_defesas_em_c():
    m = momentos([-1.0] * 5 + [0.1] * 5, [0.0] * 10, [0] * 5 + [1] * 5)
    assert ax.k3({1: m})["pct_v_negativa"] == 0.0


def contexto(fases):
    return pd.DataFrame({"frame": range(len(fases)), "terco": "meio", "fase": fases})


def test_k4_razao_entre_blocos_alto_medio_e_baixo():
    fases = ["high_block", "medium_block", "low_block", "low_block"]
    m = momentos([0.02, 0.02, 0.08, 0.08], [0] * 4, [1] * 4)
    r = ax.k4({1: m}, {1: contexto(fases)})
    assert r["razao"] == pytest.approx(0.25) and r["cumpre"]  # 0,02 / 0,08 = 25 %
    r = ax.k4({1: m.assign(v=[0.01, 0.01, 0.08, 0.08])}, {1: contexto(fases)})
    assert not r["cumpre"]


def test_k4_ignora_fases_que_nao_sao_blocos():
    fases = ["high_block", "chaotic", "low_block"]
    m = momentos([0.5, 100.0, 1.0], [0] * 3, [1] * 3)
    assert ax.k4({1: m}, {1: contexto(fases)})["media_alto_e_medio"] == pytest.approx(0.5)


def test_k5_o_maior_shapley_do_lance_tem_de_ser_o_esperado(monkeypatch):
    monkeypatch.setattr(ax, "LANCES", {"C06": (1, 10), "C03": (2, 20)})
    a1 = pd.DataFrame({"frame": [10, 10, 99], "defesa": [7, 8, 7], "shapley": [0.3, 0.1, 5.0]})
    a2 = pd.DataFrame({"frame": [20, 20], "defesa": [7, 8], "shapley": [0.1, 0.3]})
    nomes = {1: {7: "K. Jelacic", 8: "X"}, 2: {7: "Y", 8: "P. Klimala"}}
    assert ax.k5({1: a1, 2: a2}, nomes)["cumpre"]
    a2 = a2.assign(shapley=[0.3, 0.1])
    assert not ax.k5({1: a1, 2: a2}, nomes)["cumpre"]


def test_veredicto_segue_a_regra_da_d42():
    ok = {"k3": {"cumpre": True}, "k4": {"cumpre": True}, "k5": {"cumpre": True}}
    falha = {"k3": {"cumpre": False}, "k4": {"cumpre": True}, "k5": {"cumpre": True}}
    sim, nao = {"cumpre": True}, {"cumpre": False}
    assert ax.veredicto(sim, sim, {"xt": ok, "linhas": falha, "risco": falha})["moedas_que_passam"] == ["xt"]
    assert "v3.1" in ax.veredicto(sim, nao, {"xt": ok, "linhas": ok, "risco": ok})["decisao"]  # K2 falha: nenhuma moeda chega
    assert "autor" in ax.veredicto(sim, sim, {"xt": ok, "linhas": ok, "risco": falha})["decisao"]
    assert "v3.1" in ax.veredicto(sim, sim, {"xt": falha, "linhas": falha, "risco": falha})["decisao"]
