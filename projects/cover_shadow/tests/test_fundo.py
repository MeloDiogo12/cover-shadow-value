"""Defesas de fundo (D39): F vazio reproduz a v4 anterior, `p_fundo` ≤ 1, exemplos feitos à mão com um marcador sobre a linha."""

import numpy as np
import pandas as pd
import pytest
from metrica.chegada import probabilidade_de_chegar_ao_ponto
from metrica.fundo import defesas_de_fundo, p_fundo_das_linhas
from metrica.parametros import PADRAO
from metrica.v4 import ORDENS, avaliar_momento, avaliar_total, valores_por_coligacao
from test_v4 import momento_aleatorio


def recetores(*pontos):
    return pd.DataFrame({"x": [p[0] for p in pontos], "y": [p[1] for p in pontos]})


def defesa(x, y, vx=0.0, vy=0.0):
    return pd.DataFrame({"x": [x], "y": [y], "vx": [vx], "vy": [vy]})


# ---------------------------------------------------------------- F vazio = v4 anterior

def test_com_F_vazio_o_fundo_é_1_em_todas_as_linhas():
    f = defesas_de_fundo(pd.DataFrame({"jogador": [1, 2], "x": [0.0, 1.0], "y": [0.0, 1.0], "vx": [0.0, 0.0], "vy": [0.0, 0.0]}),
                         [1, 2], None)
    assert f.empty
    assert p_fundo_das_linhas((0, 0), recetores((10, 0), (0, 10)), f) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("ordem", ORDENS)
@pytest.mark.parametrize("semente", range(4))
def test_com_F_vazio_a_v4_com_fundo_é_igual_à_anterior(semente, ordem):
    x, eta, beta, p = momento_aleatorio(semente)
    sem = avaliar_momento(x, eta, beta, p, ordem=ordem)
    com = avaliar_momento(x, eta, beta, p, ordem=ordem, p_fundo=np.ones(len(x)))
    assert (com.E0, com.EC, com.I, com.D) == pytest.approx((sem.E0, sem.EC, sem.I, sem.D))
    assert com.shapley_I == pytest.approx(sem.shapley_I) and com.shapley_D == pytest.approx(sem.shapley_D)
    assert com.pC == pytest.approx(sem.pC) and com.sC == pytest.approx(sem.sC)
    leve = avaliar_total(x, eta, beta, p, p_fundo=np.ones(len(x)))
    assert leve.shapley == pytest.approx(avaliar_total(x, eta, beta, p).shapley)


# ---------------------------------------------------------------- p_fundo ≤ 1

def test_p_fundo_fica_entre_0_e_1():
    rng = np.random.default_rng(1)
    r = recetores(*rng.uniform(-30, 30, (12, 2)))
    f = pd.DataFrame({"x": rng.uniform(-30, 30, 6), "y": rng.uniform(-30, 30, 6),
                      "vx": rng.uniform(-6, 6, 6), "vy": rng.uniform(-6, 6, 6)})
    pf = p_fundo_das_linhas((0.0, 0.0), r, f)
    assert pf.shape == (12,) and (pf >= 0).all() and (pf <= 1).all()


def test_mais_defesas_de_fundo_nunca_aumentam_a_chance_de_o_passe_chegar():
    r = recetores((15, 5), (10, -8), (20, 0))
    um = pd.DataFrame({"x": [8.0], "y": [0.0], "vx": [0.0], "vy": [0.0]})
    dois = pd.concat([um, defesa(14.0, 3.0)], ignore_index=True)
    assert (p_fundo_das_linhas((0, 0), r, dois) <= p_fundo_das_linhas((0, 0), r, um) + 1e-12).all()


# ---------------------------------------------------------------- exemplo à mão: marcador em cima da linha

def test_marcador_parado_sobre_o_recetor_corta_o_passe_com_a_conta_de_tempos_da_v3():
    # Portador em (0,0), recetor em (10,0), marcador parado no recetor. Bola: 10/15 s; defesa: 0,7 s de reação e 0 m a correr.
    # Margem no fim da linha = 0,6667 − 0,7 = −0,0333 s (a maior de todos os pontos). p = logística(π/√3 · margem / 0,45).
    margem = 10 / PADRAO.velocidade_bola - PADRAO.tempo_reacao
    p_corta = float(probabilidade_de_chegar_ao_ponto(margem, PADRAO.incerteza_tempo))
    assert p_corta == pytest.approx(1 / (1 + np.exp(np.pi / np.sqrt(3) * 0.0333333 / 0.45)), rel=1e-4)
    pf = p_fundo_das_linhas((0, 0), recetores((10, 0)), defesa(10.0, 0.0))
    assert pf == pytest.approx([1 - p_corta])
    assert 0.5 < pf[0] < 0.6  # ~0,53: um marcador parado sobre o recetor corta cerca de metade dos passes de 10 m


def test_defesa_longe_da_linha_não_a_afeta_e_o_marcador_só_afeta_a_sua_linha():
    r = recetores((10, 0), (0, 25))  # a 2.ª linha vai para o lado oposto ao marcador
    pf = p_fundo_das_linhas((0, 0), r, defesa(10.0, 0.0))
    assert pf[0] < 0.6 and pf[1] > 0.99
    assert p_fundo_das_linhas((0, 0), recetores((10, 0)), defesa(80.0, 80.0)) == pytest.approx([1.0], abs=1e-6)


def test_um_defesa_a_correr_para_a_linha_corta_mais_do_que_parado():
    r = recetores((20, 0))
    parado = p_fundo_das_linhas((0, 0), r, defesa(10.0, 6.0))
    a_correr = p_fundo_das_linhas((0, 0), r, defesa(10.0, 6.0, vx=0.0, vy=-5.0))
    assert a_correr[0] < parado[0]


def test_o_guarda_redes_entra_no_fundo_e_os_de_C_não():
    d = pd.DataFrame({"jogador": [1, 2, 3], "x": [0.0, 5.0, 9.0], "y": [0.0, 0.0, 0.0], "vx": [0.0] * 3, "vy": [0.0] * 3})
    gr = defesa(50.0, 0.0)
    assert list(defesas_de_fundo(d, [1, 3], gr)["x"]) == [5.0, 50.0]  # 2 (fora de C) e o guarda-redes
    assert list(defesas_de_fundo(d, [1, 3], None)["x"]) == [5.0]  # variante sem o guarda-redes


# ---------------------------------------------------------------- o fundo só deixa valor às sombras

def test_com_uma_linha_e_um_defesa_a_ameaça_negada_é_x_vezes_p_fundo_vezes_p_do_defesa():
    # Uma linha, s = 1 (só há uma). E(∅) = x·pf; E(C) = x·pf·(1 − pd); v = x·pf·pd. Sem fundo, v = x·pd.
    x, pf, pd_ = 0.4, 0.5, 0.6
    m = avaliar_momento([x], [0.0], -1.0, [[pd_]], p_fundo=[pf])
    assert (m.E0, m.EC, m.v) == pytest.approx((x * pf, x * pf * (1 - pd_), x * pf * pd_))
    assert m.p0 == pytest.approx([pf]) and m.pC == pytest.approx([pf * (1 - pd_)])
    assert m.shapley_I[0] + m.shapley_D[0] == pytest.approx(x * pf * pd_)
    assert avaliar_momento([x], [0.0], -1.0, [[pd_]]).v == pytest.approx(x * pd_)


def test_o_fundo_não_muda_a_escolha_e_a_identidade_I_mais_D_mantém_se():
    x, eta, beta, p = momento_aleatorio(2)
    pf = np.random.default_rng(5).uniform(0.1, 1, len(x))
    a, b = valores_por_coligacao(x, eta, beta, p), valores_por_coligacao(x, eta, beta, p, pf)
    assert b.s == pytest.approx(a.s)  # a escolha só vê as sombras
    assert b.p_chegar == pytest.approx(a.p_chegar * pf[None, :])
    for ordem in ORDENS:
        I, D = b.decompor(ordem)
        assert I + D == pytest.approx(b.E[0] - b.E)
        m = avaliar_momento(x, eta, beta, p, ordem=ordem, p_fundo=pf)
        assert m.shapley_I.sum() + m.shapley_D.sum() == pytest.approx(m.v)


def test_com_o_fundo_a_ameaça_negada_só_diminui_quando_não_há_dissuasão():
    # beta = 0: a escolha é a mesma e o valor de cada linha é multiplicado pelo fundo (≤ 1)
    x, eta, _, p = momento_aleatorio(4)
    pf = np.random.default_rng(6).uniform(0.1, 1, len(x))
    com = avaliar_momento(x, eta, 0.0, p, p_fundo=pf)
    sem = avaliar_momento(x, eta, 0.0, p)
    assert 0 <= com.v <= sem.v + 1e-12


# ---------------------------------------------------------------- tabelas de comparação e legenda da folha

def test_distribuicao_de_p_fundo_conta_as_linhas_abaixo_de_0_9_e_0_5_por_recetor_marcado():
    import analise_fundo as af

    linhas = pd.DataFrame({"p_fundo": [1.0, 0.95, 0.8, 0.4, 0.2, 0.1], "recetor_marcado": [0, 0, 0, 0, 1, 1]})
    t = af.distribuicao_p_fundo(linhas)
    assert t.loc["todas", "linhas"] == 6 and t.loc["todas", "% com p_fundo < 0.9"] == pytest.approx(100 * 4 / 6)
    assert t.loc["recetor não marcado", "% com p_fundo < 0.5"] == pytest.approx(25.0)
    assert t.loc["recetor marcado", "% com p_fundo < 0.5"] == pytest.approx(100.0)


def test_totais_e_sinais_conta_so_os_momentos_com_defesas_em_C():
    import analise_fundo as af

    m = pd.DataFrame({"n_defesas": [0, 2, 3, 1], "E0": [1.0, 1, 1, 1], "I": [0.0, 0.2, 0.1, 0.1], "D": [0.0, -0.1, 0.1, 0.0],
                      "v": [0.0, 0.1, 0.2, -0.1]})
    t = af.totais_e_sinais(m)
    assert t["momentos com defesas em C"] == 3 and t["% com D < 0"] == pytest.approx(100 / 3) and t["% com v < 0"] == pytest.approx(100 / 3)
    assert t["v"] == pytest.approx(0.2) and t["v / E(∅) (%)"] == pytest.approx(5.0)


def test_a_folha_da_ronda_leva_a_legenda_para_os_treinadores(tmp_path):
    import zipfile

    import revisao_v4 as r4

    destino = tmp_path / "folha.xlsx"
    assert r4.criar_folha_xlsx(pd.DataFrame({"lance": ["W01"], "ficheiros": ["a.png + a.mp4"]}), destino)
    with zipfile.ZipFile(destino) as z:
        folha = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    assert "p = probabilidade de o passe chegar" in folha and "I = valor por cortar a linha" in folha
    assert "D = valor por fazer o portador mudar de escolha" in folha and 'mergeCell ref="A3:F3"' in folha


def test_pct_por_contexto_junta_os_jogos_e_compara_versões():
    import analise_fundo as af

    def lances(negada):
        return pd.DataFrame({"frame": [1, 2, 3], "n_avaliadas": [2, 0, 1], "disponivel": [1.0, 1.0, 1.0], "negada": negada})

    ctx = pd.DataFrame({"frame": [1, 2, 3], "fase": ["alto", "alto", "baixo"]})
    t = af.pct_por_contexto({"a": {1: lances([0.5, 0.1, 0.2]), 2: lances([0.1, 0.1, 0.4])},
                             "b": {1: lances([0.25, 0.1, 0.2]), 2: lances([0.1, 0.1, 0.4])}}, {1: ctx, 2: ctx}, "fase")
    assert t[(1, "a", "% negada")]["alto"] == pytest.approx(50.0) and t[(1, "b", "% negada")]["alto"] == pytest.approx(25.0)
    assert t[("ambos", "a", "% negada")]["alto"] == pytest.approx(30.0)  # (0,5 + 0,1) / (1 + 1)
    assert t[("ambos", "a", "lances")]["alto"] == 2 and t[("ambos", "a", "lances")]["baixo"] == 2  # o lance sem linhas avaliadas fica de fora
