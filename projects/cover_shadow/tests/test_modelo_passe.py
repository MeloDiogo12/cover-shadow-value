"""D40: modelo completo de p (cone + fora do cone), moedas, calibração com dados simulados."""

from dataclasses import replace

import calibracao_passe as cp
import numpy as np
import pandas as pd
import pytest
from metrica import passe
from metrica.chegada import ameaca_negada, valores_de_shapley
from metrica.linhas import avaliar_frame, probabilidade_na_linha
from metrica.parametros import PADRAO
from metrica.v4 import ORDENS, avaliar_momento, valores_por_coligacao
from test_v4 import momento_aleatorio


def defesas_aleatorias(semente, n=6):
    rng = np.random.default_rng(semente)
    return {"x": rng.uniform(-10, 40, n), "y": rng.uniform(-15, 15, n), "vx": rng.uniform(-4, 4, n), "vy": rng.uniform(-4, 4, n),
            "orientacao": rng.uniform(0, 360, n), "sombra": rng.uniform(size=n) < 0.7}


# ---------------------------------------------------------------- p_cone = o da v3; p ≤ 1

@pytest.mark.parametrize("semente", range(6))
def test_p_cone_vetorizado_é_o_da_v3(semente):
    d = defesas_aleatorias(semente)
    rng = np.random.default_rng(100 + semente)
    xy = rng.uniform(-10, 40, (5, 2))
    origem = (0.0, 0.0)
    r = passe.probabilidades_das_linhas(origem, xy, d, PADRAO)
    for l in range(len(xy)):
        px, py = np.linspace(origem[0], xy[l, 0], PADRAO.pontos_por_linha), np.linspace(origem[1], xy[l, 1], PADRAO.pontos_por_linha)
        for k in range(len(d["x"])):
            esperado = 0.0
            if d["sombra"][k]:
                defesa = {"x": d["x"][k], "y": d["y"][k], "orientacao": d["orientacao"][k], "vx": d["vx"][k], "vy": d["vy"][k]}
                esperado = probabilidade_na_linha(defesa, origem, px, py, PADRAO) if _tem_cone(defesa, px, py) else 0.0
            assert r.p_cone[l, k] == pytest.approx(esperado, abs=1e-12)


def _tem_cone(defesa, px, py):
    from metrica.sombra import dentro_do_cone

    return bool(dentro_do_cone(px, py, defesa["x"], defesa["y"], defesa["orientacao"], PADRAO).any())


@pytest.mark.parametrize("semente", range(4))
def test_probabilidades_entre_0_e_1(semente):
    d = defesas_aleatorias(semente, 10)
    r = passe.probabilidades_das_linhas((0.0, 0.0), np.random.default_rng(semente).uniform(-10, 40, (8, 2)), d, PADRAO)
    pf = passe.p_fundo(r.p_fora)
    assert ((r.p_cone >= 0) & (r.p_cone <= 1) & (r.p_fora >= 0) & (r.p_fora <= 1)).all()
    p_total = pf * np.prod(1 - r.p_cone, axis=1)
    assert ((pf >= 0) & (pf <= 1) & (p_total >= 0) & (p_total <= pf + 1e-12)).all()


def test_um_defesa_sem_sombra_so_conta_no_fundo():
    d = defesas_aleatorias(0)
    d["sombra"] = np.zeros(len(d["x"]), bool)
    r = passe.probabilidades_das_linhas((0.0, 0.0), np.array([[20.0, 0.0]]), d, PADRAO)
    assert (r.p_cone == 0).all() and (r.p_fora > 0).any()


# ---------------------------------------------------------------- com p_fora = 0 a v3.1 é a v3

def _frame(defesas_extra=()):
    campos = ["frame", "jogador", "portador", "equipa", "x", "y", "posse_id", "equipa_com_bola", "periodo"]
    linhas = [(1, 10, 10, 1, -30.0, 0.0), (1, 11, 10, 1, 0.0, 0.0), (1, 12, 10, 1, -10.0, 12.0)]
    jogadores = pd.DataFrame([(f, j, p, e, x, y, "8_1", 1, 1) for f, j, p, e, x, y in linhas], columns=campos)
    return jogadores


def test_v3_1_com_fundo_a_zero_coincide_com_a_v3(monkeypatch):
    import calcular_modelo_passe as cmp

    original = cmp.probabilidades_das_linhas

    def sem_fora(*args, **kw):
        r = original(*args, **kw)
        return r._replace(p_fora=np.zeros_like(r.p_fora))

    monkeypatch.setattr(cmp, "probabilidades_das_linhas", sem_fora)
    jogadores = _frame()
    defesas = pd.DataFrame({"jogador": [1, 2], "x": [-15.0, -5.0], "y": [0.5, -3.0], "vx": [0.0, 1.0], "vy": [0.0, 0.0],
                            "orientacao": [180.0, 170.0], "a_marcar": [False, False]})
    coef = np.array([0.0] * 6 + [-1.7])
    r = cmp.avaliar_frame_completo(jogadores, defesas, 1, (105.0, 68.0), set(), {11: 5.0, 12: 5.0}, coef, PADRAO, PADRAO,
                                   limite=35.0, moedas=("xt",))
    linhas = r[0]["xt"][0]
    assert (linhas["p_fundo"] == 1).all()
    origem = (-30.0, 0.0)
    rec = pd.DataFrame({"recetor": [11, 12], "x": [0.0, -10.0], "y": [0.0, 12.0], "valor_xt": linhas["valor_xt"]})
    v3, _ = avaliar_frame(origem, rec, defesas, PADRAO)
    avaliadas = ~linhas["excluida"].to_numpy() & ~v3["excluida"].to_numpy()
    assert avaliadas.any()
    assert linhas["negada_linha"].to_numpy()[avaliadas] == pytest.approx(v3["ameaca_negada"].to_numpy()[avaliadas])


def test_negada_da_v3_1_é_a_diferença_de_p_e_o_shapley_soma():
    p_cone = np.array([0.3, 0.5, 0.0])
    pf, valor = 0.8, 0.05
    com = valor * pf * (1 - np.prod(1 - p_cone))
    assert valores_de_shapley(1.0, p_cone[p_cone > 0]).sum() * valor * pf == pytest.approx(com)
    assert com == pytest.approx(valor * (pf - pf * np.prod(1 - p_cone)))
    assert ameaca_negada(valor * pf, p_cone) == pytest.approx(com)


# ---------------------------------------------------------------- moedas

def test_moeda_linhas_num_exemplo_à_mão():
    # portador em x=0, ataque para +x; adversários de campo em x = -5, 4, 10, 25; recetores em x = 20, 8, -10, 4
    # (um adversário em x=4 não conta para o recetor em x=4: não está estritamente entre os dois)
    r = np.array([[20.0, 0.0], [8.0, 5.0], [-10.0, 0.0], [4.0, 3.0]])
    n = passe.adversarios_ultrapassados((0.0, 0.0), r, np.array([-5.0, 4.0, 10.0, 25.0]), 1)
    assert n.tolist() == [2.0, 1.0, 0.0, 0.0]
    # a atacar para -x vê-se o campo ao contrário
    r2 = r * np.array([-1, 1])
    assert passe.adversarios_ultrapassados((0.0, 0.0), r2, np.array([5.0, -4.0, -10.0, -25.0]), -1).tolist() == [2.0, 1.0, 0.0, 0.0]


def test_valor_esperado_da_moeda_risco_e_sem_custo():
    assert passe.valor_esperado(0.75, 0.04, 0.02) == pytest.approx(0.75 * 0.04 - 0.25 * 0.02)
    assert passe.valor_esperado(0.75, 0.04) == pytest.approx(0.75 * 0.04)
    ganho, custo = passe.ganho_e_custo("xt", np.array([1.0]), np.array([3.0]), np.array([0.5]))
    assert ganho[0] == 1.0 and custo is None
    assert passe.ganho_e_custo("linhas", np.array([1.0]), np.array([3.0]), np.array([0.5]))[0][0] == 3.0
    a, b = passe.ganho_e_custo("risco", np.array([1.0]), np.array([3.0]), np.array([0.5]))
    assert a[0] == 1.0 and b[0] == 0.5
    with pytest.raises(ValueError):
        passe.ganho_e_custo("ouro", np.zeros(1), np.zeros(1), np.zeros(1))


def test_custo_da_perda_usa_a_grelha_do_adversario():
    # perto da baliza de quem ataca para +x (x ≈ +50) o xT do adversário é baixo; perto da baliza própria (x ≈ -50) é alto
    cortes = np.array([[50.0, 0.0], [-50.0, 0.0]])
    c = passe.custo_da_perda(cortes, 1, 105.0, 68.0)
    assert c[1] > c[0] > 0


@pytest.mark.parametrize("ordem", ORDENS)
@pytest.mark.parametrize("semente", range(4))
def test_interceção_mais_dissuasão_com_custo_é_a_ameaça_negada(semente, ordem):
    x, eta, beta, p = momento_aleatorio(semente)
    custo = np.random.default_rng(semente).uniform(0, 0.05, len(x))
    pf = np.random.default_rng(semente + 9).uniform(0.5, 1, len(x))
    v = valores_por_coligacao(x, eta, beta, p, pf, custo)
    I, D = v.decompor(ordem)
    assert I + D == pytest.approx(v.E[0] - v.E)
    m = avaliar_momento(x, eta, beta, p, ordem=ordem, p_fundo=pf, custo=custo)
    assert m.I + m.D == pytest.approx(m.v)
    assert m.shapley_I.sum() + m.shapley_D.sum() == pytest.approx(m.v)


def test_sem_custo_o_resultado_é_o_da_v4_anterior():
    x, eta, beta, p = momento_aleatorio(3)
    a = avaliar_momento(x, eta, beta, p)
    b = avaliar_momento(x, eta, beta, p, custo=None, p_defesa_escolha=None)
    assert (a.E0, a.EC, a.I, a.D) == pytest.approx((b.E0, b.EC, b.I, b.D))


def test_escolha_com_sombras_proprias_so_muda_a_escolha():
    x, eta, beta, p = momento_aleatorio(2)
    a = avaliar_momento(x, eta, beta, p)
    b = avaliar_momento(x, eta, beta, p * 0.5, p_defesa_escolha=p)  # a escolha vê p; o p do passe é metade
    assert b.sC == pytest.approx(a.sC) and b.s0 == pytest.approx(a.s0)
    assert (b.pC >= a.pC - 1e-12).all()


# ---------------------------------------------------------------- calibração com dados simulados

def _dados_sinteticos(n=900, semente=0):
    rng = np.random.default_rng(semente)
    jogo = np.where(np.arange(n) % 2 == 0, 1, 2)
    nd = cp.N_DEFESAS
    origem = rng.uniform(-10, 10, (n, 2))
    destino = origem + rng.uniform(-20, 25, (n, 2))
    x, y = rng.uniform(-30, 30, (n, nd)), rng.uniform(-20, 20, (n, nd))
    vx, vy = rng.uniform(-3, 3, (n, nd)), rng.uniform(-3, 3, (n, nd))
    orientacao = rng.uniform(0, 360, (n, nd))
    sombra = rng.uniform(size=(n, nd)) < 0.6
    P = PADRAO.pontos_por_linha
    t = np.linspace(0, 1, P)[None, :]
    px, py = origem[:, 0:1] + t * (destino[:, 0:1] - origem[:, 0:1]), origem[:, 1:2] + t * (destino[:, 1:2] - origem[:, 1:2])
    from metrica.sombra import dentro_do_cone

    cone = dentro_do_cone(px[:, None, :], py[:, None, :], x[:, :, None], y[:, :, None], orientacao[:, :, None], PADRAO) & sombra[:, :, None]
    return cp.DadosDosPasses(jogo=jogo, posse_id=np.arange(n).astype(str), chegou=np.zeros(n, int), comprimento=np.hypot(*(destino - origem).T),
                             x=x, y=y, vx=vx, vy=vy, existe=np.ones((n, nd), bool), px=px, py=py, origem=origem, no_cone=cone,
                             dist_corpo=np.hypot(px[:, None, :] - x[:, :, None], py[:, None, :] - y[:, :, None]),
                             dist_bola=np.hypot(px - origem[:, 0:1], py - origem[:, 1:2]), sombra=sombra)


def test_a_calibração_recupera_parâmetros_conhecidos():
    d = _dados_sinteticos()
    verdadeiros = {"tempo_reacao": 0.9, "velocidade_defesa": 6.5, "incerteza_tempo": 0.6, "bloqueio_nulo": 1.5}
    p = cp.parametros_de(list(verdadeiros.values()), list(verdadeiros))
    prob = cp.probabilidades(d, p)["completo"]
    assert ((prob >= 0) & (prob <= 1)).all()
    chegou = (np.random.default_rng(5).uniform(size=len(prob)) < prob).astype(int)
    d = d._replace(chegou=chegou)
    ajuste = cp.ajustar(d, ["velocidade_defesa", "incerteza_tempo"], base=replace(PADRAO, tempo_reacao=0.9, bloqueio_nulo=1.5))
    ajustado = dict(zip(ajuste.nomes, ajuste.theta))
    assert ajustado["velocidade_defesa"] == pytest.approx(6.5, rel=0.25)
    assert ajustado["incerteza_tempo"] == pytest.approx(0.6, rel=0.3)
    # e a verosimilhança nos parâmetros verdadeiros não é pior do que nos da literatura
    assert cp.log_verosimilhanca(chegou, prob) > cp.log_verosimilhanca(chegou, cp.probabilidades(d, PADRAO)["completo"])


def test_metricas_e_curva_de_calibração():
    chegou = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1] * 10)
    prob = np.linspace(0.05, 0.95, len(chegou))
    m = cp.metricas(chegou, prob)
    assert m["brier"] == pytest.approx(np.mean((prob - chegou) ** 2)) and m["log_loss"] > 0
    curva = cp.curva_de_calibracao(chegou, prob)
    assert len(curva) == 10 and curva["n"].sum() == len(chegou)
    assert cp.taxa_com_p_baixo(chegou, prob)["passes"] == int((prob < 0.5).sum())
    assert cp.metricas(np.array([0, 1]), np.array([1.0, 0.0]))["log_loss"] < 20  # p = 0/1 não rebenta
