"""D41: calibração do fundo com os 20 jogos (passes, «cortado», modelo do fundo, ECE, critérios)."""

import calibracao_fundo as cf
import calibracao_passe as cp
import numpy as np
import pandas as pd
import pytest
from metrica.parametros import PADRAO, Parametros
from metrica.passe import pontos_das_linhas


def evento(n, resultado, comprimento_alto=False, recetor=7.0, fim="pass", tipo="player_possession"):
    return {"event_id": f"e{n}", "index": n, "event_type": tipo, "end_type": fim, "pass_outcome": resultado, "high_pass": comprimento_alto,
            "player_targeted_id": recetor, "frame_start": n * 10, "frame_end": n * 10 + 5, "period": 1, "team_id": 1, "player_id": 9,
            "x_start": 0.0, "y_start": 0.0}


def test_passes_rasteiros_filtra_fora_de_jogo_pelo_ar_e_sem_recetor():
    eventos = pd.DataFrame([
        evento(1, "successful"), evento(2, "unsuccessful"), evento(3, "offside"), evento(4, "successful", comprimento_alto=True),
        evento(5, "successful", recetor=np.nan), evento(6, "successful", fim="shot"), evento(7, None, tipo="passing_option"),
    ])
    p, funil = cf.passes_rasteiros(eventos)
    assert list(p["posse_id"]) == ["e1", "e2"]
    assert list(p["chegou"]) == [1, 0]
    assert funil["fora_de_jogo"] == 1 and funil["pelo_ar"] == 1 and funil["sem_recetor_visado"] == 1
    assert list(p["frame"]) == [15, 25]  # o frame do passe é o `frame_end`


def test_passe_falhado_sem_high_pass_entra():
    eventos = pd.DataFrame([evento(1, "unsuccessful", comprimento_alto=np.nan)])
    p, _ = cf.passes_rasteiros(eventos)
    assert len(p) == 1


def test_distancia_ao_segmento():
    d, t = cf.distancia_ao_segmento(np.array([5.0, -3.0, 13.0]), np.array([2.0, 0.0, 0.0]), 0.0, 0.0, 10.0, 0.0)
    assert d == pytest.approx([2.0, 3.0, 3.0])  # ao lado, antes do início, depois do fim (ao extremo)
    assert t == pytest.approx([0.5, -0.3, 1.3])


def passes_de_teste(**mudar):
    base = {"resultado": "unsuccessful", "equipa": 1, "equipa_seguinte": 2, "x_seguinte": 5.0, "y_seguinte": 1.0,
            "x_passador": 0.0, "y_passador": 0.0, "x_recetor": 10.0, "y_recetor": 0.0}
    return pd.DataFrame([{**base, **mudar}])


def test_cortado_so_se_defesa_apanha_a_bola_perto_da_linha_antes_do_recetor():
    assert cf.marcar_cortados(passes_de_teste()).iloc[0] == 1  # defesa a 1 m da linha, a meio
    assert np.isnan(cf.marcar_cortados(passes_de_teste(y_seguinte=8.0)).iloc[0])  # longe da linha: fica de fora
    assert np.isnan(cf.marcar_cortados(passes_de_teste(equipa_seguinte=1)).iloc[0])  # a equipa do passe volta a ter a bola
    assert np.isnan(cf.marcar_cortados(passes_de_teste(x_seguinte=12.0)).iloc[0])  # depois do recetor
    assert cf.marcar_cortados(passes_de_teste(resultado="successful", equipa_seguinte=1)).iloc[0] == 0


def test_cortado_limite_de_5_m():
    assert cf.marcar_cortados(passes_de_teste(y_seguinte=4.9)).iloc[0] == 1
    assert np.isnan(cf.marcar_cortados(passes_de_teste(y_seguinte=5.0)).iloc[0])


def test_posse_seguinte_usa_a_ordem_dos_frames():
    eventos = pd.DataFrame([evento(1, "unsuccessful"), {**evento(2, None, fim="shot"), "team_id": 2, "x_start": 4.0, "y_start": 1.5}])
    p, _ = cf.passes_rasteiros(eventos)
    s = cf.posse_seguinte(eventos, p)
    assert (s.loc[0, "equipa_seguinte"], s.loc[0, "x_seguinte"], s.loc[0, "y_seguinte"]) == (2, 4.0, 1.5)


# ---------------------------------------------------------------- modelo do fundo

def dados_simulados(n=40, semente=0, p: Parametros = PADRAO, simular_resultado=False):
    rng = np.random.default_rng(semente)
    origem = rng.uniform(-10, 10, (n, 2))
    destino = origem + rng.uniform(-30, 30, (n, 2))
    px, py = np.empty((n, p.pontos_por_linha)), np.empty((n, p.pontos_por_linha))
    for k in range(n):
        a, b = pontos_das_linhas(origem[k], destino[k:k + 1], p.pontos_por_linha)
        px[k], py[k] = a[0], b[0]
    x, y = rng.uniform(-20, 30, (n, 11)), rng.uniform(-25, 25, (n, 11))
    existe = np.ones((n, 11), bool)
    existe[0, 8:] = False  # um passe com defesas em falta
    d = cf.DadosFundo(jogo=np.repeat([1, 2], n // 2), chegou=np.ones(n, int), cortado=np.full(n, np.nan),
                      comprimento=np.hypot(*(destino - origem).T), dist_defesa_linha=rng.uniform(0, 10, n),
                      x=np.where(existe, x, 1e6), y=np.where(existe, y, 1e6), vx=rng.uniform(-3, 3, (n, 11)), vy=rng.uniform(-3, 3, (n, 11)),
                      existe=existe, px=px, py=py, dist_bola=np.hypot(px - origem[:, :1], py - origem[:, 1:]))
    if simular_resultado:
        d = d._replace(chegou=(rng.uniform(size=n) < cf.p_fundo(d, p)).astype(int))
    return d


def test_p_fundo_coincide_com_o_fundo_do_modelo_completo_sem_cones():
    """Sem cones o `p_completo` da 8f é o fundo: tem de dar o mesmo que o p_fundo desta fase."""
    d = dados_simulados()
    n = len(d.chegou)
    d8 = cp.DadosDosPasses(
        jogo=d.jogo, posse_id=np.arange(n), chegou=d.chegou, comprimento=d.comprimento, x=d.x, y=d.y, vx=d.vx, vy=d.vy, existe=d.existe,
        px=d.px, py=d.py, origem=np.zeros((n, 2)), no_cone=np.zeros((n, 11, d.px.shape[1]), bool),
        dist_corpo=np.hypot(d.px[:, None, :] - d.x[:, :, None], d.py[:, None, :] - d.y[:, :, None]), dist_bola=d.dist_bola,
        sombra=np.zeros((n, 11), bool))
    p = cf.parametros_de([0.9, 6.0, 0.6])
    assert cf.p_fundo(d, p) == pytest.approx(cp.probabilidades(d8, p)["completo"], abs=1e-12)


def test_p_fundo_em_blocos_é_igual_a_tudo_de_uma_vez():
    d = dados_simulados()
    assert cf.p_fundo(d, PADRAO, bloco=7) == pytest.approx(cf.p_fundo(d, PADRAO, bloco=1000), abs=1e-14)


def test_p_fundo_sem_defesas_é_um_e_decresce_com_mais_rapidez_ou_menos_reacao():
    d = dados_simulados()
    assert 0 <= cf.p_fundo(d, PADRAO).min() and cf.p_fundo(d, PADRAO).max() <= 1
    sem = d._replace(existe=np.zeros_like(d.existe))
    assert cf.p_fundo(sem, PADRAO) == pytest.approx(1.0)
    mais_rapido = cf.parametros_de([0.7, 8.0, 0.45])
    assert np.all(cf.p_fundo(d, mais_rapido) <= cf.p_fundo(d, PADRAO) + 1e-12)
    menos_reacao = cf.parametros_de([0.3, 5.0, 0.45])
    assert cf.p_fundo(d, menos_reacao).mean() < cf.p_fundo(d, PADRAO).mean()


def test_ajuste_melhora_a_verosimilhanca_e_fica_dentro_dos_limites():
    d = dados_simulados(n=300, simular_resultado=True, p=cf.parametros_de([0.5, 7.0, 0.4]))
    y = d.chegou.astype(float)
    a = cf.ajustar(d, y, pontos_iniciais=1)
    assert a.log_verosimilhanca >= cp.log_verosimilhanca(y, cf.p_fundo(d, PADRAO)) - 1e-6
    for v, n in zip(a.theta, cf.NOMES):
        assert cf.LIMITES[n][0] <= v <= cf.LIMITES[n][1]
    assert cf.no_limite(np.array([0.05, 5.0, 0.45])) == ["tempo_reacao"]
    assert cf.no_limite(np.array([0.7, 5.0, 0.45])) == []


def test_logistico_recupera_coeficientes():
    rng = np.random.default_rng(3)
    n = 6000
    comp, dist = rng.uniform(5, 35, n), rng.uniform(0, 10, n)
    verdadeiro = np.array([0.5, -0.05, 0.3])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(verdadeiro[0] + verdadeiro[1] * comp + verdadeiro[2] * dist)))).astype(float)
    d = dados_simulados(n=n)._replace(comprimento=comp, dist_defesa_linha=dist)
    b = cf.ajustar_logistico(d, y)
    assert b == pytest.approx(verdadeiro, abs=0.12)
    assert 0 < cf.prever_logistico(b, d).min() and cf.prever_logistico(b, d).max() < 1


# ---------------------------------------------------------------- ECE e critérios

def test_ece_zero_se_calibrado_e_igual_à_diferença_se_enviesado():
    prob = np.repeat(np.linspace(0.05, 0.95, 10), 200)
    rng = np.random.default_rng(1)
    chegou = (rng.uniform(size=len(prob)) < prob).astype(float)
    assert cf.ece(chegou, prob) < 0.03
    # tudo previsto a 0,9 (em decis por ordem) com 70 % de sucesso: ECE = 0,2
    chegou2 = np.tile([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], 100).astype(float)
    assert cf.ece(chegou2, np.full(1000, 0.9)) == pytest.approx(0.2)


def theta_estavel(k=20):
    return np.tile([0.7, 5.0, 0.45], (k, 1)) * (1 + np.linspace(-0.05, 0.05, k))[:, None]


def test_criterios_aplicam_a_regra_à_letra():
    alvo = (np.arange(1000) % 10 < 9).astype(float)
    prob = np.full(1000, 0.9)
    total = np.array([0.7, 5.0, 0.45])
    ok = cf.criterios(np.full(20, 0.01), theta_estavel(), total, alvo, prob)
    assert ok["veredicto"] == "calibrável"
    # 16 jogos a bater a base chega; 15 não
    melhorias = np.r_[np.full(16, 0.01), np.full(4, -0.001)]
    assert cf.criterios(melhorias, theta_estavel(), total, alvo, prob)["criterio_1_bate_a_taxa_base"]["cumpre"]
    melhorias15 = np.r_[np.full(15, 0.01), np.full(5, -0.001)]
    assert not cf.criterios(melhorias15, theta_estavel(), total, alvo, prob)["criterio_1_bate_a_taxa_base"]["cumpre"]
    # 16 a bater mas a melhoria média negativa não chega
    assert not cf.criterios(np.r_[np.full(16, 0.001), np.full(4, -0.1)], theta_estavel(), total, alvo, prob)["criterio_1_bate_a_taxa_base"]["cumpre"]
    # critério 2: um parâmetro 21 % afastado, ou no limite
    fora = theta_estavel()
    fora[3, 1] = 5.0 * 1.21
    assert not cf.criterios(np.full(20, 0.01), fora, total, alvo, prob)["criterio_2_parametros_estaveis"]["cumpre"]
    limite = theta_estavel()
    limite[0, 0] = 0.05
    assert not cf.criterios(np.full(20, 0.01), limite, np.array([0.05, 5.0, 0.45]), alvo, prob)["criterio_2_parametros_estaveis"]["cumpre"]
    # critério 3: ECE > 0,03
    assert not cf.criterios(np.full(20, 0.01), theta_estavel(), total, alvo, np.full(1000, 0.8))["criterio_3_calibracao"]["cumpre"]
    assert cf.criterios(np.full(20, 0.01), theta_estavel(), total, alvo, np.full(1000, 0.8))["veredicto"] == "não calibrável com estes dados"


def test_validar_em_dobras_sem_paralelismo():
    d = dados_simulados(n=60, simular_resultado=True, p=cf.parametros_de([0.5, 7.0, 0.4]))
    r = cf.validar(d, d.chegou.astype(float), processos=1)
    assert [f["jogo"] for f in r["por_jogo"]] == [1, 2]
    assert r["criterios"]["criterio_1_bate_a_taxa_base"]["n_jogos"] == 2
    assert len(r["previsoes"]) == 60
    assert set(r["metricas_juntas"]) == {"modelo", "literatura", "taxa_base", "logistico"}
