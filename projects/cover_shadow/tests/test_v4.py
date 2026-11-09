"""Métrica v4 (D38): identidade I + D = v nas três ordens, simétrica = média, eficiência do Shapley, ausência de sombras e exemplos feitos à mão."""

import numpy as np
import pytest
from metrica.v4 import (
    ORDENS,
    avaliar_momento,
    p_chegar_por_coligacao,
    shapley_exato,
    valores_por_coligacao,
)


def momento_aleatorio(semente=0, n_linhas=7, n_defesas=5):
    rng = np.random.default_rng(semente)
    x = rng.uniform(0, 0.1, n_linhas)
    eta = rng.normal(size=n_linhas)
    p = rng.uniform(0, 1, (n_linhas, n_defesas)) * (rng.uniform(size=(n_linhas, n_defesas)) < 0.4)
    return x, eta, -1.7, p


@pytest.mark.parametrize("ordem", ORDENS)
@pytest.mark.parametrize("semente", range(5))
def test_interceção_mais_dissuasão_é_a_ameaça_negada_em_todas_as_coligações(semente, ordem):
    x, eta, beta, p = momento_aleatorio(semente)
    v = valores_por_coligacao(x, eta, beta, p)
    I, D = v.decompor(ordem)
    assert I + D == pytest.approx(v.E[0] - v.E)


@pytest.mark.parametrize("semente", range(5))
def test_a_simétrica_é_a_média_das_duas_ordens(semente):
    v = valores_por_coligacao(*momento_aleatorio(semente))
    (I1, D1), (I2, D2), (I, D) = (v.decompor(o) for o in ORDENS)
    assert I == pytest.approx((I1 + I2) / 2) and D == pytest.approx((D1 + D2) / 2)
    m = avaliar_momento(*momento_aleatorio(semente))  # o Shapley também (é linear)
    a, b = m.por_ordem["ordem1"], m.por_ordem["ordem2"]
    assert m.shapley_I == pytest.approx((a.shapley_I + b.shapley_I) / 2)
    assert m.shapley_D == pytest.approx((a.shapley_D + b.shapley_D) / 2)


@pytest.mark.parametrize("ordem", ORDENS)
@pytest.mark.parametrize("semente", range(5))
def test_shapley_é_eficiente_para_I_e_para_D(semente, ordem):
    m = avaliar_momento(*momento_aleatorio(semente), ordem=ordem)
    assert m.shapley_I.sum() == pytest.approx(m.I)
    assert m.shapley_D.sum() == pytest.approx(m.D)
    assert m.shapley_I.sum() + m.shapley_D.sum() == pytest.approx(m.v)


def test_ordem_desconhecida_dá_erro():
    with pytest.raises(ValueError):
        avaliar_momento(*momento_aleatorio(), ordem="outra")


def test_sem_sombras_tudo_é_zero():
    x, eta, beta, _ = momento_aleatorio()
    m = avaliar_momento(x, eta, beta, np.zeros((len(x), 3)))
    assert (m.I, m.D, m.v) == pytest.approx((0, 0, 0))
    assert m.shapley_I == pytest.approx(0) and m.shapley_D == pytest.approx(0)
    assert m.E0 == pytest.approx(m.EC)
    sem_defesas = avaliar_momento(x, eta, beta, np.zeros((len(x), 0)))
    assert sem_defesas.v == pytest.approx(0) and len(sem_defesas.shapley_I) == 0


def test_com_beta_zero_a_escolha_não_muda_e_a_dissuasão_é_nula():
    x, eta, _, p = momento_aleatorio(3)
    m = avaliar_momento(x, eta, 0.0, p)
    assert m.D == pytest.approx(0) and m.shapley_D == pytest.approx(0)
    assert m.sC == pytest.approx(m.s0)


def test_p_chegar_por_coligacao_multiplica_as_falhas_dos_membros():
    p = np.array([[0.5, 0.5], [0.0, 1.0]])  # coligações: {}, {A}, {B}, {A,B}
    assert p_chegar_por_coligacao(p) == pytest.approx(np.array([[1, 1], [0.5, 1], [0.5, 0], [0.25, 0]]))


def test_exemplo_à_mão_com_2_defesas_e_3_linhas_sem_dissuasão():
    # 3 linhas igualmente prováveis (beta = 0, eta = 0 → s = 1/3). A afeta a linha 1 (p 0,5); B afeta a 1 (0,5) e a 2 (1).
    p = np.array([[0.5, 0.5], [0.0, 1.0], [0.0, 0.0]])
    x = np.array([0.3, 0.6, 0.9])
    m = avaliar_momento(x, np.zeros(3), 0.0, p)
    assert m.E0 == pytest.approx(1.8 / 3)
    assert m.I == pytest.approx((0.3 * 0.75 + 0.6) / 3)  # 0,275
    assert m.D == pytest.approx(0)
    # I({A}) = 0,05; I({B}) = 0,25; I({A,B}) = 0,275 → Shapley: A = ½·0,05 + ½·0,025; B = ½·0,25 + ½·0,225
    assert m.shapley_I == pytest.approx([0.0375, 0.2375])
    for ordem in ORDENS:  # sem dissuasão (s constante) as três ordens coincidem
        assert avaliar_momento(x, np.zeros(3), 0.0, p, ordem=ordem).shapley_I == pytest.approx([0.0375, 0.2375])


def test_exemplo_à_mão_com_dissuasão_negativa():
    # (ordem 1)
    # 2 linhas de xT 1; um defesa tapa totalmente a linha 1. Sem sombra s = (½, ½); com sombra, beta = −ln 3 dá
    # pontuações (−ln 3, 0) → s = (¼, ¾). E(∅) = 1; E(C) = ¾; v = ¼.
    # Interceção = ½ · 1 = ½ (metade das vezes a escolha era a linha 1, que já não chega).
    # Dissuasão = (½ − ¾) · 1 = −¼ na linha 2 livre: o portador passa a escolhê-la mais, o que devolve perigo.
    m = avaliar_momento([1.0, 1.0], [0.0, 0.0], -np.log(3), [[1.0], [0.0]], ordem="ordem1")
    assert (m.E0, m.EC, m.v) == pytest.approx((1.0, 0.75, 0.25))
    assert (m.I, m.D) == pytest.approx((0.5, -0.25))
    assert m.sC == pytest.approx([0.25, 0.75])
    assert (m.shapley_I[0], m.shapley_D[0]) == pytest.approx((0.5, -0.25))  # com um só defesa, o Shapley é o total


def test_exemplo_à_mão_do_mesmo_caso_nas_três_ordens():
    # Mesmo caso (s0 = ½,½; s = ¼,¾; linha 1 tapada; x = 1; v = ¼), agora nas três ordens.
    # Ordem 1: I = Σ s0 (1 − p) x = ½;           D = Σ (s0 − s) p x = (½ − ¾) · 1 = −¼.
    # Ordem 2: I = Σ s (1 − p) x = ¼ · 1 = ¼;    D = Σ (s0 − s) x = (¼) + (−¼) = 0.
    # Simétrica: I = (½ + ¼)/2 = ⅜;              D = (−¼ + 0)/2 = −⅛.
    esperado = {"ordem1": (0.5, -0.25), "ordem2": (0.25, 0.0), "simetrica": (0.375, -0.125)}
    for ordem, (I, D) in esperado.items():
        m = avaliar_momento([1.0, 1.0], [0.0, 0.0], -np.log(3), [[1.0], [0.0]], ordem=ordem)
        assert (m.I, m.D) == pytest.approx((I, D))
        assert (m.shapley_I[0], m.shapley_D[0]) == pytest.approx((I, D))
        assert m.I + m.D == pytest.approx(0.25)
        assert (m.por_ordem[ordem].I, m.por_ordem[ordem].D) == pytest.approx((I, D))


def test_shapley_de_uma_função_aditiva_é_a_própria_parcela():
    # f(S) = soma das parcelas dos membros: cada jogador recebe a sua parcela
    parcelas = np.array([1.0, 2.0, 4.0])
    valor = np.array([sum(parcelas[i] for i in range(3) if k >> i & 1) for k in range(8)])
    assert shapley_exato(valor) == pytest.approx(parcelas)


@pytest.mark.parametrize("semente", range(5))
def test_o_total_leve_coincide_com_a_soma_de_I_e_D_do_momento_completo(semente):
    from metrica.v4 import avaliar_total

    x, eta, beta, p = momento_aleatorio(semente)
    completo = avaliar_momento(x, eta, beta, p)
    leve = avaliar_total(x, eta, beta, p)
    assert leve.E0 == pytest.approx(completo.E0) and leve.EC == pytest.approx(completo.EC)
    assert leve.v == pytest.approx(completo.v)
    assert leve.shapley == pytest.approx(completo.shapley_I + completo.shapley_D)
    assert leve.shapley.sum() == pytest.approx(leve.v)
