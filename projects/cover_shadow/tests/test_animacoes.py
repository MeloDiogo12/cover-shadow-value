"""Animação do C06: cronograma, interpolação do tracking, textos e números iguais aos das figuras do artigo."""

import json

import numpy as np
import pytest
from animacoes import (
    DURACOES,
    ORDEM,
    TEXTOS,
    Percurso,
    Vista,
    cronograma,
    estado_em,
    interpolar_vista,
    posicoes_em,
    texto,
    vista_zoom,
)

from footy.paths import OUTPUTS


def percurso_de_dois_frames(det0=(True, True), det1=(True, False)) -> Percurso:
    """Dois frames de tracking (10 fps) com dois jogadores."""
    return Percurso(frames=np.array([100, 101]), jogadores=[7, 8], x=np.array([[0.0, 10.0], [1.0, 10.0]]),
                    y=np.array([[0.0, 0.0], [2.0, 4.0]]), detetado=np.array([det0, det1]),
                    bola=np.array([[0.0, 0.0], [2.0, 2.0]]))


def test_interpolacao_linear_a_meio_e_nas_pontas():
    p = percurso_de_dois_frames()
    pos, bola = posicoes_em(p, 100.5)
    assert pos.set_index("jogador").loc[7, ["x", "y"]].tolist() == pytest.approx([0.5, 1.0])
    assert pos.set_index("jogador").loc[8, ["x", "y"]].tolist() == pytest.approx([10.0, 2.0])
    assert bola == pytest.approx((1.0, 1.0))
    pos0, _ = posicoes_em(p, 100.0)
    assert pos0.set_index("jogador").loc[7, "x"] == 0.0
    pos1, _ = posicoes_em(p, 101.0)
    assert pos1.set_index("jogador").loc[7, "x"] == 1.0


def test_so_e_detetado_se_for_nos_dois_frames_vizinhos():
    p = percurso_de_dois_frames(det0=(True, True), det1=(True, False))
    meio = posicoes_em(p, 100.4)[0].set_index("jogador")["detetado"]
    assert bool(meio[7]) and not bool(meio[8])  # o 8 foi extrapolado no segundo frame
    assert bool(posicoes_em(p, 100.0)[0].set_index("jogador")["detetado"][8])  # no frame exato conta o próprio frame


def test_jogador_sem_posicao_nao_e_desenhado():
    p = percurso_de_dois_frames()
    p.x[1, 1] = np.nan
    assert posicoes_em(p, 100.5)[0]["jogador"].tolist() == [7]


def test_cronograma_soma_as_duracoes_e_encontra_o_passo():
    plano = cronograma()
    assert plano[0][:2] == (ORDEM[0], 0.0)
    assert plano[-1][2] == pytest.approx(sum(DURACOES[n] for n in ORDEM))
    i, nome, tp, p = estado_em(DURACOES["abertura"] + DURACOES["contexto"] + 0.5, plano)
    assert (i, nome) == (2, "pausa") and tp == pytest.approx(0.5) and 0 < p < 1
    assert estado_em(1e6, plano)[1] == "cartao"  # depois do fim fica no último passo


def test_primeiro_passo_em_tempo_real():
    from animacoes import JANELA_JOGO_S, VELOCIDADE_CONTEXTO
    assert VELOCIDADE_CONTEXTO == 1.0
    assert DURACOES["contexto"] == JANELA_JOGO_S


def test_prototipo_dura_entre_40_e_55_segundos():
    assert 40 <= sum(DURACOES[n] for n in ORDEM) <= 55


def test_cartao_de_abertura_e_o_primeiro_passo_e_dura_cerca_de_4_segundos():
    assert ORDEM[0] == "abertura" and 3.5 <= DURACOES["abertura"] <= 4.5
    assert TEXTOS["pt"]["abertura_pergunta"] == "Quanto vale fechar uma linha de passe sem tocar na bola?"
    assert texto("pt", "abertura_defesa", "escura", equipa_defesa="Brisbane Roar") == "A defender: Brisbane Roar (branco)"
    assert texto("pt", "abertura_defesa", "clara", equipa_defesa="Brisbane Roar") == "A defender: Brisbane Roar (preto)"


def test_vista_zoom_segue_a_proporcao_e_a_camara_interpola():
    v = vista_zoom((10.0, 5.0), 12.5)
    assert v.ylim == (-7.5, 17.5)
    assert (v.xlim[1] - v.xlim[0]) / (v.ylim[1] - v.ylim[0]) > 1.5  # mais larga que alta
    a, b = Vista((0, 10), (0, 5)), Vista((10, 20), (5, 10))
    assert interpolar_vista(a, b, 0).xlim == (0, 10) and interpolar_vista(a, b, 1).xlim == (10, 20)
    assert interpolar_vista(a, b, 0.5).xlim == pytest.approx((5, 15))


def test_todos_os_textos_pt_existem_para_cada_passo():
    for nome in ORDEM:
        assert f"legenda_{nome}" in TEXTOS["pt"]


FICHEIRO_NUMEROS = OUTPUTS / "cover_shadow" / "artigo" / "numeros_das_figuras.json"


@pytest.mark.skipif(not FICHEIRO_NUMEROS.exists(), reason="figuras do artigo ainda não geradas")
def test_numeros_do_video_sao_os_das_figuras_do_artigo():
    """Precisa dos dados do jogo (data/interim); sem eles salta."""
    import gerar_candidatos as gc
    from animacoes import numeros_do_lance
    from figuras_lance import Lance

    from footy.viz.theme import load_brand

    try:
        J = gc.carregar(1925299)
    except FileNotFoundError:
        pytest.skip("sem os dados do jogo")
    brand = load_brand()
    artigo = json.loads(FICHEIRO_NUMEROS.read_text(encoding="utf-8"))
    moeda = artigo.get("moeda", "xt")  # as figuras do artigo estão em risco desde a D44
    n = numeros_do_lance(Lance(J, 12660, "C06 · 1.ª parte · minuto 21", brand, moeda=moeda))
    fig8 = artigo["c06_passos"]["8"]
    assert n["negada_total_xt"] == pytest.approx(fig8["negada_total_xt"])
    assert n["perigo_total_xt"] == pytest.approx(fig8["perigo_total"] if "perigo_total" in fig8 else fig8["perigo_total_xt"])
    assert n["shapley"] == pytest.approx(fig8["shapley"])
    assert n["p_com_sombra"] == pytest.approx(artigo["c06_passos"]["5"]["p_com_sombra"])
    bola = numeros_do_lance(Lance(J, 12660, "C06 · 1.ª parte · minuto 21", brand, "bola", moeda))
    figura9 = artigo["c06_corpo_bola"]["Cone virado para a bola"]
    assert bola["negada_total_xt"] == pytest.approx(figura9["total_xt"])
    assert {k: v for k, v in bola["shapley"].items()} == pytest.approx(figura9["por_defesa"])
