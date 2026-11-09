"""Verificações a seguir à 2.ª revisão: a regra que impediu uma marcação, o guarda-redes e os frames mais negados."""

import pandas as pd
import pytest
from metrica.parametros import PADRAO, PADRAO_V2
from verificacoes import (
    condicoes_da_marcacao,
    guarda_redes_no_frame,
    jogador_por_sobrenome,
    marcacao_na_janela,
    regra_que_impediu,
    resumo_dos_mais_negados,
)


def _caracteristicas(dist_atacante, vantagem, persistencia=1.2, dist_portador=20.0):
    return pd.DataFrame({"dist_atacante": [dist_atacante], "vantagem_baliza": [vantagem],
                         "lado_da_baliza": [vantagem > 0], "persistencia_s": [persistencia],
                         "dist_portador": [dist_portador]})


def test_regra_que_impediu_e_a_primeira_condicao_que_falha():
    c = pd.concat([_caracteristicas(3.5, 1.0), _caracteristicas(2.0, -0.4), _caracteristicas(2.0, 1.0, 0.4),
                   _caracteristicas(2.0, 1.0, dist_portador=3.0), _caracteristicas(2.0, 1.0)], ignore_index=True)
    impediu = regra_que_impediu(condicoes_da_marcacao(c, PADRAO_V2))
    assert impediu.tolist() == ["perto_o_bastante", "lado_da_baliza", "tempo_suficiente", "sem_pressao", ""]


def test_a_v3_deixa_passar_o_que_a_v2_impedia():
    c = pd.concat([_caracteristicas(3.5, 1.0), _caracteristicas(2.0, -0.4)], ignore_index=True)
    assert regra_que_impediu(condicoes_da_marcacao(c, PADRAO)).tolist() == ["", ""]


def test_guarda_redes_fora_da_tabela_de_defesas():
    tabela = pd.DataFrame({"frame": [1, 1, 1], "jogador": [1, 2, 3], "defende": [True, True, False],
                           "posicao": ["GK", "CB", "GK"], "x": [-50.0, -30.0, 50.0], "y": [0.0, 1.0, 0.0]})
    defesas = pd.DataFrame({"frame": [1], "jogador": [2], "posicao": ["CB"]})
    r = guarda_redes_no_frame(tabela, defesas, 1)
    assert r["guarda_redes"] == [1] and r["na_tabela_de_defesas"] == 0


def test_resumo_dos_mais_negados_conta_terco_ofensivo_e_fora_da_area():
    lances = pd.DataFrame({"frame": [1, 2, 3], "negada": [0.3, 0.2, 0.1]})
    tabela = pd.DataFrame({
        "frame": [1, 2, 3], "jogador": [9, 9, 9], "portador": [9, 9, 9],
        "terco": ["attacking_third", "attacking_third", "middle_third"],
        "corredor": ["wide_left", "center", "center"], "x": [40.0, 40.0, 0.0], "y": [30.0, 5.0, 0.0]})
    r = resumo_dos_mais_negados(lances, tabela, n=3)
    assert (r["terco_ofensivo"], r["terco_ofensivo_fora_da_area"], r["zona_de_cruzamento"]) == (2, 1, 1)


def test_jogador_por_sobrenome_distingue_homonimos_pela_equipa():
    match = {"players": [{"id": 1, "short_name": "B. Warland", "team_id": 10},
                         {"id": 2, "short_name": "R. Warland", "team_id": 20},
                         {"id": 3, "short_name": "L. Jovanović", "team_id": 10}]}
    assert jogador_por_sobrenome(match, "Warland", outra_equipa_que=10) == 2
    assert jogador_por_sobrenome(match, "jovanovic") == 3
    with pytest.raises(ValueError):
        jogador_por_sobrenome(match, "Warland")


def test_marcacao_na_janela_conta_so_os_frames_da_janela():
    d = pd.DataFrame({"jogador": 7, "frame": [0, 2, 4, 40], "a_marcar": [True, False, True, True],
                      "dist_atacante": [2.0, 3.5, 3.0, 1.0], "vantagem_baliza": [0.5, -0.2, 0.1, 9.0]})
    r = marcacao_na_janela(d, 7, 2, meia_janela=5)
    assert (r["frames"], r["a_marcar"], r["dist_max"], r["vantagem_min"]) == (3, 2, 3.5, -0.2)
