"""Contagem das linhas excluídas por motivo (D13, D26, D27) na agregação por lance e no resumo do jogo."""

import pandas as pd
import pytest
from metrica.agregacao import (
    com_motivo,
    contexto_dos_frames,
    excluidas_por_motivo,
    fase_defensiva_por_frame,
    por_lance,
    por_zona_ou_fase,
    resumo_do_jogo,
)


def _linhas():
    return pd.DataFrame({
        "frame": [1, 1, 1, 1, 2, 2], "posse_id": "a", "recetor": [1, 2, 3, 4, 1, 2],
        "excluida": [False, True, True, True, False, False],
        "motivo": ["", "fora_de_jogo", "cruzamento", "sem_pose", "", ""],
        "valor_xt": [0.1, 0.2, 0.3, 0.4, 0.1, 0.1], "ameaca_negada": [0.05, None, None, None, 0.0, 0.1],
    })


def test_excluidas_por_motivo_conta_todos_os_motivos():
    assert excluidas_por_motivo(_linhas()) == {"sem_pose": 1, "fora_de_jogo": 1, "cruzamento": 1, "mais_de_35m": 0}
    assert excluidas_por_motivo(_linhas().iloc[4:]) == {"sem_pose": 0, "fora_de_jogo": 0, "cruzamento": 0, "mais_de_35m": 0}


def test_por_lance_separa_as_excluidas_por_motivo():
    lance = por_lance(_linhas()).set_index("frame")
    assert lance.loc[1, ["n_sem_pose", "n_fora_de_jogo", "n_cruzamento"]].tolist() == [1, 1, 1]
    assert lance.loc[1, "n_avaliadas"] == 1 and lance.loc[1, "disponivel"] == pytest.approx(0.1)
    assert lance.loc[2, ["n_sem_pose", "n_fora_de_jogo", "n_cruzamento"]].tolist() == [0, 0, 0]


def test_resumo_traz_os_motivos():
    r = resumo_do_jogo(_linhas())
    assert (r["excl_sem_pose"], r["excl_fora_de_jogo"], r["excl_cruzamento"]) == (1, 1, 1)
    assert r["excluidas"] == 3 and r["avaliadas"] == 3


def test_resultados_antigos_sem_motivo_contam_como_sem_pose():
    antigas = _linhas().drop(columns="motivo")
    assert com_motivo(antigas)["motivo"].tolist() == ["", "sem_pose", "sem_pose", "sem_pose", "", ""]


def _fases():
    return pd.DataFrame({"frame_start": [0, 10], "frame_end": [8, 20],
                         "team_out_of_possession_phase_type": ["high_block", "low_block"],
                         "team_in_possession_phase_type": ["build_up", "create"]})


def test_fase_defensiva_por_frame_respeita_os_limites_da_fase():
    fase = fase_defensiva_por_frame(_fases(), [0, 8, 9, 15, 21])
    assert fase.loc[0] == "high_block" and fase.loc[8] == "high_block"
    assert pd.isna(fase.loc[9])  # entre duas fases
    assert fase.loc[15] == "low_block"
    assert pd.isna(fase.loc[21])  # depois da última


def test_por_zona_ou_fase_soma_por_grupo_e_ignora_lances_sem_linhas_avaliadas():
    lances = pd.DataFrame({"frame": [1, 2, 3], "n_avaliadas": [2, 1, 0], "disponivel": [0.4, 0.6, 0.0],
                           "negada": [0.1, 0.3, 0.0]})
    tabela = pd.DataFrame({"frame": [1, 1, 2, 3], "terco": ["attacking_third"] * 2 + ["middle_third"] * 2,
                           "corredor": ["center", "center", "wide_left", "wide_left"]})
    contexto = contexto_dos_frames(tabela, _fases())
    por_terco = por_zona_ou_fase(lances, contexto, "terco").set_index("terco")
    assert por_terco.loc["attacking_third", "lances"] == 1 and por_terco.loc["middle_third", "lances"] == 1
    assert por_terco.loc["attacking_third", "pct_negada"] == pytest.approx(25.0)
    por_fase = por_zona_ou_fase(lances, contexto, ["fase"]).set_index("fase")
    assert por_fase.loc["high_block", "negada"] == pytest.approx(0.4)
