"""Filtros dos candidatos a lance de exemplo (fase 6)."""

import pandas as pd
from candidatos import (
    defesas_com_mais_valor,
    defesas_que_afetam_alguma_linha,
    frames_concentrados,
    marcacao_estavel,
    minuto_do_frame,
    perto_de_um_limiar,
)
from metrica.parametros import PADRAO


def _defesa(frame, dist_atacante=2.0, dist_portador=15.0, vantagem=2.0, persistencia=3.0):
    return {"frame": frame, "dist_atacante": dist_atacante, "dist_portador": dist_portador,
            "vantagem_baliza": vantagem, "persistencia_s": persistencia}


def test_perto_de_um_limiar_por_cada_limite():
    d = pd.DataFrame([_defesa(1), _defesa(2, dist_atacante=3.8), _defesa(3, dist_atacante=5.2),
                      _defesa(4, dist_portador=4.3), _defesa(5, vantagem=-0.9), _defesa(6, persistencia=1.2),
                      _defesa(7, dist_atacante=9.0, vantagem=-1.0, persistencia=1.0)])
    r = perto_de_um_limiar(d, PADRAO)
    assert r.tolist() == [False, True, True, True, True, True, False]


def test_defesas_que_afetam_alguma_linha_ignora_os_que_estao_longe():
    d = pd.DataFrame({"x": [5.0, 5.0], "y": [1.0, 40.0], "a_marcar": [False, False]})
    r = defesas_que_afetam_alguma_linha(d, (0.0, 0.0), [(10.0, 0.0)], PADRAO)
    assert r.tolist() == [True, False]


def test_marcacao_estavel_exige_o_mesmo_conjunto_em_todos_os_frames():
    d = pd.DataFrame({"frame": [1, 1, 2, 2, 3, 3], "jogador": [7, 8] * 3, "a_marcar": [True, False] * 3})
    assert marcacao_estavel(d, [1, 2, 3])
    d.loc[4, "a_marcar"] = False
    assert not marcacao_estavel(d, [1, 2, 3])


def test_defesas_com_mais_valor_soma_por_defesa_e_ordena():
    a = pd.DataFrame({"frame": 1, "recetor": [1, 2, 3], "defesa": [10, 10, 20], "shapley": [0.1, 0.2, 0.25]})
    r = defesas_com_mais_valor(a, 1, n=2)
    assert r["defesa"].tolist() == [10, 20] and abs(r["negada"].iloc[0] - 0.3) < 1e-9


def test_frames_concentrados_limita_defesas_e_cones_por_linha():
    a = pd.DataFrame({"frame": [1, 1, 2, 2, 2, 2], "recetor": [1, 2, 1, 1, 1, 2],
                      "defesa": [10, 11, 10, 11, 12, 13], "shapley": 0.1})
    assert frames_concentrados(a).tolist() == [1]  # o frame 2 tem 3 sombras na mesma linha e 4 defesas


def test_minuto_do_frame_usa_o_inicio_da_fase():
    fases = pd.DataFrame({"frame_start": [0, 1000], "frame_end": [999, 2000], "minute_start": [0, 5],
                          "second_start": [0, 30]})
    assert minuto_do_frame(fases, 1600) == 6  # 5 min 30 s + 60 s
