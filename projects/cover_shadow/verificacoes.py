"""Fase 5 · verificações feitas a seguir à 2.ª revisão do treinador (guarda-redes, cruzamentos, marcações).

Cada função recebe tabelas já calculadas e devolve números, para o notebook 05 os mostrar sem os
interpretar. Nada aqui altera a métrica.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from metrica.cruzamento import AREA_MEIA_LARGURA, TERCO_OFENSIVO, em_zona_de_cruzamento
from metrica.marcacao import do_lado_da_baliza
from metrica.parametros import Parametros

CONDICOES = ("perto_o_bastante", "lado_da_baliza", "tempo_suficiente", "sem_pressao")
ROTULOS_CONDICAO = {"perto_o_bastante": "distância ao atacante", "lado_da_baliza": "lado da baliza",
                    "tempo_suficiente": "tempo seguido como mais próximo", "sem_pressao": "pressão ao portador (D28)",
                    "": "nenhuma (marca)"}


def guarda_redes_no_frame(tabela: pd.DataFrame, defesas: pd.DataFrame, frame: int) -> dict:
    """Quem é o guarda-redes de quem defende neste frame e se ficou fora da tabela de defesas (D25)."""
    jogadores = tabela[tabela["frame"] == frame]
    guarda_redes = jogadores[jogadores["defende"] & (jogadores["posicao"] == "GK")]
    nos_defesas = defesas[(defesas["frame"] == frame) & (defesas["posicao"] == "GK")]
    return {"guarda_redes": guarda_redes["jogador"].tolist(), "x": guarda_redes["x"].tolist(),
            "y": guarda_redes["y"].tolist(), "na_tabela_de_defesas": len(nos_defesas)}


def defesas_com_cone_junto_a_baliza(defesas: pd.DataFrame, frame: int, baliza_x: float, n: int = 3) -> pd.DataFrame:
    """Os `n` defesas (sem guarda-redes) mais perto da própria baliza neste frame, com cone (não marcam
    e têm pose): são eles que desenham as sombras junto à baliza."""
    d = defesas[defesas["frame"] == frame].copy()
    d["dist_baliza"] = np.hypot(d["x"] - baliza_x, d["y"])
    d["tem_cone"] = d["orientacao"].notna() & ~d["a_marcar"]
    return d.sort_values("dist_baliza").head(n)[["jogador", "posicao", "x", "y", "dist_baliza", "tem_cone"]]


def etiquetas_de_zona(tabela: pd.DataFrame) -> pd.DataFrame:
    """Frames em posse por (terço, corredor) da posse, e se a zona é a de cruzamento do D27."""
    t = tabela.groupby("frame")[["terco", "corredor"]].first().value_counts().rename("frames").reset_index()
    t["zona_de_cruzamento"] = [em_zona_de_cruzamento(c, e) for c, e in zip(t["corredor"], t["terco"])]
    return t.sort_values("frames", ascending=False, ignore_index=True)


def onde_esta_o_portador(tabela: pd.DataFrame, frames) -> pd.DataFrame:
    """Posição do portador e etiquetas da posse nos `frames` dados: `terco`, `corredor`, `y` e `fora_da_area`
    (|y| acima da meia-largura da grande área)."""
    t = tabela[tabela["frame"].isin(frames) & (tabela["jogador"] == tabela["portador"])]
    t = t.drop_duplicates("frame")[["frame", "terco", "corredor", "x", "y"]].copy()
    t["fora_da_area"] = t["y"].abs() > AREA_MEIA_LARGURA
    return t


def resumo_dos_mais_negados(lances: pd.DataFrame, tabela: pd.DataFrame, n: int = 100) -> dict:
    """Dos `n` frames com mais ameaça negada: quantos têm o portador no terço ofensivo, quantos fora da
    largura da grande área nesse terço, quantos na zona de cruzamento do D27. `lances`: `agregacao.por_lance`."""
    top = lances.nlargest(n, "negada")
    p = onde_esta_o_portador(tabela, top["frame"])
    ofensivo = p["terco"] == TERCO_OFENSIVO
    return {"frames": len(p), "terco_ofensivo": int(ofensivo.sum()),
            "terco_ofensivo_fora_da_area": int((ofensivo & p["fora_da_area"]).sum()),
            "zona_de_cruzamento": int(sum(em_zona_de_cruzamento(c, e) for c, e in zip(p["corredor"], p["terco"]))),
            "por_corredor": p["corredor"].value_counts().to_dict()}


def condicoes_da_marcacao(c: pd.DataFrame, p: Parametros) -> pd.DataFrame:
    """Para cada defesa e frame (características de `caracteristicas_de_marcacao`), quais das condições
    para começar a marcar se cumprem com os parâmetros `p`: distância (D14, D33), lado da baliza (D33),
    tempo seguido (D14) e ausência de pressão (D28)."""
    return pd.DataFrame({
        "perto_o_bastante": c["dist_atacante"] <= p.distancia_marcacao,
        "lado_da_baliza": do_lado_da_baliza(c, p),
        "tempo_suficiente": c["persistencia_s"] >= p.tempo_marcacao,
        "sem_pressao": c["dist_portador"] > p.distancia_pressao,
    }, index=c.index)


def regra_que_impediu(condicoes: pd.DataFrame) -> pd.Series:
    """Primeira condição (por ordem de `CONDICOES`) que falha em cada linha; "" se todas se cumprem."""
    falha = ~condicoes[list(CONDICOES)]
    return falha.idxmax(axis=1).where(falha.any(axis=1), "")


def caso_de_marcacao(tabela: pd.DataFrame, c: pd.DataFrame, frame: int, defesa: int, atacante: int,
                     p: Parametros) -> dict:
    """Factos de um defesa que devia marcar um atacante num frame: distância ao atacante indicado e ao mais
    próximo (sem o portador), se o indicado é o portador, vantagem para a baliza, tempo seguido e a regra que
    impediu a marcação com `p`. `c`: características (com `sem_portador=True`) do jogo."""
    f = tabela[tabela["frame"] == frame].set_index("jogador")
    linha = c[(c["frame"] == frame) & (c["jogador"] == defesa)].iloc[0]
    d, a = f.loc[defesa], f.loc[atacante]
    distancia = float(np.hypot(d["x"] - a["x"], d["y"] - a["y"]))
    impediu = regra_que_impediu(condicoes_da_marcacao(linha.to_frame().T.infer_objects(), p)).iloc[0]
    return {"atacante_indicado_e_o_portador": bool(f.loc[atacante, "portador"] == atacante),
            "dist_atacante_indicado": distancia, "mais_proximo_sem_portador": int(linha["atacante"]),
            "dist_mais_proximo": float(linha["dist_atacante"]), "vantagem_baliza": float(linha["vantagem_baliza"]),
            "persistencia_s": float(linha["persistencia_s"]), "dist_portador": float(linha["dist_portador"]),
            "regra_que_impediu": ROTULOS_CONDICAO[impediu]}


def sem_acentos(texto: str) -> str:
    return unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode().lower()


def jogador_por_sobrenome(match: dict, sobrenome: str, equipa: int | None = None, outra_equipa_que: int | None = None) -> int:
    """Id do jogador cujo nome curto contém `sobrenome` (sem acentos nem maiúsculas). `equipa` fixa a equipa;
    `outra_equipa_que` exclui essa equipa (para distinguir homónimos, como dois Warland em equipas contrárias)."""
    achados = [p["id"] for p in match["players"] if sem_acentos(sobrenome) in sem_acentos(p["short_name"])
               and (equipa is None or p["team_id"] == equipa) and p["team_id"] != outra_equipa_que]
    if len(achados) != 1:
        raise ValueError(f"{sobrenome!r}: {len(achados)} jogadores encontrados")
    return achados[0]


def marcacao_na_janela(defesas: pd.DataFrame, defesa: int, frame: int, meia_janela: int = 30) -> dict:
    """Como se comporta um defesa nos frames à volta de `frame` (por omissão ±30 frames de tracking = ±3 s):
    quantos frames alinhados, em quantos está a marcar e entre que valores variam a distância ao atacante mais
    próximo e a vantagem para a baliza. `defesas`: saída de `pipeline.preparar_defesas`."""
    j = defesas[(defesas["jogador"] == defesa) & defesas["frame"].between(frame - meia_janela, frame + meia_janela)]
    return {"frames": len(j), "a_marcar": int(j["a_marcar"].sum()),
            "dist_min": float(j["dist_atacante"].min()), "dist_max": float(j["dist_atacante"].max()),
            "vantagem_min": float(j["vantagem_baliza"].min()), "vantagem_max": float(j["vantagem_baliza"].max())}


# Em linguagem corrente: depois da segunda revisão o treinador levantou três dúvidas: havia uma sombra
# no guarda-redes, os lances com mais ameaça negada eram quase todos cruzamentos e quatro marcações
# não tinham sido reconhecidas. Este módulo responde com números: quem é o guarda-redes e se ficou
# mesmo fora das defesas, onde está o portador nos frames com mais ameaça negada, e, para cada
# marcação em falta, qual das condições da regra (distância, lado da baliza, tempo, pressão) a impediu.
