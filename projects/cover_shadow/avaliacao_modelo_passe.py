"""Fase 8f · tabelas do notebook 10: números da calibração, da v3.1 e da v4 completo, nas três moedas (D40). Sem veredictos.

Versões comparadas (nomes das colunas):
- `v3` (avaliacao_v3/base) e `v4` (a v4 com fundo da D39, pasta `v4/`);
- `v3_1_<moeda>` e `v4c_<moeda>` (v4 completo), com `<moeda>` em `xt`, `linhas`, `risco`.
Cada função recebe tabelas já calculadas, por isso testa-se sem dados reais.
"""

from __future__ import annotations

import avaliacao as av
import avaliacao_v4 as av4
import calcular_modelo_passe as cmp
import calcular_v4 as c4
import numpy as np
import pandas as pd
from metrica import agregacao
from metrica.passe import MOEDAS
from metrica.pipeline import carregar_tabela, preparar_defesas

from footy.ingest import skillcorner as sc

JOGOS = tuple(sc.JOGOS_COM_POSE)
VERSOES = ("v3", "v4", *(f"v3_1_{m}" for m in MOEDAS), *(f"v4c_{m}" for m in MOEDAS))
LANCES = {"C06": (1925299, 12660), "C03": (1996435, 39646), "D01": (1996435, 23728), "D02": (1996435, 25338),
          "D03": (1996435, 46832), "W04": (1996435, 45252)}  # D01–D03 e W04 da 4.ª ronda (indice_dos_lances.csv)


def carregar_versoes(jogo: int) -> dict:
    """{versão: {momentos, atribuicoes, linhas}} de um jogo. Na v3 os momentos são a `por_lance` (E0 = disponível, v = negada)."""
    linhas3, atrib3 = av.ler_variante(jogo, "base", av.PASTA_AVALIACAO_V3)
    lances3 = agregacao.por_lance(linhas3)
    momentos3 = lances3.rename(columns={"disponivel": "E0", "negada": "v"})
    momentos3["EC"] = momentos3["E0"] - momentos3["v"]
    out = {"v3": {"momentos": momentos3, "atribuicoes": atrib3, "linhas": linhas3}, "v4": c4.ler_v4(jogo)}
    for m in MOEDAS:
        out[f"v3_1_{m}"] = cmp.ler_v3_1(jogo, m)
        out[f"v4c_{m}"] = cmp.ler_v4_completo(jogo, m)
    return out


def totais(momentos: pd.DataFrame) -> dict:
    """Soma de E(∅), E(C), I, D e v nos momentos com alguma linha, % negada e % de momentos com D < 0 e com v < 0 (v4)."""
    coluna = "n_avaliadas" if "n_avaliadas" in momentos else "n_linhas"
    m = momentos[momentos[coluna] > 0]
    r = {"momentos": len(m), "E0": m["E0"].sum(), "v": m["v"].sum(), "pct_negada": 100 * m["v"].sum() / m["E0"].sum(),
         "pct_v_negativa": 100 * (m["v"] < -1e-12).mean()}
    if "D" in m:
        r.update({"I": m["I"].sum(), "D": m["D"].sum(), "pct_D_negativa": 100 * (m["D"] < -1e-12).mean()})
    return r


def tabela_de_totais(versoes_por_jogo: dict) -> pd.DataFrame:
    """Uma linha por (jogo, versão)."""
    linhas = [{"jogo": jogo, "versao": nome, **totais(t["momentos"])} for jogo, vs in versoes_por_jogo.items() for nome, t in vs.items()]
    return pd.DataFrame(linhas)


def por_zona_ou_fase(versoes_por_jogo: dict, contextos: dict, por: str) -> pd.DataFrame:
    """% negada (soma dos dois jogos) por terço ou fase defensiva, uma coluna por versão (`por`: "terco" ou "fase")."""
    partes = []
    for jogo, vs in versoes_por_jogo.items():
        for nome, t in vs.items():
            lances = t["momentos"].rename(columns={"E0": "disponivel", "v": "negada"})
            if "n_avaliadas" not in lances:
                lances = lances.assign(n_avaliadas=lances["n_linhas"])
            g = agregacao.por_zona_ou_fase(lances[["frame", "n_avaliadas", "disponivel", "negada"]], contextos[jogo], por)
            partes.append(g.assign(versao=nome, jogo=jogo))
    t = pd.concat(partes).groupby([por, "versao"], dropna=False)[["disponivel", "negada"]].sum().reset_index()
    t["pct_negada"] = 100 * t["negada"] / t["disponivel"]
    return t.pivot(index=por, columns="versao", values="pct_negada").reindex(columns=list(VERSOES))


def por_jogador(jogo: int, versoes: dict, presenca: pd.DataFrame, jogadores: pd.DataFrame) -> dict:
    """{versão: ameaça negada por jogador} (D17)."""
    saida = {"v3": av.negada_por_jogador(versoes["v3"]["linhas"], versoes["v3"]["atribuicoes"], presenca, jogadores)}
    for nome, t in versoes.items():
        if nome != "v3":
            saida[nome] = av4.negada_por_jogador(t["momentos"], t["atribuicoes"], presenca, jogadores, "shapley")
    return {n: t.assign(jogo=jogo) for n, t in saida.items()}


def matriz_de_rho(por_versao: dict, nomes: tuple = VERSOES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(ρ médio, ρ mínimo) de Spearman por jogador dentro da posição entre cada par de versões (`av.spearman_por_posicao`)."""
    media = pd.DataFrame(np.nan, index=nomes, columns=nomes)
    minimo = media.copy()
    for a in nomes:
        for b in nomes:
            r = av.spearman_por_posicao(por_versao[a], por_versao[b])
            media.loc[a, b], minimo.loc[a, b] = r["rho_media"], r["rho_min"]
    return media, minimo


def lance(jogo: int, frame: int, versoes: dict, nomes_jogadores: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(totais, Shapley por defesa) de um frame em todas as versões. Totais: E(∅), E(C), I, D, v (I e D só na v4)."""
    cols = {}
    for nome, t in versoes.items():
        m = t["momentos"][t["momentos"]["frame"] == frame]
        if m.empty:
            cols[nome] = {}
            continue
        r = m.iloc[0]
        cols[nome] = {k: float(r[k]) for k in ("E0", "EC", "I", "D", "v") if k in r and pd.notna(r[k])}
    tot = pd.DataFrame(cols).reindex(["E0", "EC", "I", "D", "v"])
    partes = {}
    for nome, t in versoes.items():
        a = t["atribuicoes"]
        a = a[a["frame"] == frame]
        partes[nome] = a.groupby("defesa")["shapley"].sum()
    defesas = pd.DataFrame(partes).fillna(0.0)
    defesas.index = defesas.index.map(lambda j: nomes_jogadores.get(j, j))
    return tot, defesas.reindex(columns=list(versoes))


def contexto_do_jogo(jogo: int) -> pd.DataFrame:
    return agregacao.contexto_dos_frames(carregar_tabela(jogo), sc.ler_fases(jogo))


def presenca_do_jogo(jogo: int, tabela: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(presença, jogadores, nomes) do jogo: os defesas por frame, o plantel e {id: nome}."""
    match = sc.ler_match(jogo)
    tabela = carregar_tabela(jogo) if tabela is None else tabela
    defesas = preparar_defesas(tabela, match, c4.PADRAO)
    jogadores = c4.jogadores_do_jogo(match)
    return defesas[["frame", "posse_id", "jogador"]], jogadores, jogadores.set_index("jogador")["nome"].to_dict()


def distribuicao_de_linhas(linhas: pd.DataFrame) -> pd.DataFrame:
    """Distribuição do nº de adversários ultrapassados (`ganho` da moeda `linhas`): nº de linhas e % por valor."""
    t = linhas["ganho"].astype(int).value_counts().sort_index()
    return pd.DataFrame({"linhas": t, "pct": 100 * t / t.sum()})


# Em linguagem corrente: este módulo junta as tabelas de todas as versões da métrica (a v3, a v4 anterior e as versões com o
# modelo completo, uma por moeda) e responde às mesmas perguntas em cada uma: quanto perigo se nega, em que fase e em que terço,
# o que dá em quatro lances concretos, e se os jogadores ficam pela mesma ordem dentro de cada posição.
