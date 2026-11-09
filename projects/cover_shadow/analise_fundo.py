"""Números do fundo (D39): v4 sem fundo (D38) contra v4 com fundo, só tabelas, sem leituras.

Cada função recebe tabelas já calculadas (`calcular_v4.ler_v4`, `avaliacao_v4.ler`) e devolve um DataFrame, por isso testa-se
sem dados reais. Usam-se nos cadernos 08 (distribuição de `p_fundo`, totais, lances, ρ por jogador) e 09 (por fase e terço, C4/C5).
"""

from __future__ import annotations

import avaliacao as av
import avaliacao_v4 as a4
import pandas as pd
from metrica import agregacao

EPS = 1e-9
LIMITES_P_FUNDO = (0.9, 0.5)


# ---------------------------------------------------------------- 1. distribuição de p_fundo

def distribuicao_p_fundo(linhas: pd.DataFrame) -> pd.DataFrame:
    """`p_fundo` das linhas (todas, e por `recetor_marcado` = 0 / 1): nº de linhas, % abaixo de 0,9 e de 0,5, média e quantis.
    `linhas`: tabela de linhas da v4 com `p_fundo` e `recetor_marcado`."""
    grupos = {"todas": linhas, "recetor não marcado": linhas[linhas["recetor_marcado"] == 0],
              "recetor marcado": linhas[linhas["recetor_marcado"] == 1]}
    saida = {}
    for nome, g in grupos.items():
        p = g["p_fundo"]
        saida[nome] = {"linhas": len(g), **{f"% com p_fundo < {limite:g}": 100 * (p < limite).mean() for limite in LIMITES_P_FUNDO},
                       "média": p.mean(), "p10": p.quantile(0.1), "mediana": p.median(), "p90": p.quantile(0.9)}
    return pd.DataFrame(saida).T


# ---------------------------------------------------------------- 2. totais por jogo

def totais_e_sinais(momentos: pd.DataFrame) -> dict:
    """E(∅), I, D e v (somas, xT esperado) e, nos momentos com algum defesa em C, a % com D < 0 e com v < 0."""
    com_c = momentos[momentos["n_defesas"] > 0]
    return {"momentos": len(momentos), "momentos com defesas em C": len(com_c),
            "E(∅)": momentos["E0"].sum(), "I": momentos["I"].sum(), "D": momentos["D"].sum(), "v": momentos["v"].sum(),
            "v / E(∅) (%)": 100 * momentos["v"].sum() / momentos["E0"].sum(),
            "% com D < 0": 100 * (com_c["D"] < -EPS).mean(), "% com v < 0": 100 * (com_c["v"] < -EPS).mean()}


def antes_e_depois(sem: dict, com: dict, sem_guarda_redes: dict | None = None) -> pd.DataFrame:
    """Uma coluna por (jogo, versão). `sem`, `com`, `sem_guarda_redes`: {jogo: tabela de momentos}."""
    versoes = {"sem fundo": sem, "com fundo": com, **({"com fundo, sem o guarda-redes": sem_guarda_redes} if sem_guarda_redes else {})}
    return pd.DataFrame({(jogo, versao): totais_e_sinais(m[jogo]) for versao, m in versoes.items() for jogo in m}
                        ).rename_axis(["jogo", "versão"], axis=1)


# ---------------------------------------------------------------- 3. por fase e por terço

def lances_da_v3(linhas: pd.DataFrame) -> pd.DataFrame:
    """Lances (frames) da v3: frame, n_avaliadas, disponivel, negada."""
    return agregacao.por_lance(linhas)[["frame", "n_avaliadas", "disponivel", "negada"]]


def lances_da_v4(momentos: pd.DataFrame) -> pd.DataFrame:
    """Lances (momentos) da v4 com as mesmas colunas: o perigo disponível é E(∅) e o negado é v."""
    return momentos.rename(columns={"n_linhas": "n_avaliadas", "E0": "disponivel", "v": "negada"})[
        ["frame", "n_avaliadas", "disponivel", "negada"]]


def pct_por_contexto(lances: dict, contextos: dict, por: str) -> pd.DataFrame:
    """% negada (100 · Σ negada / Σ disponível) e nº de lances com alguma linha avaliada, por grupo de `por` ("fase" ou "terco"),
    para cada jogo e para os dois juntos ("ambos"). `lances`: {versão: {jogo: lances}}; `contextos`: {jogo: frame, terco, fase}.
    Colunas: (jogo, versão, medida). Os grupos vêm ordenados pelos lances de "ambos" na última versão."""
    por_jogo = {}
    for versao, d in lances.items():
        for jogo, t in d.items():
            c = contextos[jogo]
            por_jogo[(versao, jogo)] = t[t["n_avaliadas"] > 0].merge(c[["frame", por]], on="frame", how="left").assign(jogo=jogo)
    colunas = {}
    for versao, d in lances.items():
        jogos = list(d)
        for jogo in [*jogos, "ambos"]:
            t = pd.concat([por_jogo[(versao, j)] for j in jogos]) if jogo == "ambos" else por_jogo[(versao, jogo)]
            g = t.groupby(por, dropna=False).agg(lances=("frame", "size"), disponivel=("disponivel", "sum"), negada=("negada", "sum"))
            g["pct_negada"] = (100 * g["negada"] / g["disponivel"]).where(g["disponivel"] > 0)
            colunas[(jogo, versao, "% negada")] = g["pct_negada"]
            colunas[(jogo, versao, "lances")] = g["lances"]
    saida = pd.DataFrame(colunas)
    saida.columns = pd.MultiIndex.from_tuples(saida.columns, names=["jogo", "versão", "medida"])
    ultima = list(lances)[-1]
    return saida.sort_values(("ambos", ultima, "lances"), ascending=False).sort_index(axis=1, level=[0, 1, 2], sort_remaining=False)


# ---------------------------------------------------------------- 4. lances

def totais_do_lance(momentos: pd.DataFrame, frame: int) -> dict:
    m = momentos.set_index("frame").loc[frame]
    return {"E(∅)": m["E0"], "E(C)": m["EC"], "I": m["I"], "D": m["D"], "v": m["v"]}


def lance_antes_e_depois(sem: dict, com: dict, frame: int, nomes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(totais, Shapley por defesa) de um frame na v4 sem fundo e com fundo. `sem`, `com`: dicts com `momentos` e `atribuicoes`;
    `nomes`: {jogador: nome}. Por defesa: I, D e total de cada versão (0 para quem não está em C numa delas)."""
    totais = pd.DataFrame({"sem fundo": totais_do_lance(sem["momentos"], frame), "com fundo": totais_do_lance(com["momentos"], frame)})
    partes = []
    for rotulo, d in (("sem fundo", sem), ("com fundo", com)):
        a = d["atribuicoes"][d["atribuicoes"]["frame"] == frame].set_index("defesa")
        partes.append(a[["shapley_I", "shapley_D", "shapley"]].rename(
            columns={"shapley_I": f"I {rotulo}", "shapley_D": f"D {rotulo}", "shapley": f"total {rotulo}"}))
    por_defesa = pd.concat(partes, axis=1).fillna(0.0)
    por_defesa.index = por_defesa.index.map(nomes)
    return totais, por_defesa.sort_values("total com fundo", ascending=False)


# ---------------------------------------------------------------- 5. ρ por jogador

def jogadores_de_uma_versao(momentos: dict, atribuicoes: dict, presenca: dict, jogadores: dict, coluna: str = "shapley_simetrica") -> pd.DataFrame:
    """Ameaça negada por jogador (total) dos dois jogos juntos, com a coluna `jogo`. Cada argumento é {jogo: tabela}."""
    return pd.concat([a4.negada_por_jogador(momentos[j], atribuicoes[j], presenca[j], jogadores[j], coluna).assign(jogo=j)
                      for j in momentos], ignore_index=True)


def rho_entre_versoes(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """ρ de Spearman por jogador dentro da posição entre duas tabelas por jogador (média ponderada, mínimo e por posição)."""
    r = av.spearman_por_posicao(a, b)
    return {"rho_media": r["rho_media"], "rho_min": r["rho_min"], "posicoes": r["posicoes"]}


def jogadores_v3(linhas_e_atribuicoes: dict, presenca: dict, jogadores: dict) -> pd.DataFrame:
    """Ameaça negada por jogador na v3, os dois jogos juntos. `linhas_e_atribuicoes`: {jogo: (linhas, atribuicoes)}."""
    return pd.concat([av.negada_por_jogador(*linhas_e_atribuicoes[j], presenca[j], jogadores[j]).assign(jogo=j)
                      for j in linhas_e_atribuicoes], ignore_index=True)


def diferenca_de_totais(a: pd.DataFrame, b: pd.DataFrame, colunas=("E(∅)", "I", "D", "v")) -> pd.DataFrame:
    """b − a (e b / a − 1, em %) para as `colunas` de duas tabelas de totais (colunas = versões), por linha de `colunas`."""
    dif = (b.loc[list(colunas)] - a.loc[list(colunas)]).astype(float)
    return pd.concat({"diferença": dif, "variação (%)": 100 * (b.loc[list(colunas)].astype(float) / a.loc[list(colunas)].astype(float) - 1)}, axis=1)


# Em linguagem corrente: este módulo só faz contas de comparação entre a v4 de antes (o passe sem sombra chega sempre) e a de
# agora (chega com a probabilidade que o fundo deixa). Mostra quantas linhas ficam mais difíceis, quanto mudam os totais,
# como mudam os lances que já estudámos e se a ordem dos jogadores muda. Não tira conclusões: devolve tabelas.
