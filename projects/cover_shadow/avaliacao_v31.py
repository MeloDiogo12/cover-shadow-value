"""Fase 9b · segunda leitura (moeda «linhas quebradas»), momento da decisão e números da v3.1 para o artigo (D43).

Funções puras sobre as tabelas já calculadas (`avaliacao_v3_1/`, tabela da fase 3, eventos), por isso testam-se sem dados reais.
A v3.1 é a v3 mais a exclusão das linhas com mais de 35 m (D37); o valor continua a ser um índice de fecho da linha, não uma
probabilidade calibrada (D43). Duas moedas, com o mesmo Shapley e a mesma agregação:
- `xt`: valor da linha = ganho de xT (a moeda principal);
- `linhas`: valor da linha = nº de adversários de campo (sem GR) ultrapassados pelo passe (a definição da 8f).
Para a v3.1, `negada = Σ valor_l · (1 − p_l(C))` (com `p_l(∅) = 1`), o Shapley é o da v3 escalado pelo valor da linha.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from escolha import PASTA_SAIDA
from metrica.chegada import valores_de_shapley
from metrica.linhas import recetores_com_valor
from metrica.parametros import PADRAO
from metrica.passe import adversarios_ultrapassados
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.xt import sentido_de_ataque

from footy.ingest import skillcorner as sc

MOEDAS = ("xt", "linhas")
JANELAS_S = (1.0, 2.0)  # segundos antes do fim da posse individual
FRAMES_POR_SEGUNDO = 10  # tracking a 10 fps; a tabela alinhada tem um frame de 0,2 s em 0,2 s
PASTA_MOEDAS = PASTA_SAIDA / "v3_1_moedas"


# ---------------------------------------------------------------- moeda «linhas quebradas»

def ultrapassados_do_jogo(jogo: int, progresso: bool = True) -> pd.DataFrame:
    """`frame`, `recetor`, `ultrapassados`: adversários de campo (sem GR) entre o portador e o recetor, em cada linha de cada frame
    da tabela da fase 3 (a mesma definição da 8f: `metrica.passe.adversarios_ultrapassados`). Só geometria, sem física."""
    match, tabela = sc.ler_match(jogo), carregar_tabela(jogo)
    defesas = preparar_defesas(tabela, match, PADRAO)
    comprimento, largura = sc.dimensoes_campo(match)
    defesas_por_frame = dict(tuple(defesas.groupby("frame")))
    partes, inicio = [], time.time()
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        origem, recetores = recetores_com_valor(jogadores, sentido, comprimento, largura, PADRAO)
        n = adversarios_ultrapassados(origem, recetores[["x", "y"]].to_numpy(float), defesas_por_frame[frame]["x"].to_numpy(), sentido)
        partes.append(pd.DataFrame({"frame": frame, "recetor": recetores["recetor"].to_numpy(), "ultrapassados": n}))
    if progresso:
        print(f"  ultrapassados {jogo}: {time.time() - inicio:.0f} s", flush=True)
    return pd.concat(partes, ignore_index=True)


def ultrapassados_guardados(jogo: int) -> pd.DataFrame:
    """Lê (ou calcula e guarda) a tabela de `ultrapassados_do_jogo`."""
    ficheiro = PASTA_MOEDAS / f"{jogo}_ultrapassados.parquet"
    if not ficheiro.exists():
        PASTA_MOEDAS.mkdir(parents=True, exist_ok=True)
        ultrapassados_do_jogo(jogo).to_parquet(ficheiro, index=False)
    return pd.read_parquet(ficheiro)


def shapley_unitario(atribuicoes: pd.DataFrame) -> pd.Series:
    """Parte de Shapley de cada defesa para uma linha de valor 1 (índice = posição em `atribuicoes`). Como a ameaça negada é linear
    no valor da linha, o Shapley de qualquer moeda é este multiplicado pelo valor."""
    saida = pd.Series(0.0, index=atribuicoes.index)
    for _, g in atribuicoes.groupby(["frame", "recetor"], sort=False):
        saida.loc[g.index] = valores_de_shapley(1.0, g["p_defesa"].to_numpy(float))
    return saida


def moeda_linhas(linhas: pd.DataFrame, atribuicoes: pd.DataFrame, ultrapassados: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As tabelas da v3.1 na moeda «linhas»: `valor_xt` passa a ser o nº de adversários ultrapassados (a coluna mantém o nome para
    reaproveitar a agregação), `ameaca_negada = ultrapassados · (1 − p_chegar)` e o Shapley fica escalado por esse valor."""
    valor = linhas[["frame", "recetor"]].merge(ultrapassados, on=["frame", "recetor"], how="left")["ultrapassados"].to_numpy(float)
    if np.isnan(valor).any():
        raise ValueError("há linhas sem nº de adversários ultrapassados")
    novas = linhas.assign(valor_xt=valor, ameaca_negada=np.where(linhas["excluida"], np.nan, valor * (1 - linhas["p_chegar"])))
    escala = atribuicoes[["frame", "recetor"]].merge(ultrapassados, on=["frame", "recetor"], how="left")["ultrapassados"].to_numpy(float)
    return novas, atribuicoes.assign(shapley=shapley_unitario(atribuicoes).to_numpy() * escala)


# ---------------------------------------------------------------- % negada por fase e por terço

def pct_por_contexto(lances: pd.DataFrame, contexto: pd.DataFrame, por: str) -> pd.DataFrame:
    """Lances (frames com alguma linha avaliada), ameaça disponível, negada e % negada por `por` ("fase" ou "terco"), dois jogos
    juntos. `lances`: `agregacao.por_lance` com `jogo`; `contexto`: `contexto_dos_frames` com `jogo`."""
    t = lances[lances["n_avaliadas"] > 0].merge(contexto, on=["jogo", "frame"])
    g = t.groupby(por, dropna=False).agg(lances=("frame", "size"), disponivel=("disponivel", "sum"), negada=("negada", "sum")).reset_index()
    g["pct_negada"] = (100 * g["negada"] / g["disponivel"]).where(g["disponivel"] > 0)
    return g.sort_values("pct_negada", ascending=False, ignore_index=True)


# ---------------------------------------------------------------- momento da decisão

def fim_das_posses(eventos: pd.DataFrame) -> pd.DataFrame:
    """`posse_id`, `frame_end`, `end_type` de cada posse individual (`player_possession`)."""
    p = eventos[eventos["event_type"] == "player_possession"]
    return pd.DataFrame({"posse_id": p["event_id"].to_numpy(), "frame_end": p["frame_end"].to_numpy(),
                         "end_type": p["end_type"].to_numpy()})


def presenca_na_janela(presenca: pd.DataFrame, fim: pd.DataFrame, segundos: float) -> pd.DataFrame:
    """Só os frames dos `segundos` anteriores ao fim da posse individual (`frame_end` incluído). `presenca`: frame, posse_id, jogador."""
    t = presenca.merge(fim[["posse_id", "frame_end"]], on="posse_id", how="left")
    return t[t["frame"] >= t["frame_end"] - segundos * FRAMES_POR_SEGUNDO][presenca.columns.tolist()]


def posses_por_tipo_de_fim(frames_por_posse: pd.DataFrame, fim: pd.DataFrame, janelas: dict[float, pd.DataFrame]) -> dict:
    """Nº de posses por `end_type`: as alinhadas na tabela (`posse_id` em `frames_por_posse`) e as que têm algum frame em cada janela."""
    base = fim[fim["posse_id"].isin(frames_por_posse["posse_id"])]
    saida = {"posses_alinhadas": base["end_type"].value_counts().to_dict()}
    for s, presenca in janelas.items():
        com = base[base["posse_id"].isin(presenca["posse_id"])]
        saida[f"posses_com_frames_{s:g}s"] = com["end_type"].value_counts().to_dict()
    return saida


def sobem_e_descem(atual: pd.DataFrame, janela: pd.DataFrame, n: int = 10, minimo: int = 4) -> dict:
    """Os `n` jogadores que mais sobem e mais descem de posição (percentil da ameaça negada dentro da posição, entre 0 e 1) da agregação
    atual para a da janela. `atual`, `janela`: tabelas por jogador (`jogo`, `jogador`, `posicao`, `negada`, `nome`). Descritivo."""
    t = atual.merge(janela[["jogo", "jogador", "negada"]], on=["jogo", "jogador"], suffixes=("_atual", "_janela"))
    t = t[t.groupby("posicao")["jogador"].transform("size") >= minimo].copy()
    for c in ("atual", "janela"):
        t[f"percentil_{c}"] = t.groupby("posicao")[f"negada_{c}"].rank(pct=True)
    t["variacao"] = t["percentil_janela"] - t["percentil_atual"]
    colunas = ["jogo", "jogador", "nome", "posicao", "negada_atual", "negada_janela", "percentil_atual", "percentil_janela", "variacao"]
    ordenado = t.sort_values("variacao")
    return {"sobem": ordenado.tail(n)[::-1][colunas].round(4).to_dict("records"),
            "descem": ordenado.head(n)[colunas].round(4).to_dict("records")}


# ---------------------------------------------------------------- por lance e por comprimento

def negada_por_defesa_no_lance(linhas: pd.DataFrame, atribuicoes: pd.DataFrame, frame: int, nomes: dict) -> dict:
    """Perigo, ameaça negada e Shapley por defesa de um frame (`nomes`: jogador → nome)."""
    avaliadas = linhas[(linhas["frame"] == frame) & ~linhas["excluida"]]
    a = atribuicoes[atribuicoes["frame"] == frame].groupby("defesa")["shapley"].sum().sort_values(ascending=False)
    return {"linhas_avaliadas": len(avaliadas), "perigo": float(avaliadas["valor_xt"].sum()),
            "negada": float(avaliadas["ameaca_negada"].sum()),
            "shapley": {nomes.get(int(j), str(j)): float(v) for j, v in a.items()}}


BANDAS_M = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 35.0), (35.0, 45.0), (45.0, 1000.0))


def negada_por_comprimento(linhas: pd.DataFrame, bandas=BANDAS_M) -> pd.DataFrame:
    """Linhas avaliadas, perigo e ameaça negada por banda de comprimento ([inf, sup) em m), com a % do total de cada uma.
    `linhas` tem `comprimento`, `excluida`, `valor_xt`, `ameaca_negada`."""
    a = linhas[~linhas["excluida"]]
    saida = []
    for inf, sup in bandas:
        b = a[(a["comprimento"] >= inf) & (a["comprimento"] < sup)]
        saida.append({"de_m": inf, "ate_m": sup, "linhas": len(b), "disponivel": float(b["valor_xt"].sum()), "negada": float(b["ameaca_negada"].sum())})
    t = pd.DataFrame(saida)
    t["pct_do_total_negada"] = 100 * t["negada"] / t["negada"].sum()
    t["pct_das_linhas"] = 100 * t["linhas"] / t["linhas"].sum()
    return t


# ---------------------------------------------------------------- moedas dentro de cada fase defensiva

BLOCOS = ("high_block", "medium_block", "low_block")


def parte_da_negada_por_fase(lances: pd.DataFrame, contexto: pd.DataFrame) -> pd.DataFrame:
    """Parte (%) da ameaça negada total que cada fase defensiva reúne, numa moeda (dois jogos juntos). `lances`: `agregacao.por_lance` com
    `jogo`; `contexto`: `contexto_dos_frames` com `jogo`. Os lances sem fase contam no total e ficam com `fase` NaN."""
    t = lances[lances["n_avaliadas"] > 0].merge(contexto[["jogo", "frame", "fase"]], on=["jogo", "frame"], how="left")
    g = t.groupby("fase", dropna=False)["negada"].sum().reset_index()
    g["pct_da_negada_total"] = 100 * g["negada"] / g["negada"].sum()
    return g.sort_values("pct_da_negada_total", ascending=False, ignore_index=True)


def presenca_na_fase(presenca: pd.DataFrame, contexto_do_jogo: pd.DataFrame, fase: str) -> pd.DataFrame:
    """Só os frames de uma fase defensiva (`presenca`: frame, posse_id, jogador; `contexto_do_jogo`: frame, fase)."""
    frames = contexto_do_jogo.loc[contexto_do_jogo["fase"] == fase, "frame"]
    return presenca[presenca["frame"].isin(frames)]


def momentos_que_mais_discordam(lances_xt: pd.DataFrame, lances_linhas: pd.DataFrame, contexto: pd.DataFrame, fases: tuple,
                                n: int = 2) -> dict:
    """Momentos das `fases` em que as duas moedas mais discordam: percentil da ameaça negada de cada momento entre os momentos dessas
    fases (dois jogos juntos, empates com a média) numa moeda menos o da outra. `xt_alto`: `n` momentos com xT alto e linhas baixo;
    `linhas_alto`: o contrário. Uma só posse por momento escolhido."""
    chave = ["jogo", "frame"]
    t = lances_xt[chave + ["posse_id", "n_avaliadas", "negada", "disponivel"]].merge(
        lances_linhas[chave + ["negada", "disponivel"]], on=chave, suffixes=("_xt", "_linhas"))
    t = t.merge(contexto[chave + ["fase"]], on=chave)
    t = t[t["fase"].isin(fases) & (t["n_avaliadas"] > 0)].copy()
    for m in ("xt", "linhas"):
        t[f"percentil_{m}"] = t[f"negada_{m}"].rank(pct=True)
    t["dif"] = t["percentil_xt"] - t["percentil_linhas"]
    saida = {}
    for nome, ordem in (("xt_alto", False), ("linhas_alto", True)):
        escolhidos, posses = [], set()
        for _, r in t.sort_values(["dif", "frame"], ascending=[ordem, True]).iterrows():
            if (r["jogo"], r["posse_id"]) not in posses:
                escolhidos.append(r)
                posses.add((r["jogo"], r["posse_id"]))
            if len(escolhidos) == n:
                break
        saida[nome] = pd.DataFrame(escolhidos).reset_index(drop=True)
    return saida


# Em linguagem corrente: este módulo reúne as contas da fase 9b que não mudam a métrica. Duas moedas: o valor de uma linha como
# ganho de xT ou como número de adversários que o passe ultrapassa (a «linha quebrada»). Para cada uma vê-se onde se nega mais
# (por fase defensiva e por terço) e quem nega mais. Verifica-se também se o resultado por jogador muda quando só se contam os
# últimos 1 ou 2 segundos antes de a posse acabar (o momento da decisão), e como se distribui a ameaça negada pelo comprimento
# das linhas.
