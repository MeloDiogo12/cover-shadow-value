"""Fase 9c · moeda «risco» na v3.1 e índice por fase defensiva (D43).

Funções puras sobre as tabelas já calculadas (`avaliacao_v3_1/`, tabela da fase 3, fases da SkillCorner), por isso testam-se sem dados reais.
A moeda `risco` mantém as sombras da v3.1 (parâmetros da literatura, cone pelo tronco, linhas até 35 m) e muda só o valor da linha:
- `g_l` = ganho de xT (o da moeda `xt`, nunca negativo);
- `c_l` = xT do **adversário** no ponto onde a linha seria cortada (a definição da 8f, `metrica.passe.custo_da_perda`: grelha vista da
  outra equipa; ponto = onde o defesa com maior probabilidade de corte a corta com mais probabilidade, contando todos os defesas,
  o guarda-redes incluído).
Valor negado por uma linha: `(p_l(∅) − p_l(C)) · (g_l + c_l)`, com `p_l(∅) = 1` na v3.1 (sem fundo): `(1 − p_chegar) · (g + c)`. Nunca é
negativa. O Shapley é o da v3 escalado por `g + c` e a agregação é a mesma; a «ameaça disponível» da linha passa a ser `g + c`.
"""

from __future__ import annotations

import time

import avaliacao_v31 as a31
import numpy as np
import pandas as pd
from calcular_modelo_passe import arrays_dos_defesas
from calcular_v4 import guarda_redes_por_frame
from metrica.linhas import recetores_com_valor
from metrica.parametros import PADRAO
from metrica.passe import custo_da_perda, probabilidades_das_linhas
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.xt import sentido_de_ataque

from footy.ingest import skillcorner as sc

MOEDAS_3 = ("xt", "linhas", "risco")
AREA_PROFUNDIDADE_M = 16.5  # área grande: 16,5 m a partir da linha de baliza
AREA_LARGURA_M = 40.32  # e 40,32 m de largura (16,5 m de cada lado da baliza de 7,32 m)
MIN_MOMENTOS = 100  # índice por fase: jogadores com menos momentos (20 s) na fase ficam fora das estatísticas da distribuição


# ---------------------------------------------------------------- ponto de corte e custo da perda

def dentro_da_area_propria(x, y, sentido: int, comprimento: float) -> np.ndarray:
    """True onde (x, y) está na área grande da equipa que tem a bola (a que ataca para `sentido`), ou seja, junto à sua própria baliza."""
    distancia_a_propria_baliza = np.asarray(x, float) * sentido + comprimento / 2
    return (distancia_a_propria_baliza <= AREA_PROFUNDIDADE_M) & (np.abs(np.asarray(y, float)) <= AREA_LARGURA_M / 2)


def custos_do_jogo(jogo: int, progresso: bool = True) -> pd.DataFrame:
    """`frame`, `recetor`, `custo_xt`, `corte_x`, `corte_y`, `p_corte`, `corte_na_area_propria` de cada linha de cada frame da tabela da fase 3.
    Parâmetros da literatura (`PADRAO`); todos os defesas, o guarda-redes incluído, entram na escolha do ponto de corte."""
    match, tabela = sc.ler_match(jogo), carregar_tabela(jogo)
    defesas = preparar_defesas(tabela, match, PADRAO)
    comprimento, largura = sc.dimensoes_campo(match)
    defesas_por_frame = dict(tuple(defesas.groupby("frame")))
    guardas = guarda_redes_por_frame(tabela, PADRAO)
    partes, inicio = [], time.time()
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        origem, recetores = recetores_com_valor(jogadores, sentido, comprimento, largura, PADRAO)
        xy = recetores[["x", "y"]].to_numpy(float)
        d = arrays_dos_defesas(defesas_por_frame[frame], guardas.get(frame))
        prob = probabilidades_das_linhas(origem, xy, d, PADRAO)
        partes.append(pd.DataFrame({
            "frame": frame, "recetor": recetores["recetor"].to_numpy(),
            "custo_xt": custo_da_perda(prob.ponto_corte, sentido, comprimento, largura),
            "corte_x": prob.ponto_corte[:, 0], "corte_y": prob.ponto_corte[:, 1], "p_corte": prob.p_corte.max(axis=1),
            "corte_na_area_propria": dentro_da_area_propria(prob.ponto_corte[:, 0], prob.ponto_corte[:, 1], sentido, comprimento)}))
    if progresso:
        print(f"  custos de perda {jogo}: {time.time() - inicio:.0f} s", flush=True)
    return pd.concat(partes, ignore_index=True)


def custos_guardados(jogo: int) -> pd.DataFrame:
    """Lê (ou calcula e guarda) a tabela de `custos_do_jogo`."""
    ficheiro = a31.PASTA_MOEDAS / f"{jogo}_custos_risco.parquet"
    if not ficheiro.exists():
        a31.PASTA_MOEDAS.mkdir(parents=True, exist_ok=True)
        custos_do_jogo(jogo).to_parquet(ficheiro, index=False)
    return pd.read_parquet(ficheiro)


# ---------------------------------------------------------------- a moeda

def moeda_risco(linhas: pd.DataFrame, atribuicoes: pd.DataFrame, custos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As tabelas da v3.1 na moeda «risco»: `valor_xt` passa a ser `g + c` (nome mantido para reaproveitar a agregação),
    `ameaca_negada = (g + c) · (1 − p_chegar)` e o Shapley fica escalado por `g + c`. Acrescenta `ganho`, `custo` e `corte_na_area_propria`."""
    c = linhas[["frame", "recetor"]].merge(custos, on=["frame", "recetor"], how="left")
    if c["custo_xt"].isna().any():
        raise ValueError("há linhas sem custo da perda")
    ganho, custo = linhas["valor_xt"].to_numpy(float), c["custo_xt"].to_numpy(float)
    valor = ganho + custo
    novas = linhas.assign(ganho=ganho, custo=custo, corte_na_area_propria=c["corte_na_area_propria"].to_numpy(bool), valor_xt=valor,
                          ameaca_negada=np.where(linhas["excluida"], np.nan, valor * (1 - linhas["p_chegar"])))
    escala = atribuicoes[["frame", "recetor"]].merge(
        pd.DataFrame({"frame": linhas["frame"], "recetor": linhas["recetor"], "valor": valor}), on=["frame", "recetor"], how="left")["valor"].to_numpy(float)
    return novas, atribuicoes.assign(shapley=a31.shapley_unitario(atribuicoes).to_numpy() * escala)


def moeda_risco_do_lance(linhas: pd.DataFrame, atribuicoes: pd.DataFrame, custos_do_jogo: pd.DataFrame, frame: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`moeda_risco` para um só lance, com as tabelas tal como saem de `revisao.dados_do_frame` (sem coluna `frame`).
    `custos_do_jogo`: a tabela de `custos_guardados`; só se usam as linhas do `frame`. A saída não tem a coluna `frame`."""
    custos = custos_do_jogo[custos_do_jogo["frame"] == frame]
    novas, partes = moeda_risco(linhas.assign(frame=frame), atribuicoes.assign(frame=frame), custos)
    return novas.drop(columns="frame"), partes.drop(columns="frame")


def custo_por_zona_do_corte(linhas_risco: pd.DataFrame) -> dict:
    """Linhas avaliadas com o ponto de corte na área de quem tem a bola contra fora dela: nº, custo médio e máximo, valor médio `g + c`
    e parte (%) da ameaça negada total da moeda (e, no total, o nº de linhas com sombra, ou seja, com negada > 0)."""
    a = linhas_risco[~linhas_risco["excluida"]]
    total = a["ameaca_negada"].sum()
    saida = {}
    for nome, g in (("corte_na_area_propria", a[a["corte_na_area_propria"]]), ("corte_fora_da_area", a[~a["corte_na_area_propria"]])):
        saida[nome] = {"linhas": len(g), "pct_das_linhas": float(100 * len(g) / len(a)), "custo_medio": float(g["custo"].mean()),
                       "custo_maximo": float(g["custo"].max()), "ganho_medio": float(g["ganho"].mean()),
                       "valor_medio_g_mais_c": float(g["valor_xt"].mean()), "linhas_com_sombra": int((g["ameaca_negada"] > 0).sum()),
                       "negada_media_por_linha_com_sombra": float(g.loc[g["ameaca_negada"] > 0, "ameaca_negada"].mean()),
                       "pct_da_negada_total": float(100 * g["ameaca_negada"].sum() / total)}
    return saida


# ---------------------------------------------------------------- índice por fase

def indice_por_fase(atribuicoes: pd.DataFrame, presenca: pd.DataFrame, contexto: pd.DataFrame, jogadores: pd.DataFrame,
                    minimo: int = MIN_MOMENTOS) -> pd.DataFrame:
    """Índice de cada jogador em cada fase defensiva, numa moeda. Um momento = um frame alinhado em que o jogador defende nessa fase.
    - `negada_por_momento` = soma da ameaça negada (Shapley) do jogador nos momentos da fase / nº de momentos (os sem nada contam zero);
    - `media_da_fase` = a mesma conta com todos os jogadores da fase juntos (total negado / total de pares jogador–momento);
    - `indice` = 100 · `negada_por_momento` / `media_da_fase` (100 = média da fase);
    - `percentil` = percentil do índice entre os jogadores da fase com pelo menos `minimo` momentos (0–100; NaN para os restantes).
    `presenca`: frame, jogador (os defesas por frame); `contexto`: frame, fase; `atribuicoes`: frame, defesa, shapley; `jogadores`: jogador,
    nome, posicao. Frames sem fase ficam de fora."""
    negada = atribuicoes.groupby(["frame", "defesa"], as_index=False)["shapley"].sum().rename(columns={"defesa": "jogador", "shapley": "negada"})
    t = presenca[["frame", "jogador"]].merge(contexto[["frame", "fase"]], on="frame").merge(negada, on=["frame", "jogador"], how="left")
    t = t.dropna(subset=["fase"]).fillna({"negada": 0.0})
    g = t.groupby(["fase", "jogador"], as_index=False).agg(momentos=("frame", "size"), negada=("negada", "sum"))
    g["negada_por_momento"] = g["negada"] / g["momentos"]
    fase = g.groupby("fase").agg(total=("negada", "sum"), n=("momentos", "sum"))
    g["media_da_fase"] = g["fase"].map(fase["total"] / fase["n"])
    g["indice"] = 100 * g["negada_por_momento"] / g["media_da_fase"]
    g["percentil"] = np.nan
    elegiveis = g["momentos"] >= minimo
    g.loc[elegiveis, "percentil"] = 100 * g[elegiveis].groupby("fase")["negada_por_momento"].rank(pct=True)
    return g.merge(jogadores, on="jogador", how="left")


def distribuicao_do_indice(indices: pd.DataFrame, minimo: int = MIN_MOMENTOS) -> pd.DataFrame:
    """Mediana, p90 e máximo do índice por fase (jogadores com pelo menos `minimo` momentos na fase), com o nº de jogadores contados
    e dos que ficam de fora. `indices`: saída de `indice_por_fase` (um ou mais jogos juntos)."""
    linhas = []
    for fase, g in indices.groupby("fase"):
        e = g[g["momentos"] >= minimo]["indice"]
        linhas.append({"fase": fase, "jogadores": len(e), "fora_por_poucos_momentos": len(g) - len(e),
                       "mediana": float(e.median()) if len(e) else np.nan, "p90": float(e.quantile(0.9)) if len(e) else np.nan,
                       "maximo": float(e.max()) if len(e) else np.nan})
    return pd.DataFrame(linhas)


# Em linguagem corrente: na moeda do xT, fechar uma saída curta junto à própria área vale quase zero porque ali o xT quase não
# cresce. A moeda «risco» acrescenta ao ganho o que custaria perder a bola nesse sítio: o perigo que o adversário teria se a bola
# fosse cortada onde a linha seria cortada. Perto da área de quem tem a bola esse custo é grande, por isso uma linha tapada ali
# passa a valer muito. O índice por fase compara cada jogador com a média da sua fase (100 = média): assim a pressão alta, onde os
# valores absolutos são minúsculos, e o bloco baixo ficam na mesma escala.
