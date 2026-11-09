"""Fase 5 · funções de avaliação da Cover Shadow Value (critérios C3 a C5 e comparação com a SkillCorner).

Não há variável-alvo: avalia-se por variação de parâmetros, comparação de alternativas e
coerência com o que se passou no jogo. Cada função recebe tabelas já calculadas, por isso testa-se
sem dados reais. O cálculo pesado (uma execução completa da métrica por variante e jogo) fica em
`avaliar_variantes.py`, que guarda os resultados em `data/processed/cover_shadow/avaliacao/`.

Variantes: `base` (valores de partida), uma por cada valor testado de cada parâmetro (um de cada
vez, via `dataclasses.replace` sobre `PADRAO`) e `bola` (cone virado para o lado oposto ao portador).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from metrica import agregacao
from metrica.parametros import PADRAO, PADRAO_V2, PADRAO_V3, Parametros
from scipy.stats import spearmanr

from footy.paths import DATA_PROCESSED

PASTA_AVALIACAO = DATA_PROCESSED / "cover_shadow" / "avaliacao"  # v1: antes da 1.ª ronda de revisão (D24)
PASTA_AVALIACAO_V2 = DATA_PROCESSED / "cover_shadow" / "avaliacao_v2"  # v2: com as decisões D25 a D30
PASTA_AVALIACAO_V3 = DATA_PROCESSED / "cover_shadow" / "avaliacao_v3"  # v3: com D33 e D34
PASTA_AVALIACAO_V31 = DATA_PROCESSED / "cover_shadow" / "avaliacao_v3_1"  # v3.1: v3 + linhas de até 35 m (D43)
PASTAS = {"v1": PASTA_AVALIACAO, "v2": PASTA_AVALIACAO_V2, "v3": PASTA_AVALIACAO_V3, "v3_1": PASTA_AVALIACAO_V31}

# Valores testados em cada parâmetro, um de cada vez; o valor de partida (PADRAO) não se repete.
# Os da primeira avaliação (v1):
VALORES_TESTADOS_V1 = {
    "angulo_cone": (45.0, 90.0),
    "comprimento_cone": (8.0, 12.0),
    "largura_junto_ao_corpo": (1.0, 3.0),
    "tempo_reacao": (0.5, 0.9),
    "velocidade_bola": (12.0, 18.0),
    "incerteza_tempo": (0.3, 0.6),
    "distancia_marcacao": (2.0,),
    "tempo_marcacao": (0.5, 1.5),
}
# Na v2 juntam-se os limiares novos (A11); 0 e 3 desligam a regra (pressão, bloqueio, continuidade):
VALORES_TESTADOS = {
    **VALORES_TESTADOS_V1,
    "distancia_pressao": (0.0, 3.0, 5.0),
    "distancia_saida_marcacao": (3.0, 4.0, 6.0),
    "raio_bloqueio_corpo": (0.0, 0.5, 0.7, 1.5),
    "bloqueio_gradual": (True,),
    "comprimento_passe_curto": (8.0, 15.0),
    "avanco_maximo_atrasado": (0.0, 2.0),
}


# Na v3 só se recalculam as variantes do bloqueio pelo corpo (raio fixo, em m; D34), para a estabilidade relativa.
RAIOS_BLOQUEIO_V3 = (0.0, 0.5, 0.7, 1.0, 1.5)


# Na v3 repete-se também a sensibilidade dos parâmetros antigos (C4); a entrada da marcação testa-se a 3 e 5 m.
VALORES_ANTIGOS_V3 = {**{k: v for k, v in VALORES_TESTADOS_V1.items() if k != "distancia_marcacao"},
                      "distancia_marcacao": (3.0, 5.0)}


def variantes_v3(p: Parametros = PADRAO_V3) -> dict[str, tuple[Parametros, str]]:
    """{nome: (parâmetros, orientação)}: a base v3 (bloqueio gradual) e o bloqueio por raio fixo de cada
    valor de `RAIOS_BLOQUEIO_V3`, uma por cada valor de `VALORES_ANTIGOS_V3` (um parâmetro de cada vez) e
    `bola` (C5), tudo o resto igual à v3."""
    todas = {"base": (p, "tronco")}
    for raio in RAIOS_BLOQUEIO_V3:
        todas[nome_da_variante("raio_bloqueio_corpo", raio)] = (
            replace(p, bloqueio_gradual=False, raio_bloqueio_corpo=raio), "tronco")
    for parametro, valores in VALORES_ANTIGOS_V3.items():
        for valor in valores:
            todas[nome_da_variante(parametro, valor)] = (replace(p, **{parametro: valor}), "tronco")
    todas["bola"] = (p, "bola")  # C5: cone virado para a bola, tudo o resto como na v3
    return todas


# Na v3.1 repetem-se as variantes da v3 e juntam-se os limites de comprimento das linhas (a base tem 35 m).
LIMITES_COMPRIMENTO_V31 = (30.0, 40.0)


def variantes_v31(p: Parametros = PADRAO) -> dict[str, tuple[Parametros, str]]:
    """As variantes da v3 (`variantes_v3`) sobre a base v3.1 e o limite de comprimento das linhas a 30 e a 40 m."""
    todas = variantes_v3(p)
    for limite in LIMITES_COMPRIMENTO_V31:
        todas[nome_da_variante("comprimento_maximo_linha", limite)] = (replace(p, comprimento_maximo_linha=limite), "tronco")
    return todas


def nome_da_variante(parametro: str, valor: float) -> str:
    return f"{parametro}={valor:g}"


def variantes(p: Parametros = PADRAO_V2, testados: dict = VALORES_TESTADOS) -> dict[str, tuple[Parametros, str]]:
    """{nome: (parâmetros, orientação)} de todas as variantes da v2, a base primeiro. Parte da definição
    v2 (`PADRAO_V2`), com que os resultados de `avaliacao_v2/` foram calculados."""
    todas = {"base": (p, "tronco")}
    for parametro, valores in testados.items():
        for valor in valores:
            todas[nome_da_variante(parametro, valor)] = (replace(p, **{parametro: valor}), "tronco")
    todas["bola"] = (p, "bola")
    return todas


# ---------------------------------------------------------------- guardar e ler variantes

def caminho_variante(jogo: int, variante: str, tabela: str, pasta: Path = PASTA_AVALIACAO) -> Path:
    return pasta / f"{jogo}_{variante}_{tabela}.parquet"


def guardar_variante(resultado, jogo: int, variante: str, pasta: Path = PASTA_AVALIACAO) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    resultado.linhas.to_parquet(caminho_variante(jogo, variante, "linhas", pasta), index=False)
    resultado.atribuicoes.to_parquet(caminho_variante(jogo, variante, "atribuicoes", pasta), index=False)


def ler_variante(jogo: int, variante: str, pasta: Path = PASTA_AVALIACAO) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(caminho_variante(jogo, variante, "linhas", pasta)),
            pd.read_parquet(caminho_variante(jogo, variante, "atribuicoes", pasta)))


def variante_calculada(jogo: int, variante: str, pasta: Path = PASTA_AVALIACAO) -> bool:
    return all(caminho_variante(jogo, variante, t, pasta).exists() for t in ("linhas", "atribuicoes"))


# ---------------------------------------------------------------- C4 e C5: comparar variantes

def negada_por_jogador(linhas: pd.DataFrame, atribuicoes: pd.DataFrame, presenca: pd.DataFrame,
                       jogadores: pd.DataFrame) -> pd.DataFrame:
    """Ameaça negada por jogador (D17). `presenca`: frame, posse_id, jogador (os defesas por frame)."""
    lances = agregacao.por_lance(linhas)
    por_posse = agregacao.por_posse_e_defesa(atribuicoes, presenca, lances)
    return agregacao.por_jogador(por_posse, jogadores)


def spearman(a, b) -> float:
    """Correlação de Spearman; NaN se algum dos lados for constante ou houver menos de 3 pares."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def spearman_por_posicao(base: pd.DataFrame, variante: pd.DataFrame, minimo: int = 4) -> dict:
    """Spearman da ameaça negada por jogador entre duas variantes, dentro de cada posição.
    `base`, `variante`: tabelas por jogador com `jogo`, `jogador`, `posicao`, `negada`. A média é
    ponderada pelo número de jogadores de cada posição; posições com menos de `minimo` ficam de fora."""
    t = base.merge(variante, on=["jogo", "jogador", "posicao"], suffixes=("_base", "_var"))
    pares = []
    for posicao, g in t.groupby("posicao"):
        if len(g) >= minimo:
            pares.append((posicao, len(g), spearman(g["negada_base"], g["negada_var"])))
    tabela = pd.DataFrame(pares, columns=["posicao", "n", "rho"]).dropna()
    media = float(np.average(tabela["rho"], weights=tabela["n"])) if len(tabela) else float("nan")
    return {"rho_media": media, "rho_min": float(tabela["rho"].min()) if len(tabela) else float("nan"),
            "posicoes": tabela}


def spearman_por_lance(base: pd.DataFrame, variante: pd.DataFrame) -> float:
    """Spearman da ameaça negada por lance (frame) entre duas variantes, nos lances em que ambas avaliaram
    alguma linha. `base`, `variante`: saída de `agregacao.por_lance`, com uma coluna `jogo`."""
    t = base.merge(variante, on=["jogo", "frame"], suffixes=("_base", "_var"))
    t = t[(t["n_avaliadas_base"] > 0) & (t["n_avaliadas_var"] > 0)]
    return spearman(t["negada_base"], t["negada_var"])


def pct_negada(linhas: pd.DataFrame) -> float:
    """% da ameaça disponível (linhas avaliadas) que foi negada."""
    avaliadas = linhas[~linhas["excluida"]]
    return 100 * avaliadas["ameaca_negada"].sum() / avaliadas["valor_xt"].sum()


def sombras_por_linha(atribuicoes: pd.DataFrame) -> pd.Series:
    """Conjunto de defesas com sombra sobre cada linha (frame, recetor), como texto ordenado."""
    return atribuicoes.groupby(["frame", "recetor"])["defesa"].apply(lambda s: ",".join(map(str, sorted(s))))


def linhas_que_mudam(base: pd.DataFrame, variante: pd.DataFrame, atrib_base: pd.DataFrame,
                     atrib_variante: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por linha de passe avaliada nas duas variantes, com `muda` = quem faz sombra sobre ela
    é diferente, e `dif_negada` = ameaça negada da variante menos a da base (xT)."""
    chave = ["frame", "recetor"]
    t = base[~base["excluida"]][chave + ["valor_xt", "ameaca_negada", "n_sombras"]].merge(
        variante[~variante["excluida"]][chave + ["ameaca_negada", "n_sombras"]], on=chave, suffixes=("_base", "_var"))
    quem_base = sombras_por_linha(atrib_base).rename("quem_base")
    quem_var = sombras_por_linha(atrib_variante).rename("quem_var")
    t = t.join(quem_base, on=chave).join(quem_var, on=chave).fillna({"quem_base": "", "quem_var": ""})
    t["muda"] = t["quem_base"] != t["quem_var"]
    t["dif_negada"] = t["ameaca_negada_var"] - t["ameaca_negada_base"]
    return t


def discordancia_por_lance(mudancas: pd.DataFrame) -> pd.DataFrame:
    """Por frame: ameaça negada que muda entre as duas variantes (soma do valor absoluto, em xT),
    e quantas linhas mudam de sombra."""
    return mudancas.assign(abs_dif=mudancas["dif_negada"].abs()).groupby("frame", as_index=False).agg(
        discordancia=("abs_dif", "sum"), linhas_que_mudam=("muda", "sum"), linhas=("muda", "size"))


# ---------------------------------------------------------------- C3: coerência com o jogo

def passes_do_jogo(eventos: pd.DataFrame) -> pd.DataFrame:
    """Passes reais (eventos `player_possession` que acabam em passe com resultado conhecido):
    posse, portador, destino e se foi completo. Foras de jogo ficam de fora."""
    p = eventos[(eventos["event_type"] == "player_possession") & (eventos["end_type"] == "pass")
                & eventos["pass_outcome"].isin(["successful", "unsuccessful"])
                & eventos["player_targeted_id"].notna()]
    return pd.DataFrame({
        "posse_id": p["event_id"].to_numpy(), "portador": p["player_id"].astype(int).to_numpy(),
        "recetor": p["player_targeted_id"].astype(int).to_numpy(),
        "completo": (p["pass_outcome"] == "successful").to_numpy(),
    })


def ultimo_frame_da_posse(linhas: pd.DataFrame) -> pd.DataFrame:
    """Último frame alinhado de cada posse: o mais próximo do momento do passe."""
    return linhas.groupby("posse_id", as_index=False)["frame"].max()


def passes_com_linha(passes: pd.DataFrame, linhas: pd.DataFrame) -> pd.DataFrame:
    """Junta cada passe real à linha de passe que ele seguiu, no último frame alinhado da posse.
    Tem `p_chegar` e `sombra` (1 − p_chegar) da linha escolhida, `sombra_disponivel` (média das linhas
    avaliadas do frame, incluindo a escolhida) e `percentil`: fração das outras linhas avaliadas com
    menos sombra do que a escolhida (empates valem metade); acima de 0,5 a escolhida está mais à
    sombra do que o habitual. Só passes cuja linha foi avaliada, e posses com mais de uma linha."""
    avaliadas = linhas[~linhas["excluida"]].merge(ultimo_frame_da_posse(linhas), on=["posse_id", "frame"])
    avaliadas = avaliadas.assign(sombra=1 - avaliadas["p_chegar"])
    escolhida = avaliadas.merge(passes[["posse_id", "recetor", "completo"]], on=["posse_id", "recetor"])
    todas = avaliadas.merge(escolhida[["posse_id", "sombra"]].rename(columns={"sombra": "sombra_escolhida"}),
                            on="posse_id")
    todas["menos"] = todas["sombra"] < todas["sombra_escolhida"]
    todas["igual"] = todas["sombra"] == todas["sombra_escolhida"]
    por_posse = todas.groupby("posse_id").agg(sombra_disponivel=("sombra", "mean"), n_linhas=("sombra", "size"),
                                              menos=("menos", "sum"), igual=("igual", "sum"))
    escolhida = escolhida.join(por_posse, on="posse_id")
    escolhida = escolhida[escolhida["n_linhas"] > 1].copy()
    escolhida["percentil"] = (escolhida["menos"] + 0.5 * (escolhida["igual"] - 1)) / (escolhida["n_linhas"] - 1)
    extra = [c for c in ("comprimento", "valor_xt") if c in escolhida]
    return escolhida[["posse_id", "frame", "recetor", "completo", "p_chegar", "sombra", "sombra_disponivel",
                      "n_linhas", "percentil", *extra]].reset_index(drop=True)


def adicionar_comprimento(linhas: pd.DataFrame, tabela: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `comprimento` (m, do portador ao recetor no frame) a cada linha de passe, a partir das
    posições da tabela de trabalho (a mesma que a métrica usou)."""
    posicoes = tabela[["frame", "jogador", "x", "y"]]
    portador = posicoes.rename(columns={"jogador": "portador", "x": "x0", "y": "y0"})
    recetor = posicoes.rename(columns={"jogador": "recetor", "x": "x1", "y": "y1"})
    t = linhas.merge(portador, on=["frame", "portador"]).merge(recetor, on=["frame", "recetor"])
    return t.assign(comprimento=np.hypot(t["x1"] - t["x0"], t["y1"] - t["y0"])).drop(columns=["x0", "y0", "x1", "y1"])


def com_comparaveis(escolhidas: pd.DataFrame, linhas: pd.DataFrame, tol_comprimento: float = 5.0,
                    tol_xt: float = 0.02) -> pd.DataFrame:
    """Controla o comprimento e o ganho de xT: para cada passe, `sombra_comparavel` é a média da sombra das
    OUTRAS linhas avaliadas do mesmo frame cujo comprimento difere até `tol_comprimento` m e cujo ganho de
    xT difere até `tol_xt`; `n_comparaveis` conta-as. `escolhidas` (de `passes_com_linha`) e `linhas` têm de
    ter `comprimento` e `valor_xt`. Os passes sem nenhuma linha comparável ficam com NaN."""
    avaliadas = linhas[~linhas["excluida"]].merge(ultimo_frame_da_posse(linhas), on=["posse_id", "frame"])
    avaliadas = avaliadas.assign(sombra_outra=1 - avaliadas["p_chegar"])
    chave = escolhidas[["posse_id", "recetor", "comprimento", "valor_xt"]].rename(
        columns={"recetor": "recetor_escolhido", "comprimento": "comprimento_escolhido", "valor_xt": "xt_escolhido"})
    t = avaliadas[["posse_id", "recetor", "comprimento", "valor_xt", "sombra_outra"]].merge(chave, on="posse_id")
    parecidas = t[(t["recetor"] != t["recetor_escolhido"])
                  & ((t["comprimento"] - t["comprimento_escolhido"]).abs() <= tol_comprimento)
                  & ((t["valor_xt"] - t["xt_escolhido"]).abs() <= tol_xt)]
    resumo = parecidas.groupby("posse_id").agg(sombra_comparavel=("sombra_outra", "mean"),
                                               n_comparaveis=("sombra_outra", "size"))
    return escolhidas.join(resumo, on="posse_id").fillna({"n_comparaveis": 0})


def comparar_com_comparaveis(escolhidas: pd.DataFrame) -> dict:
    """Diferença de sombra (linha seguida − linhas comparáveis em comprimento e ganho de xT) nos passes que
    têm alguma, com intervalo de confiança, e a mesma diferença face a todas as disponíveis nesse subconjunto
    (para ver quanto muda com o controlo)."""
    t = escolhidas[escolhidas["n_comparaveis"] > 0]
    controlada = intervalo_bootstrap(t["sombra"] - t["sombra_comparavel"])
    sem_controlo = intervalo_bootstrap(t["sombra"] - t["sombra_disponivel"])
    return {"n": len(t), "dif_controlada": controlada[0], "dif_controlada_inf": controlada[1],
            "dif_controlada_sup": controlada[2], "dif_sem_controlo": sem_controlo[0],
            "dif_sem_controlo_inf": sem_controlo[1], "dif_sem_controlo_sup": sem_controlo[2]}


BANDAS_COMPRIMENTO = ((0.0, 10.0), (10.0, 20.0), (20.0, 1000.0))  # m, [inf, sup)


def taxa_por_faixa_e_comprimento(escolhidas: pd.DataFrame, bandas=BANDAS_COMPRIMENTO) -> pd.DataFrame:
    """Taxa de passes completos por faixa de probabilidade de chegar, dentro de cada banda de comprimento do
    passe (com intervalo de Wilson): separa o efeito da sombra do de passes mais longos falharem mais."""
    partes = []
    for inf, sup in bandas:
        g = escolhidas[(escolhidas["comprimento"] >= inf) & (escolhidas["comprimento"] < sup)]
        nome = f"{inf:g} a {sup:g} m" if sup < 1000 else f"{inf:g} m ou mais"
        partes.append(taxa_de_completos_por_faixa(g).assign(comprimento=nome))
    return pd.concat(partes, ignore_index=True)


def intervalo_bootstrap(valores, estatistica=np.mean, repeticoes: int = 2000, nivel: float = 0.95,
                        semente: int = 0) -> tuple[float, float, float]:
    """(estimativa, limite inferior, limite superior) por bootstrap percentil."""
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    gerador = np.random.default_rng(semente)
    reamostras = gerador.choice(v, size=(repeticoes, len(v)), replace=True)
    estimativas = np.apply_along_axis(estatistica, 1, reamostras)
    alfa = (1 - nivel) / 2
    return float(estatistica(v)), float(np.quantile(estimativas, alfa)), float(np.quantile(estimativas, 1 - alfa))


def spearman_com_intervalo(a, b, repeticoes: int = 2000, semente: int = 0) -> tuple[float, float, float]:
    """(Spearman, limite inferior, limite superior): intervalo por bootstrap dos pares."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gerador = np.random.default_rng(semente)
    indices = gerador.integers(0, len(a), size=(repeticoes, len(a)))
    reamostras = [spearman(a[i], b[i]) for i in indices]
    return spearman(a, b), float(np.nanquantile(reamostras, 0.025)), float(np.nanquantile(reamostras, 0.975))


def intervalo_wilson(sucessos: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proporção, limite inferior, limite superior): intervalo de Wilson, bom com amostras pequenas."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = sucessos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margem = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(p), float(centro - margem), float(centro + margem)


def comparar_com_disponiveis(escolhidas: pd.DataFrame) -> dict:
    """Passe feito vs linhas disponíveis no mesmo momento: diferença de sombra (escolhida − média das
    disponíveis), com intervalo de confiança, e percentil médio da linha escolhida. Diferença negativa
    = os passes feitos atravessam menos sombra do que as linhas disponíveis."""
    dif = escolhidas["sombra"] - escolhidas["sombra_disponivel"]
    media, baixo, alto = intervalo_bootstrap(dif)
    perc, perc_baixo, perc_alto = intervalo_bootstrap(escolhidas["percentil"])
    return {"n": len(escolhidas), "sombra_escolhida": float(escolhidas["sombra"].mean()),
            "sombra_disponivel": float(escolhidas["sombra_disponivel"].mean()),
            "dif": media, "dif_inf": baixo, "dif_sup": alto,
            "percentil": perc, "percentil_inf": perc_baixo, "percentil_sup": perc_alto}


FAIXAS_P_CHEGAR = ((0.0, 0.5), (0.5, 0.9), (0.9, 1.0), (1.0, 1.0001))  # [inf, sup); a última = sem sombra


def taxa_de_completos_por_faixa(escolhidas: pd.DataFrame, faixas=FAIXAS_P_CHEGAR) -> pd.DataFrame:
    """Taxa de passes completos por faixa de probabilidade de chegar calculada, com intervalo de Wilson."""
    linhas = []
    for inf, sup in faixas:
        g = escolhidas[(escolhidas["p_chegar"] >= inf) & (escolhidas["p_chegar"] < sup)]
        taxa, baixo, alto = intervalo_wilson(int(g["completo"].sum()), len(g))
        linhas.append({"faixa": f"{inf:g} a {min(sup, 1):g}" if sup <= 1 else "1 (sem sombra)", "n": len(g),
                       "completos": int(g["completo"].sum()), "taxa": taxa, "inf": baixo, "sup": alto})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- comparação com a SkillCorner (D15)

def opcoes_de_passe(eventos: pd.DataFrame) -> pd.DataFrame:
    """Opções de passe da SkillCorner: portador, recetor, frames em que existiram e `xthreat`."""
    o = eventos[eventos["event_type"] == "passing_option"]
    return pd.DataFrame({
        "opcao_id": o["event_id"].to_numpy(), "portador": o["player_in_possession_id"].astype(int).to_numpy(),
        "recetor": o["player_id"].astype(int).to_numpy(), "frame_inicio": o["frame_start"].astype(int).to_numpy(),
        "frame_fim": o["frame_end"].astype(int).to_numpy(), "xthreat": o["xthreat"].to_numpy(),
    })


def linhas_que_coincidem(opcoes: pd.DataFrame, linhas: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por opção da SkillCorner que coincide com linhas nossas: mesmo portador e recetor, em
    frames alinhados dentro da duração da opção. Compara o `xthreat` deles com o nosso `valor_xt`
    (média sobre os frames em que coincidem) e conta os frames."""
    nossas = linhas[["frame", "portador", "recetor", "valor_xt"]]
    t = opcoes.merge(nossas, on=["portador", "recetor"])
    t = t[(t["frame"] >= t["frame_inicio"]) & (t["frame"] <= t["frame_fim"])]
    return t.groupby("opcao_id", as_index=False).agg(
        xthreat=("xthreat", "first"), valor_xt=("valor_xt", "mean"), frames=("frame", "size"))


# ---------------------------------------------------------------- C1 e C2: escolha dos lances a rever

def escolher_lances(lances: pd.DataFrame, discordancia: pd.DataFrame, n_por_grupo: int | dict = 5,
                    semente: int = 0, ja_revistas=()) -> pd.DataFrame:
    """Escolhe `n_por_grupo` lances em cada grupo, um por posse, sem repetir lances entre grupos:
    `negada` (mais ameaça negada), `ao_acaso` e `corpo_vs_bola` (maior discordância).
    `lances`: jogo, frame, posse_id, n_avaliadas, negada. `discordancia`: jogo, frame, discordancia.
    `ja_revistas`: pares (jogo, posse_id) de rondas anteriores; essas posses não voltam a sair.
    `n_por_grupo` pode ser um dicionário por grupo, p. ex. {"negada": 3, "ao_acaso": 3, "corpo_vs_bola": 2}."""
    quantos = n_por_grupo if isinstance(n_por_grupo, dict) else dict.fromkeys(("negada", "ao_acaso", "corpo_vs_bola"), n_por_grupo)
    candidatos = lances[lances["n_avaliadas"] > 0]
    escolhidos, posses_usadas = [], {(int(j), p) for j, p in ja_revistas}

    def tirar(ordenados: pd.DataFrame, grupo: str) -> None:
        n = 0
        for _, r in ordenados.iterrows():
            chave = (r["jogo"], r["posse_id"])
            if chave in posses_usadas:
                continue
            posses_usadas.add(chave)
            escolhidos.append({"grupo": grupo, "jogo": int(r["jogo"]), "frame": int(r["frame"]),
                               "posse_id": r["posse_id"]})
            n += 1
            if n == quantos[grupo]:
                break

    tirar(candidatos.sort_values("negada", ascending=False), "negada")
    tirar(candidatos.sort_values("jogo").sample(frac=1, random_state=semente), "ao_acaso")
    por_discordancia = candidatos.merge(discordancia, on=["jogo", "frame"]).sort_values("discordancia", ascending=False)
    tirar(por_discordancia, "corpo_vs_bola")
    return pd.DataFrame(escolhidos)


# Em linguagem corrente: este módulo responde às perguntas da fase 5 sem julgar. A métrica corre
# várias vezes, mexendo num parâmetro de cada vez, e aqui compara-se cada corrida com a de partida:
# quanta ameaça passa a negar e se os jogadores e os lances ficam pela mesma ordem (Spearman).
# Compara-se ainda o cone pelo corpo com o cone virado para a bola, os passes que as equipas
# fizeram com as linhas que tinham disponíveis e com a taxa de passes completos, e o perigo
# calculado com o da SkillCorner. As mesmas funções escolhem os lances a mostrar a um treinador.
