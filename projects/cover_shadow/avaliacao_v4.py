"""Fase 8d · avaliação da métrica v4 (D38) com os critérios C4, C5 e C8, sem veredictos: só números.

O cálculo pesado (uma execução da v4 por variante e jogo) fica em `avaliar_variantes_v4.py`, que guarda as tabelas em
`data/processed/cover_shadow/avaliacao_v4/`. Aqui estão as definições das variantes, a leitura dessas tabelas e as
funções que as resumem (cada uma recebe tabelas já calculadas, por isso testa-se sem dados reais).

Variantes: as da v3 (`avaliacao.variantes_v3`: parâmetros do cone, tempos, velocidades, limiares da marcação e bloqueio
pelo corpo, mais `bola`), mais o limite de comprimento das linhas (30 e 40 m; o de 35 m é a base). Os coeficientes do
modelo de escolha são os do M2 ajustados no outro jogo (os do M1 na variante `bola`) e não se reajustam por variante.
Na base calculam-se ainda, sobre as mesmas linhas e sombras, os coeficientes ajustados com os dois jogos e `N_CONJUNTOS`
conjuntos tirados da distribuição estimada dos coeficientes (`alt_*`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import avaliacao as av
import numpy as np
import pandas as pd
from metrica import agregacao
from metrica.parametros import PADRAO, Parametros

from footy.paths import DATA_PROCESSED

PASTA = DATA_PROCESSED / "cover_shadow" / "avaliacao_v4"  # v4 sem fundo (D38)
PASTA_FUNDO = DATA_PROCESSED / "cover_shadow" / "avaliacao_v4_fundo"  # v4 com fundo (D39): só as variantes de VARIANTES_FUNDO
VARIANTES_FUNDO = ("base", "bola", *(f"raio_bloqueio_corpo={r}" for r in ("0", "0.5", "0.7", "1", "1.5")), "limite=30", "limite=40")
LIMITES_COMPRIMENTO = (30.0, 40.0)  # m; a base usa 35 (D37)
N_CONJUNTOS = 50  # conjuntos de coeficientes tirados da distribuição estimada (incerteza, C4)
ALTERNATIVOS_DA_BASE = ("ambos", *(f"inc{k:02d}" for k in range(N_CONJUNTOS)))
COLUNAS_LINHAS = ["frame", "posse_id", "recetor", "comprimento", "valor_xt", "linha_sem_pose", "s0", "sC", "pC",
                  "n_sombras", "quem_sombra", "negada_linha", "p_fundo"]


@dataclass(frozen=True)
class VarianteV4:
    parametros: Parametros
    orientacao: str = "tronco"
    limite: float = 35.0


def variantes_v4(p: Parametros = PADRAO) -> dict[str, VarianteV4]:
    """{nome: variante}: as da v3 com os mesmos nomes (a `base` primeiro) e o limite de comprimento das linhas."""
    todas = {nome: VarianteV4(par, orient) for nome, (par, orient) in av.variantes_v3(p).items()}
    for limite in LIMITES_COMPRIMENTO:
        todas[f"limite={limite:g}"] = VarianteV4(p, "tronco", limite)
    return todas


# ---------------------------------------------------------------- guardar e ler

def caminho(jogo: int, variante: str, tabela: str, pasta: Path = PASTA) -> Path:
    return pasta / f"{jogo}_{variante}_{tabela}.parquet"


def tabelas_da_variante(variante: str, pasta: Path = PASTA) -> tuple[str, ...]:
    """Tabelas de uma variante; a base da v4 sem fundo tem ainda as dos conjuntos alternativos de coeficientes."""
    return ("momentos", "atribuicoes", "linhas") + (("alt_momentos", "alt_atribuicoes") if variante == "base" and pasta == PASTA else ())


def variante_calculada(jogo: int, variante: str, pasta: Path = PASTA) -> bool:
    return all(caminho(jogo, variante, t, pasta).exists() for t in tabelas_da_variante(variante, pasta))


def guardar(resultado: dict, jogo: int, variante: str, pasta: Path = PASTA) -> None:
    """Escreve as tabelas de uma variante. Os momentos e as atribuições têm as colunas completas (I, D e Shapley nas três
    ordens); as linhas só as colunas de `COLUNAS_LINHAS`; as alternativas (só na base) vêm numa tabela com a coluna `alt`."""
    pasta.mkdir(parents=True, exist_ok=True)
    resultado["momentos"].to_parquet(caminho(jogo, variante, "momentos", pasta), index=False)
    resultado["atribuicoes"].to_parquet(caminho(jogo, variante, "atribuicoes", pasta), index=False)
    resultado["linhas"][COLUNAS_LINHAS].to_parquet(caminho(jogo, variante, "linhas", pasta), index=False)
    if resultado.get("alternativos"):
        for tabela in ("momentos", "atribuicoes"):
            pd.concat([d[tabela].assign(alt=nome) for nome, d in resultado["alternativos"].items()], ignore_index=True
                      ).to_parquet(caminho(jogo, variante, f"alt_{tabela}", pasta), index=False)


def ler(jogo: int, variante: str, pasta: Path = PASTA) -> dict:
    return {t: pd.read_parquet(caminho(jogo, variante, t, pasta)) for t in tabelas_da_variante(variante, pasta)}


# ---------------------------------------------------------------- agregação (como na v3: média por posse, soma por jogador)

def negada_por_jogador(momentos: pd.DataFrame, atribuicoes: pd.DataFrame, presenca: pd.DataFrame, jogadores: pd.DataFrame,
                       coluna: str = "shapley_simetrica") -> pd.DataFrame:
    """Ameaça negada por jogador (total) com o Shapley da `coluna` das atribuições (nas tabelas leves, `shapley`);
    `disponivel` = E(∅). `presenca`: frame, posse_id, jogador (todos os frames da posse)."""
    lances = momentos.rename(columns={"E0": "disponivel"})[["frame", "disponivel"]]
    a = atribuicoes[["frame", "defesa"]].assign(shapley=atribuicoes[coluna])
    return agregacao.por_jogador(agregacao.por_posse_e_defesa(a, presenca, lances), jogadores)


def pct_negada(momentos: pd.DataFrame) -> float:
    """% do perigo esperado sem sombras (E(∅)) que as sombras negam, somado sobre os momentos."""
    return float(100 * momentos["v"].sum() / momentos["E0"].sum())


# ---------------------------------------------------------------- C4: estabilidade

def comparar_com_a_base(base: pd.DataFrame, variante: pd.DataFrame, minimo: int = 4) -> dict:
    """ρ de Spearman por jogador dentro da posição entre a base e uma variante (média ponderada e mínimo), como no C4 da
    v3 (`av.spearman_por_posicao`, jogadores dos dois jogos juntos). `base`, `variante`: jogo, jogador, posicao, negada."""
    r = av.spearman_por_posicao(base, variante, minimo)
    return {"rho_media": r["rho_media"], "rho_min": r["rho_min"], "n_posicoes": len(r["posicoes"])}


def resumo_das_variantes(por_variante: dict, pct: dict, base: str = "base") -> pd.DataFrame:
    """Uma linha por variante: ρ face à base (média e mínimo) e % negada por jogo. `por_variante`: {nome: tabela
    por jogador (com `jogo`)}; `pct`: {nome: {jogo: % negada}}."""
    linhas = []
    for nome, t in por_variante.items():
        r = {"variante": nome, **comparar_com_a_base(por_variante[base], t)}
        r.update({f"pct_negada_{jogo}": v for jogo, v in pct[nome].items()})
        linhas.append(r)
    return pd.DataFrame(linhas)


def distribuicao_das_alternativas(rhos: pd.DataFrame) -> dict:
    """Resumo (média, mínimo, máximo e percentis 2,5 e 97,5) de cada coluna de `rhos` (uma linha por conjunto)."""
    return {c: {"media": float(rhos[c].mean()), "min": float(rhos[c].min()), "max": float(rhos[c].max()),
                "p2_5": float(rhos[c].quantile(0.025)), "p97_5": float(rhos[c].quantile(0.975))} for c in rhos.columns}


# ---------------------------------------------------------------- C5: corpo vs bola

def linhas_que_mudam(tronco: pd.DataFrame, bola: pd.DataFrame) -> dict:
    """Nas linhas que existem nas duas versões (frame, recetor): quantas, e a % em que muda o conjunto de defesas que
    faz sombra. Também a % só entre as linhas com sombra em alguma das duas versões."""
    chave = ["frame", "recetor"]
    t = tronco[chave + ["quem_sombra"]].merge(bola[chave + ["quem_sombra"]], on=chave, suffixes=("_tronco", "_bola"))
    muda = t["quem_sombra_tronco"] != t["quem_sombra_bola"]
    com_sombra = (t["quem_sombra_tronco"] != "") | (t["quem_sombra_bola"] != "")
    return {"linhas": len(t), "linhas_que_mudam": int(muda.sum()), "pct_muda": float(100 * muda.mean()),
            "linhas_com_sombra_em_alguma": int(com_sombra.sum()),
            "pct_muda_entre_as_com_sombra": float(100 * muda[com_sombra].mean()) if com_sombra.any() else float("nan")}


# ---------------------------------------------------------------- secção 5: descrição para o artigo

COMPRIMENTOS = [(0, 10), (10, 20), (20, 30), (30, 35)]
ROTULOS_COMPRIMENTO = ["0–10 m", "10–20 m", "20–30 m", "30–35 m"]


def por_comprimento(comprimento: pd.Series, negada: pd.Series, disponivel: pd.Series) -> pd.DataFrame:
    """Ameaça negada e disponível por intervalo de comprimento da linha (0–10, 10–20, 20–30, 30–35 m; > 35 m à parte),
    com a parte de cada intervalo no total. Intervalos abertos à esquerda, fechados à direita, o primeiro com o 0."""
    limites = [0, 10, 20, 30, 35, np.inf]
    rotulos = [*ROTULOS_COMPRIMENTO, "> 35 m"]
    faixa = pd.cut(comprimento, limites, labels=rotulos, include_lowest=True)
    t = pd.DataFrame({"faixa": faixa, "negada": negada, "disponivel": disponivel}).groupby("faixa", observed=False).agg(
        linhas=("negada", "size"), negada=("negada", "sum"), disponivel=("disponivel", "sum"))
    t["pct_da_negada"] = 100 * t["negada"] / t["negada"].sum()
    t["pct_do_disponivel"] = 100 * t["disponivel"] / t["disponivel"].sum()
    return t.reindex(rotulos)


def peso_de_I_e_D(por_jogador: pd.DataFrame) -> pd.DataFrame:
    """Por posição: soma da ameaça negada total, da parte de interceção (I) e de dissuasão (D), e o peso de cada parte
    no total (%). `por_jogador`: posicao, negada, negada_I, negada_D."""
    t = por_jogador.groupby("posicao").agg(jogadores=("jogador", "size"), negada=("negada", "sum"),
                                           negada_I=("negada_I", "sum"), negada_D=("negada_D", "sum"))
    t["pct_I"] = 100 * t["negada_I"] / t["negada"]
    t["pct_D"] = 100 * t["negada_D"] / t["negada"]
    return t


def lances_v4_por_zona_ou_fase(momentos: pd.DataFrame, contexto: pd.DataFrame, por: str) -> pd.DataFrame:
    """% negada (v / E(∅)) por terço ou fase defensiva, com o nº de momentos com alguma linha. Usa `agregacao.por_zona_ou_fase`
    com a ameaça disponível da v4 (E(∅)) e a negada (v)."""
    lances = momentos.rename(columns={"n_linhas": "n_avaliadas", "E0": "disponivel", "v": "negada"})[
        ["frame", "n_avaliadas", "disponivel", "negada"]]
    return agregacao.por_zona_ou_fase(lances, contexto, por)


# ---------------------------------------------------------------- C8: resumo da fase 8b

def tabela_c8(modelo_escolha: dict) -> dict:
    """Do `modelo_escolha.json`: log-loss, top-1 e top-3 fora da amostra de M0–M3 (e das referências) nas duas direções,
    as diferenças com IC 95 %, e os coeficientes do M3 ajustados com os dois jogos."""
    fora = modelo_escolha["fora_da_amostra"]
    linhas = []
    for direcao, d in fora.items():
        for modelo, r in d["modelos"].items():
            linhas.append({"direcao": direcao, "treino": d["treino"], "teste": d["teste"], "modelo": modelo, **r})
        linhas.append({"direcao": direcao, "treino": d["treino"], "teste": d["teste"], "modelo": "ao acaso",
                       **d["ao_acaso"], "n_passes": d["modelos"]["M0"]["n_passes"]})
    diferencas = [{"direcao": direcao, "comparacao": chave.replace("_menos_", " − "), **valor}
                  for direcao, d in fora.items() for chave, valor in d.items() if "_menos_" in chave]
    m3 = pd.DataFrame(modelo_escolha["coeficientes_conjunto"]["M3"]).T
    return {"modelos": pd.DataFrame(linhas), "diferencas": pd.DataFrame(diferencas),
            "coeficientes_M3": m3[["coef", "erro_padrao", "ic95_inf", "ic95_sup", "razao_de_chances"]]}


# Em linguagem corrente: este módulo pega nas tabelas que a v4 deixa em disco para cada variante dos parâmetros e
# responde a perguntas de números: se mudar um parâmetro, a ordem dos jogadores de cada posição mantém-se? Quanta
# ameaça se nega? E se em vez de o corpo se olhar para a bola, quem faz sombra muda? Não escolhe vencedores nem
# concorda ou discorda de nada: só calcula.
