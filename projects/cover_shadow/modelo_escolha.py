"""Fase 8b · logit condicional da escolha do passe ("se passar, para quem?").

Em cada passe o portador escolhe uma das linhas do conjunto de escolhas (`escolha.py`). O modelo dá a
cada linha uma pontuação `x · b` e a probabilidade de ser a escolhida é a pontuação em softmax dentro do
passe (logit condicional). Ajusta-se por máxima verosimilhança; o erro-padrão vem da matriz de
informação (o Hessiano), por isso não se precisa de `statsmodels`.

Três modelos, com exatamente os mesmos passes e linhas (D37):
- M0: variáveis da linha, sem sombra;
- M1: M0 + sombra virada para a bola;
- M2: M0 + sombra pelo tronco (a da v3);
- M3: M0 + as duas sombras ao mesmo tempo (complemento da fase 8b). Também se ajusta como sombra_bola +
  (sombra_tronco − sombra_bola): é o mesmo modelo noutra parametrização, e o coeficiente da diferença
  lê-se como a informação que o tronco acrescenta à bola.
As linhas em que algum defesa não tem pose levam a sombra só dos defesas com pose (valor mínimo) e
`linha_sem_pose = 1`.
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pandas as pd
from entradas import exigir
from escolha import LIMITE_INTERVALO_S, PASTA_SAIDA
from scipy.optimize import minimize

from footy.paths import output_dir

LIMITE_COMPRIMENTO = 35.0  # m, D37: vale para o modelo, para as linhas e, depois, para a métrica v4
BASE = ["comprimento", "avanco", "valor_xt", "recetor_marcado", "dist_adversario", "linha_sem_pose"]
MODELOS = {"M0": [], "M1": ["sombra_bola"], "M2": ["sombra_tronco"], "M3": ["sombra_bola", "sombra_tronco"]}
M3_DIFERENCA = ["sombra_bola", "tronco_menos_bola"]  # o M3 reparametrizado
SO_COMPRIMENTO = ["comprimento"]
SEMENTE = 20251104
REPETICOES_BOOTSTRAP = 5000


# ---------------------------------------------------------------- dados

def preparar_dados(opcoes: pd.DataFrame, limite_comprimento: float = LIMITE_COMPRIMENTO,
                   limite_intervalo: float = LIMITE_INTERVALO_S, so_conjuntos_completos: bool = False) -> pd.DataFrame:
    """Linhas do modelo: passes rasteiros (`high_pass` não é verdadeiro e comprimento até `limite_comprimento`),
    frame da escolha até `limite_intervalo` s do passe, só linhas até esse comprimento, e só passes em que a
    linha escolhida ficou no conjunto. Acrescenta `passe` (identificador único), `linha_sem_pose` e as sombras
    sem valores em falta (`sombra_*_pose`). Com `so_conjuntos_completos`, descarta os passes com alguma linha sem pose."""
    o = opcoes[(opcoes["intervalo_s"] <= limite_intervalo) & opcoes["high_pass"].ne(True).fillna(True)
               & (opcoes["comprimento_passe"] <= limite_comprimento) & (opcoes["comprimento"] <= limite_comprimento)].copy()
    o["passe"] = o["jogo"].astype(str) + "_" + o["posse_id"]
    o["linha_sem_pose"] = (o["motivo"] == "sem_pose").astype(int)
    o["sombra_tronco"], o["sombra_bola"] = o["sombra_tronco_pose"], o["sombra_bola_pose"]
    o["tronco_menos_bola"] = o["sombra_tronco"] - o["sombra_bola"]
    g = o.groupby("passe")
    ok = (g["escolhida"].transform("sum") == 1) & (g["escolhida"].transform("size") >= 2)
    if so_conjuntos_completos:
        ok &= g["linha_sem_pose"].transform("sum") == 0
    return o[ok].sort_values(["passe", "recetor"]).reset_index(drop=True)


def funil_de_passes(opcoes: pd.DataFrame) -> pd.DataFrame:
    """Passes por jogo em cada passo da definição D37, e com as opções A (conjuntos completos) e B (marcar)."""
    passes = opcoes.drop_duplicates(["jogo", "posse_id"])
    rasteiros = passes[passes["high_pass"].ne(True).fillna(True) & (passes["comprimento_passe"] <= LIMITE_COMPRIMENTO)]
    no_limite = rasteiros[rasteiros["intervalo_s"] <= LIMITE_INTERVALO_S]
    b = preparar_dados(opcoes)
    a = preparar_dados(opcoes, so_conjuntos_completos=True)
    passos = {"passes com linhas no frame da escolha": passes, "rasteiros (não pelo ar, ≤ 35 m)": rasteiros,
              "frame da escolha ≤ 0,4 s do passe": no_limite,
              "B · linha escolhida no conjunto (≤ 35 m)": b.drop_duplicates("passe"),
              "A · todas as linhas com pose": a.drop_duplicates("passe")}
    return pd.DataFrame({nome: d.groupby("jogo").size() for nome, d in passos.items()}).T.assign(
        total=lambda t: t.sum(axis=1))


# ---------------------------------------------------------------- o modelo

class Dados(NamedTuple):
    """Linhas ordenadas por passe: `X` (linhas × variáveis), `escolhida` (0/1) e `inicios` (primeira linha de cada passe)."""
    X: np.ndarray
    escolhida: np.ndarray
    inicios: np.ndarray


def montar(tabela: pd.DataFrame, variaveis: list[str]) -> Dados:
    """`tabela` tem de vir ordenada por `passe` (como sai de `preparar_dados`)."""
    inicios = np.flatnonzero(~tabela["passe"].duplicated().to_numpy())
    return Dados(tabela[variaveis].to_numpy(float), tabela["escolhida"].to_numpy(float), inicios)


def _por_passe(valores: np.ndarray, inicios: np.ndarray) -> np.ndarray:
    return np.add.reduceat(valores, inicios, axis=0)


def probabilidades(d: Dados, coef: np.ndarray) -> np.ndarray:
    """Probabilidade de cada linha ser a escolhida (softmax dentro do passe)."""
    pontuacao = d.X @ coef
    n = np.diff(np.append(d.inicios, len(pontuacao)))
    pontuacao = pontuacao - np.repeat(np.maximum.reduceat(pontuacao, d.inicios), n)  # estabilidade numérica
    e = np.exp(pontuacao)
    return e / np.repeat(_por_passe(e, d.inicios), n)


def _menos_log_verosimilhanca(coef, d: Dados):
    p = probabilidades(d, coef)
    return -np.log(p[d.escolhida == 1]).sum()


def _gradiente(coef, d: Dados):
    p = probabilidades(d, coef)
    return -(d.X * (d.escolhida - p)[:, None]).sum(axis=0)


def _hessiano(coef, d: Dados):
    """Soma, por passe, da covariância das variáveis sob as probabilidades do modelo."""
    p = probabilidades(d, coef)
    media = _por_passe(d.X * p[:, None], d.inicios)  # variáveis esperadas em cada passe
    return (d.X * p[:, None]).T @ d.X - media.T @ media


class Ajuste(NamedTuple):
    variaveis: list[str]
    coef: np.ndarray
    cov: np.ndarray
    log_verosimilhanca: float
    n_passes: int


def ajustar(tabela: pd.DataFrame, variaveis: list[str]) -> Ajuste:
    d = montar(tabela, variaveis)
    r = minimize(_menos_log_verosimilhanca, np.zeros(len(variaveis)), args=(d,), jac=_gradiente, hess=_hessiano,
                 method="trust-exact")
    if not r.success and np.abs(r.jac).max() > 1e-4:
        raise RuntimeError(f"o ajuste não convergiu: {r.message}")
    return Ajuste(variaveis, r.x, np.linalg.inv(_hessiano(r.x, d)), -r.fun, len(d.inicios))


def tabela_de_coeficientes(a: Ajuste) -> pd.DataFrame:
    """Coeficiente, erro-padrão, IC 95 % e razão de chances (exp(coef)) por unidade de cada variável."""
    ep = np.sqrt(np.diag(a.cov))
    t = pd.DataFrame({"coef": a.coef, "erro_padrao": ep, "ic95_inf": a.coef - 1.96 * ep, "ic95_sup": a.coef + 1.96 * ep},
                     index=a.variaveis)
    t["razao_de_chances"] = np.exp(t["coef"])
    t["rc_ic95_inf"], t["rc_ic95_sup"] = np.exp(t["ic95_inf"]), np.exp(t["ic95_sup"])
    return t


# ---------------------------------------------------------------- avaliação fora da amostra

def avaliar_passes(tabela: pd.DataFrame, a: Ajuste) -> pd.DataFrame:
    """Por passe: log-loss (−log da probabilidade da linha escolhida), top-1 e top-3, e número de linhas."""
    d = montar(tabela, a.variaveis)
    return avaliar_probabilidades(tabela, probabilidades(d, a.coef))


def avaliar_probabilidades(tabela: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Como `avaliar_passes`, a partir de uma pontuação `p` por linha (a probabilidade, ou só uma ordem para o
    top-k). `tabela` vem ordenada por `passe`."""
    inicios = np.flatnonzero(~tabela["passe"].duplicated().to_numpy())
    escolhida = tabela["escolhida"].to_numpy(float)
    passe = np.repeat(np.arange(len(inicios)), np.diff(np.append(inicios, len(p))))
    p_escolhida = np.zeros(len(inicios))
    p_escolhida[passe[escolhida == 1]] = p[escolhida == 1]
    melhores = np.bincount(passe, weights=(p > p_escolhida[passe]).astype(float))  # linhas mais prováveis que a escolhida
    return pd.DataFrame({"passe": tabela["passe"].to_numpy()[inicios], "log_loss": -np.log(p_escolhida),
                         "top1": melhores == 0, "top3": melhores < 3,
                         "n_linhas": np.diff(np.append(inicios, len(p)))})


def referencia_ao_acaso(tabela: pd.DataFrame) -> dict:
    """Escolha ao acaso entre as linhas do passe: log-loss log(n), top-1 1/n, top-3 min(3, n)/n."""
    n = tabela.groupby("passe").size().to_numpy()
    return {"log_loss": float(np.log(n).mean()), "top1": float((1 / n).mean()), "top3": float((np.minimum(3, n) / n).mean())}


def resumo(passes: pd.DataFrame) -> dict:
    return {"log_loss": float(passes["log_loss"].mean()), "top1": float(passes["top1"].mean()),
            "top3": float(passes["top3"].mean()), "n_passes": len(passes)}


def intervalo_bootstrap(diferencas: np.ndarray, repeticoes: int = REPETICOES_BOOTSTRAP, semente: int = SEMENTE) -> dict:
    """Média da diferença por passe e IC 95 % (percentis) por bootstrap sobre os passes."""
    rng = np.random.default_rng(semente)
    medias = rng.choice(diferencas, size=(repeticoes, len(diferencas))).mean(axis=1)
    inf, sup = np.percentile(medias, [2.5, 97.5])
    return {"diferenca": float(diferencas.mean()), "ic95_inf": float(inf), "ic95_sup": float(sup)}


def fora_da_amostra(dados: pd.DataFrame) -> dict:
    """Ajusta num jogo e testa no outro, nos dois sentidos. Por direção: resumo de cada modelo e das referências,
    e a diferença de log-loss (por passe) de M2 contra M1 e M0 e de M3 contra M2, M1 e M0, com IC por bootstrap
    (negativa = o primeiro modelo prevê melhor)."""
    jogos = sorted(dados["jogo"].unique())
    resultado = {}
    for treino_jogo in jogos:
        teste_jogo = next(j for j in jogos if j != treino_jogo)
        treino, teste = dados[dados["jogo"] == treino_jogo], dados[dados["jogo"] == teste_jogo]
        por_passe = {m: avaliar_passes(teste, ajustar(treino, BASE + extra)) for m, extra in MODELOS.items()}
        por_passe["so_comprimento"] = avaliar_passes(teste, ajustar(treino, SO_COMPRIMENTO))
        direcao = {"treino": int(treino_jogo), "teste": int(teste_jogo),
                   "modelos": {m: resumo(p) for m, p in por_passe.items()}, "ao_acaso": referencia_ao_acaso(teste)}
        for modelo, outro in (("M2", "M1"), ("M2", "M0"), ("M3", "M2"), ("M3", "M1"), ("M3", "M0")):
            direcao[f"{modelo}_menos_{outro}"] = intervalo_bootstrap(
                por_passe[modelo]["log_loss"].to_numpy() - por_passe[outro]["log_loss"].to_numpy())
        resultado[f"{treino_jogo}_para_{teste_jogo}"] = direcao
    return resultado


# ---------------------------------------------------------------- verificações e resultados

def correlacoes(dados: pd.DataFrame) -> pd.DataFrame:
    return dados[[*BASE, "sombra_bola", "sombra_tronco"]].corr()


def passes_com_duas_opcoes_da_skillcorner(dados: pd.DataFrame) -> int:
    """Passes do modelo em que a SkillCorner lista pelo menos duas opções (para a comparação opcional)."""
    return int((dados.groupby("passe")["xpass_sc"].count() >= 2).sum())


def passes_com_opcoes_da_skillcorner(dados: pd.DataFrame, minimo: int = 2) -> pd.DataFrame:
    """Só as linhas que a SkillCorner lista (`score_sc` existe), nos passes em que lista pelo menos `minimo`
    opções e em que a linha escolhida também está listada (sem ela não há log-loss nem top-k a comparar)."""
    listadas = dados[dados["score_sc"].notna()]
    g = listadas.groupby("passe")
    ok = (g["score_sc"].transform("size") >= minimo) & (g["escolhida"].transform("sum") == 1)
    return listadas[ok].reset_index(drop=True)


def renormalizar(valores: np.ndarray, tabela: pd.DataFrame) -> np.ndarray:
    """Divide `valores` pela soma dentro de cada passe (`tabela` ordenada por passe): fica uma distribuição por passe."""
    inicios = np.flatnonzero(~tabela["passe"].duplicated().to_numpy())
    n = np.diff(np.append(inicios, len(valores)))
    return valores / np.repeat(_por_passe(valores, inicios), n)


def comparar_com_skillcorner(dados: pd.DataFrame) -> dict:
    """Nos passes com pelo menos duas opções listadas pela SkillCorner: top-1 e top-3 com o `passing_option_score`
    dela e com o M2 (ajustado no outro jogo) restrito às mesmas opções, com as probabilidades renormalizadas.
    O score não é uma probabilidade (ver `descrever_score_da_skillcorner`): o "score normalizado" divide-o pela
    soma das opções listadas, só para haver uma distribuição com que calcular a log-loss."""
    partes = {"M2_restrito": [], "score_sc": [], "score_sc_normalizado": []}
    testes = []
    for treino_jogo in sorted(dados["jogo"].unique()):
        treino = dados[dados["jogo"] == treino_jogo]
        teste = passes_com_opcoes_da_skillcorner(dados[dados["jogo"] != treino_jogo])
        a = ajustar(treino, BASE + MODELOS["M2"])
        p_m2 = renormalizar(probabilidades(montar(teste, a.variaveis), a.coef), teste)
        score = teste["score_sc"].to_numpy(float)
        testes.append(teste)
        partes["M2_restrito"].append(avaliar_probabilidades(teste, p_m2))
        partes["score_sc"].append(avaliar_probabilidades(teste, score))
        partes["score_sc_normalizado"].append(avaliar_probabilidades(teste, renormalizar(score, teste)))
    passes = {nome: pd.concat(d, ignore_index=True) for nome, d in partes.items()}
    resultado = {nome: resumo(p) for nome, p in passes.items()}
    resultado["ao_acaso"] = {**referencia_ao_acaso(pd.concat(testes)), "n_passes": len(passes["score_sc"])}
    resultado["score_sc"].pop("log_loss")  # o score não é uma probabilidade: do score cru só se reportam top-1 e top-3
    resultado["M2_restrito_menos_score_normalizado"] = intervalo_bootstrap(
        passes["M2_restrito"]["log_loss"].to_numpy() - passes["score_sc_normalizado"]["log_loss"].to_numpy())
    return resultado


def descrever_score_da_skillcorner(dados: pd.DataFrame) -> dict:
    """O que se vê no `passing_option_score` nas linhas listadas dos passes com ≥ 2 opções: intervalo, soma por passe
    (probabilidades de escolha somariam 1) e valor médio na linha escolhida e nas outras."""
    t = passes_com_opcoes_da_skillcorner(dados)
    soma = t.groupby("passe")["score_sc"].sum()
    return {"minimo": float(t["score_sc"].min()), "maximo": float(t["score_sc"].max()),
            "soma_por_passe_mediana": float(soma.median()), "soma_por_passe_min": float(soma.min()),
            "soma_por_passe_max": float(soma.max()),
            "score_medio_escolhida": float(t.loc[t["escolhida"] == 1, "score_sc"].mean()),
            "score_medio_outras": float(t.loc[t["escolhida"] == 0, "score_sc"].mean()),
            "passes": int(t["passe"].nunique()), "linhas": len(t)}


def calcular() -> dict:
    """Todos os números da fase 8b: funil, ajuste conjunto, avaliação fora da amostra e verificações."""
    opcoes = pd.read_parquet(exigir(PASTA_SAIDA / "escolhas_v3.parquet", "escolha.py"))
    dados = preparar_dados(opcoes)
    ajustes = {m: ajustar(dados, BASE + extra) for m, extra in MODELOS.items()}
    ajustes["M3 (bola + diferença)"] = ajustar(dados, BASE + M3_DIFERENCA)  # o M3 com outra parametrização
    conjunto = {m: tabela_de_coeficientes(a) for m, a in ajustes.items()}
    return {
        "funil": funil_de_passes(opcoes), "dados": dados, "conjunto": conjunto,
        "log_verosimilhanca": {m: a.log_verosimilhanca for m, a in ajustes.items()},
        "fora_da_amostra": fora_da_amostra(dados), "correlacoes": correlacoes(dados),
        "passes_por_jogo": {int(j): int(n) for j, n in dados.drop_duplicates("passe").groupby("jogo").size().items()},
        "passes_com_duas_opcoes_sc": passes_com_duas_opcoes_da_skillcorner(dados),
        "ao_acaso_conjunto": referencia_ao_acaso(dados),
        "skillcorner": comparar_com_skillcorner(dados), "score_skillcorner": descrever_score_da_skillcorner(dados),
    }


def para_json(r: dict) -> dict:
    return {
        "definicao": {"limite_comprimento_m": LIMITE_COMPRIMENTO, "limite_intervalo_s": LIMITE_INTERVALO_S,
                      "linhas_sem_pose": "B: sombra só com os defesas com pose e linha_sem_pose = 1",
                      "variaveis_base": BASE, "bootstrap": {"repeticoes": REPETICOES_BOOTSTRAP, "semente": SEMENTE}},
        "funil": json.loads(r["funil"].to_json(orient="index")),
        "passes_por_jogo": r["passes_por_jogo"], "passes_com_duas_opcoes_sc": r["passes_com_duas_opcoes_sc"],
        "log_verosimilhanca": r["log_verosimilhanca"],
        "coeficientes_conjunto": {m: json.loads(t.to_json(orient="index")) for m, t in r["conjunto"].items()},
        "fora_da_amostra": r["fora_da_amostra"], "ao_acaso_conjunto": r["ao_acaso_conjunto"],
        "skillcorner": r["skillcorner"], "score_skillcorner": r["score_skillcorner"],
        "correlacoes": json.loads(r["correlacoes"].to_json(orient="index")),
    }


def main() -> None:
    r = calcular()
    destino = output_dir("cover_shadow") / "modelo_escolha.json"
    destino.write_text(json.dumps(para_json(r), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{destino}\n{r['funil']}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: cada passe é uma pergunta de escolha múltipla: o jogador tinha cerca de oito
# colegas e escolheu um. O modelo dá a cada colega uma pontuação a partir de coisas como a distância,
# o avanço e o perigo, e transforma as pontuações em probabilidades que somam 1. Procura os pesos que
# tornam as escolhas reais o mais prováveis possível. Compara-se depois com versões que sabem também
# quanto a sombra de um defesa tapa cada linha. Para ser justo, treina-se num jogo e testa-se no outro,
# e mede-se quão bem o modelo previa o colega que acabou por receber a bola.
