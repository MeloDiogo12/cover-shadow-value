"""Fase 8f · calibração do modelo completo de p com os passes reais (D40).

Para cada passe rasteiro (D37: `high_pass` não verdadeiro, ≤ 35 m, frame da escolha ≤ 0,4 s do passe) olha-se para a linha
escolhida no frame da escolha e calcula-se `p(θ)` com o modelo completo (`metrica.passe`): sombras (cone) × fundo (fora do cone).
O resultado observado é `pass_outcome`: `successful` = chegou; `unsuccessful` = não chegou (a SkillCorner não distingue
interceção de bola fora ou passe mal executado: um só rótulo); `offside` fica de fora (a bola chegou, o que falhou foi a posição).

θ = tempo de reação, velocidade do defesa, incerteza (σ), fim do bloqueio pelo corpo (e, numa variante, a velocidade da bola);
ajustam-se por máxima verosimilhança de Bernoulli. A validação é entre jogos (ajusta num, testa no outro).

Os dados de cada passe (posições, velocidades e orientações dos defesas, pontos da linha) guardam-se em arrays (N × n × P) para
cada avaliação de θ ser uma conta vetorizada, sem voltar ao tracking.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import NamedTuple

import modelo_escolha as me
import numpy as np
import pandas as pd
from escolha import PASTA_SAIDA
from metrica.chegada import probabilidade_de_chegar_ao_ponto
from metrica.linhas import probabilidade_de_bloqueio
from metrica.parametros import PADRAO, Parametros
from metrica.passe import pontos_das_linhas
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.sombra import dentro_do_cone
from metrica.velocidade import velocidade_dos_jogadores
from scipy.optimize import minimize

from footy.ingest import skillcorner as sc

N_DEFESAS = 11  # 10 jogadores de campo + guarda-redes
EPS = 1e-4  # p fica em [EPS, 1 − EPS] na verosimilhança, para um passe falhado com p = 1 não dar −∞
OBSERVADOS = {"successful": 1, "unsuccessful": 0}  # "offside" fica de fora

# θ ajustado: nome do campo de `Parametros`, valor inicial (literatura), limites
CAMPOS = {"tempo_reacao": (0.7, 0.05, 2.0), "velocidade_defesa": (5.0, 2.0, 9.0), "incerteza_tempo": (0.45, 0.05, 2.0),
          "bloqueio_nulo": (1.2, 0.35, 3.0)}
CAMPO_BOLA = {"velocidade_bola": (15.0, 8.0, 30.0)}


class DadosDosPasses(NamedTuple):
    """Arrays por passe (N), defesa (n) e ponto da linha (P). `no_cone`: o ponto está no cone do defesa (só se projeta sombra)."""
    jogo: np.ndarray
    posse_id: np.ndarray
    chegou: np.ndarray  # (N,) 0/1
    comprimento: np.ndarray
    x: np.ndarray  # (N × n) defesas; NaN-preenchidos se faltar defesa
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    existe: np.ndarray  # (N × n) bool
    px: np.ndarray  # (N × P)
    py: np.ndarray
    origem: np.ndarray  # (N × 2)
    no_cone: np.ndarray  # (N × n × P) bool
    dist_corpo: np.ndarray  # (N × n × P) distância do defesa a cada ponto
    dist_bola: np.ndarray  # (N × P) distância da origem a cada ponto
    sombra: np.ndarray  # (N × n) bool: o defesa projeta sombra


def passes_do_modelo(limite: float = me.LIMITE_COMPRIMENTO) -> pd.DataFrame:
    """Um passe por linha: jogo, posse, frame, recetor visado, posições, `chegou` (1/0) — só os passes do modelo da D37, sem foras de jogo."""
    opcoes = pd.read_parquet(PASTA_SAIDA / "escolhas_v3.parquet")
    escolhidas = me.preparar_dados(opcoes, limite)
    escolhidas = escolhidas[escolhidas["escolhida"] == 1]
    t = escolhidas[escolhidas["pass_outcome"].isin(OBSERVADOS)].copy()
    t["chegou"] = t["pass_outcome"].map(OBSERVADOS)
    return t[["jogo", "posse_id", "frame", "recetor", "x_passador", "y_passador", "x_recetor", "y_recetor", "comprimento",
              "chegou", "pass_outcome"]].reset_index(drop=True)


def defesas_do_jogo(jogo: int, frames, p: Parametros = PADRAO) -> pd.DataFrame:
    """Todos os defesas dos `frames` (guarda-redes incluído): x, y, vx, vy, orientacao, `sombra` (tem pose e não marca)."""
    match, tabela = sc.ler_match(jogo), carregar_tabela(jogo)
    d = preparar_defesas(tabela, match, p)
    d = d[d["frame"].isin(frames)]
    d = d.assign(sombra=d["orientacao"].notna() & ~d["a_marcar"])[["frame", "jogador", "x", "y", "vx", "vy", "orientacao", "sombra"]]
    gr = tabela[tabela["defende"] & (tabela["posicao"] == "GK") & tabela["frame"].isin(frames)][["frame", "jogador", "x", "y"]]
    gr = gr.merge(velocidade_dos_jogadores(tabela, p.velocidade_maxima_tracking), on=["frame", "jogador"], how="left")
    gr = gr.assign(orientacao=np.nan, sombra=False)
    return pd.concat([d, gr], ignore_index=True)


def montar_dados(passes: pd.DataFrame, p: Parametros = PADRAO) -> DadosDosPasses:
    """Prepara os arrays de cada passe: os defesas do frame, os pontos da linha escolhida e, para cada defesa, que pontos estão no cone."""
    n = len(passes)
    arr = {k: np.full((n, N_DEFESAS), np.nan) for k in ("x", "y", "vx", "vy", "orientacao")}
    sombra = np.zeros((n, N_DEFESAS), bool)
    for jogo, g in passes.groupby("jogo"):
        por_frame = dict(tuple(defesas_do_jogo(jogo, set(g["frame"]), p).groupby("frame")))
        for i, frame in zip(g.index, g["frame"]):
            d = por_frame[frame].iloc[:N_DEFESAS]
            k = len(d)
            for c, valores in arr.items():
                valores[i, :k] = d[c].to_numpy(float)
            sombra[i, :k] = d["sombra"].to_numpy(bool)
    existe = ~np.isnan(arr["x"])
    for c in ("vx", "vy"):
        arr[c] = np.nan_to_num(arr[c])
    origem = passes[["x_passador", "y_passador"]].to_numpy(float)
    px = np.empty((n, p.pontos_por_linha))
    py = np.empty_like(px)
    for i in range(n):
        a, b = pontos_das_linhas(origem[i], passes[["x_recetor", "y_recetor"]].to_numpy(float)[i:i + 1], p.pontos_por_linha)
        px[i], py[i] = a[0], b[0]
    X, Y = np.nan_to_num(arr["x"], nan=1e6), np.nan_to_num(arr["y"], nan=1e6)
    cone = dentro_do_cone(px[:, None, :], py[:, None, :], X[:, :, None], Y[:, :, None], arr["orientacao"][:, :, None], p)
    cone &= sombra[:, :, None]
    return DadosDosPasses(
        jogo=passes["jogo"].to_numpy(), posse_id=passes["posse_id"].to_numpy(), chegou=passes["chegou"].to_numpy(int),
        comprimento=passes["comprimento"].to_numpy(float), x=X, y=Y, vx=arr["vx"], vy=arr["vy"], existe=existe, px=px, py=py,
        origem=origem, no_cone=cone, dist_corpo=np.hypot(px[:, None, :] - X[:, :, None], py[:, None, :] - Y[:, :, None]),
        dist_bola=np.hypot(px - origem[:, 0:1], py - origem[:, 1:2]), sombra=sombra)


def subconjunto(d: DadosDosPasses, mascara: np.ndarray) -> DadosDosPasses:
    return DadosDosPasses(*(a[mascara] for a in d))


# ---------------------------------------------------------------- o modelo sobre os arrays

def parametros_de(theta: np.ndarray, nomes: list[str], base: Parametros = PADRAO) -> Parametros:
    return replace(base, **dict(zip(nomes, map(float, theta))))


def probabilidades(d: DadosDosPasses, p: Parametros, bloqueio: bool = True) -> dict:
    """(N,) `sombras` (Π_d 1 − p_cone), `fundo` (Π_d 1 − p_fora) e `completo` (produto) da linha escolhida de cada passe."""
    margem_tempo = d.dist_bola[:, None, :] / p.velocidade_bola - (
        p.tempo_reacao + np.hypot(d.px[:, None, :] - (d.x + d.vx * p.tempo_reacao)[:, :, None],
                                  d.py[:, None, :] - (d.y + d.vy * p.tempo_reacao)[:, :, None]) / p.velocidade_defesa)
    chega = probabilidade_de_chegar_ao_ponto(margem_tempo, p.incerteza_tempo)
    chega = np.where(d.existe[:, :, None], chega, 0.0)
    cone = d.no_cone
    p_chegada_cone = np.where(cone, chega, 0.0).max(axis=2)
    p_bloqueio = np.where(cone, probabilidade_de_bloqueio(d.dist_corpo, p), 0.0).max(axis=2) if bloqueio else 0.0
    p_cone = np.where(cone.any(axis=2), np.maximum(p_bloqueio, p_chegada_cone), 0.0)
    p_fora = np.where(~cone, chega, 0.0).max(axis=2)
    sombras, fundo = np.prod(1 - p_cone, axis=1), np.prod(1 - p_fora, axis=1)
    return {"sombras": sombras, "fundo": fundo, "completo": sombras * fundo}


def log_verosimilhanca(chegou: np.ndarray, prob: np.ndarray) -> float:
    q = np.clip(prob, EPS, 1 - EPS)
    return float(np.sum(chegou * np.log(q) + (1 - chegou) * np.log(1 - q)))


def metricas(chegou: np.ndarray, prob: np.ndarray) -> dict:
    """Log-loss e Brier por passe."""
    q = np.clip(prob, EPS, 1 - EPS)
    return {"n": len(chegou), "log_loss": float(-np.mean(chegou * np.log(q) + (1 - chegou) * np.log(1 - q))),
            "brier": float(np.mean((prob - chegou) ** 2))}


def curva_de_calibracao(chegou: np.ndarray, prob: np.ndarray, grupos: int = 10) -> pd.DataFrame:
    """Por decis da probabilidade prevista (grupos de igual dimensão): previsto médio vs observado."""
    t = pd.DataFrame({"p": prob, "chegou": chegou})
    t["decil"] = pd.qcut(t["p"].rank(method="first"), grupos, labels=False)
    g = t.groupby("decil").agg(n=("p", "size"), previsto=("p", "mean"), observado=("chegou", "mean"))
    return g.reset_index()


# ---------------------------------------------------------------- ajuste

class Ajuste(NamedTuple):
    nomes: list
    theta: np.ndarray
    erro_padrao: np.ndarray
    log_verosimilhanca: float
    n: int


def ajustar(d: DadosDosPasses, nomes: list[str] | None = None, modelo: str = "completo", bloqueio: bool = True,
            base: Parametros = PADRAO) -> Ajuste:
    """Máxima verosimilhança de θ (os `nomes` de campos de `Parametros`) com Nelder-Mead a partir de vários pontos iniciais. O erro-padrão
    vem do Hessiano numérico (NaN onde não se inverte: parâmetro mal identificado)."""
    nomes = nomes or list(CAMPOS)
    limites = {**CAMPOS, **CAMPO_BOLA}
    inicio = np.array([limites[n][0] for n in nomes])
    baixo, alto = np.array([limites[n][1] for n in nomes]), np.array([limites[n][2] for n in nomes])

    def custo(theta):
        if np.any(theta < baixo) or np.any(theta > alto):
            return 1e9
        return -log_verosimilhanca(d.chegou, probabilidades(d, parametros_de(theta, nomes, base), bloqueio)[modelo])

    melhor = None
    rng = np.random.default_rng(0)
    for k in range(3):
        x0 = inicio if k == 0 else np.clip(inicio * rng.uniform(0.7, 1.4, len(nomes)), baixo + 1e-3, alto - 1e-3)
        r = minimize(custo, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 600})
        if melhor is None or r.fun < melhor.fun:
            melhor = r
    return Ajuste(nomes, melhor.x, erro_padrao(custo, melhor.x, baixo, alto), float(-melhor.fun), len(d.chegou))


def erro_padrao(custo, theta: np.ndarray, baixo, alto) -> np.ndarray:
    """Erro-padrão pelo Hessiano numérico (diferenças finitas, passo relativo de 3 %)."""
    k = len(theta)
    h = np.maximum(0.03 * np.abs(theta), 1e-3)
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h[i], np.eye(k)[j] * h[j]
            hess[i, j] = hess[j, i] = (custo(theta + ei + ej) - custo(theta + ei - ej) - custo(theta - ei + ej)
                                       + custo(theta - ei - ej)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(hess)
        dp = np.sqrt(np.diag(cov))
        return np.where(np.diag(cov) > 0, dp, np.nan)
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


def validacao_entre_jogos(d: DadosDosPasses, nomes: list[str] | None = None) -> dict:
    """Ajusta num jogo e testa no outro (as duas direções): por direção, o ajuste e as métricas fora da amostra do modelo
    atual (só sombras, literatura), do completo com a literatura e do completo calibrado, mais a taxa base."""
    jogos = sorted(set(d.jogo))
    saida = {}
    for treino in jogos:
        teste = next(j for j in jogos if j != treino)
        dt, de = subconjunto(d, d.jogo == treino), subconjunto(d, d.jogo == teste)
        a = ajustar(dt, nomes)
        calibrado = probabilidades(de, parametros_de(a.theta, a.nomes))["completo"]
        lit = probabilidades(de, PADRAO)
        base = np.full(len(de.chegou), dt.chegou.mean())
        saida[f"{treino}->{teste}"] = {
            "ajuste": a, "teste": teste, "n_teste": len(de.chegou),
            "metricas": {"so_sombras_literatura": metricas(de.chegou, lit["sombras"]),
                         "completo_literatura": metricas(de.chegou, lit["completo"]),
                         "completo_calibrado": metricas(de.chegou, calibrado), "taxa_base_do_treino": metricas(de.chegou, base)},
            "curvas": {"so_sombras_literatura": curva_de_calibracao(de.chegou, lit["sombras"]),
                       "completo_literatura": curva_de_calibracao(de.chegou, lit["completo"]),
                       "completo_calibrado": curva_de_calibracao(de.chegou, calibrado)},
            "previsto": {"so_sombras_literatura": lit["sombras"], "completo_literatura": lit["completo"], "completo_calibrado": calibrado},
            "indices_teste": np.where(d.jogo == teste)[0]}
    return saida


def taxa_com_p_baixo(chegou: np.ndarray, prob: np.ndarray, limiar: float = 0.5) -> dict:
    """C3: passes com p < limiar e % deles que chegou."""
    m = prob < limiar
    return {"passes": int(m.sum()), "chegaram": int(chegou[m].sum()),
            "pct_chegaram": float(100 * chegou[m].mean()) if m.any() else float("nan")}


def _ajuste_json(a: Ajuste) -> dict:
    return {"parametros": {n: float(v) for n, v in zip(a.nomes, a.theta)},
            "erro_padrao": {n: (None if np.isnan(e) else float(e)) for n, e in zip(a.nomes, a.erro_padrao)},
            "log_verosimilhanca": a.log_verosimilhanca, "n": a.n,
            "no_limite": [n for n, v in zip(a.nomes, a.theta) if v <= {**CAMPOS, **CAMPO_BOLA}[n][1] + 1e-2
                          or v >= {**CAMPOS, **CAMPO_BOLA}[n][2] - 1e-2]}


def calibrar_tudo() -> dict:
    """Ajustes (jogos juntos, e entre jogos), variante com a velocidade da bola, métricas, curvas e C3. Devolve o dicionário que
    vai para `outputs/cover_shadow/calibracao_passe.json` e a tabela de previsões por passe."""
    inicio = time.time()
    passes = passes_do_modelo()
    d = montar_dados(passes)
    nomes = list(CAMPOS)
    literatura = {n: float(getattr(PADRAO, n)) for n in (*CAMPOS, *CAMPO_BOLA)}
    conjunto = ajustar(d, nomes)
    com_bola = ajustar(d, nomes + list(CAMPO_BOLA))
    p_cal = parametros_de(conjunto.theta, conjunto.nomes)
    p_bola = parametros_de(com_bola.theta, com_bola.nomes)
    lit = probabilidades(d, PADRAO)
    previsoes = pd.DataFrame({"jogo": d.jogo, "posse_id": d.posse_id, "chegou": d.chegou, "comprimento": d.comprimento,
                              "p_sombras_literatura": lit["sombras"], "p_completo_literatura": lit["completo"],
                              "p_completo_calibrado": probabilidades(d, p_cal)["completo"],
                              "p_completo_calibrado_com_bola": probabilidades(d, p_bola)["completo"]})
    previsoes["p_fundo_calibrado"] = probabilidades(d, p_cal)["fundo"]
    modelos = ["p_sombras_literatura", "p_completo_literatura", "p_completo_calibrado", "p_completo_calibrado_com_bola"]
    nao_amostra = {}
    for treino in sorted(set(d.jogo)):
        teste = next(j for j in sorted(set(d.jogo)) if j != treino)
        dt, de = subconjunto(d, d.jogo == treino), subconjunto(d, d.jogo == teste)
        a4, a5 = ajustar(dt, nomes), ajustar(dt, nomes + list(CAMPO_BOLA))
        prev = {"p_sombras_literatura": probabilidades(de, PADRAO)["sombras"],
                "p_completo_literatura": probabilidades(de, PADRAO)["completo"],
                "p_completo_calibrado": probabilidades(de, parametros_de(a4.theta, a4.nomes))["completo"],
                "p_completo_calibrado_com_bola": probabilidades(de, parametros_de(a5.theta, a5.nomes))["completo"],
                "taxa_base_do_treino": np.full(len(de.chegou), dt.chegou.mean())}
        nao_amostra[f"ajuste_{treino}_teste_{teste}"] = {
            "treino": int(treino), "teste": int(teste), "n_teste": len(de.chegou),
            "ajuste": _ajuste_json(a4), "ajuste_com_bola": _ajuste_json(a5),
            "metricas": {m: metricas(de.chegou, q) for m, q in prev.items()},
            "c3_p_menor_que_0_5": {m: taxa_com_p_baixo(de.chegou, q) for m, q in prev.items() if m != "taxa_base_do_treino"},
            "curvas": {m: curva_de_calibracao(de.chegou, q).to_dict("records") for m, q in prev.items() if m != "taxa_base_do_treino"}}
        previsoes.loc[d.jogo == teste, "fora_da_amostra_calibrado"] = prev["p_completo_calibrado"]
        previsoes.loc[d.jogo == teste, "fora_da_amostra_calibrado_com_bola"] = prev["p_completo_calibrado_com_bola"]
    chegou = previsoes["chegou"].to_numpy()
    resumo = {"n_passes": len(chegou), "passes_por_jogo": {int(j): int(n) for j, n in previsoes.groupby("jogo").size().items()},
              "pct_chegaram": float(100 * chegou.mean()),
              "parametros_literatura": literatura, "ajuste_jogos_juntos": _ajuste_json(conjunto),
              "ajuste_jogos_juntos_com_bola": _ajuste_json(com_bola),
              "parametros_calibrados": {n: float(v) for n, v in zip(conjunto.nomes, conjunto.theta)},
              "metricas_na_amostra": {m: metricas(chegou, previsoes[m].to_numpy()) for m in modelos},
              "metricas_fora_da_amostra_dos_dois_jogos": {
                  m: metricas(chegou, previsoes[c].to_numpy()) for m, c in
                  (("p_completo_calibrado", "fora_da_amostra_calibrado"), ("p_completo_calibrado_com_bola", "fora_da_amostra_calibrado_com_bola"))},
              "c3_p_menor_que_0_5_na_amostra": {m: taxa_com_p_baixo(chegou, previsoes[m].to_numpy()) for m in modelos},
              "c3_p_menor_que_0_5_fora_da_amostra": taxa_com_p_baixo(chegou, previsoes["fora_da_amostra_calibrado"].to_numpy()),
              "curvas_na_amostra": {m: curva_de_calibracao(chegou, previsoes[m].to_numpy()).to_dict("records") for m in modelos},
              "fora_da_amostra": nao_amostra, "tempo_s": time.time() - inicio}
    return {"json": resumo, "previsoes": previsoes}


def main() -> None:
    import json

    from footy.paths import output_dir

    r = calibrar_tudo()
    pasta = output_dir("cover_shadow")
    with open(pasta / "calibracao_passe.json", "w", encoding="utf-8") as f:
        json.dump(r["json"], f, ensure_ascii=False, indent=1, default=float)
    r["previsoes"].to_parquet(PASTA_SAIDA / "calibracao_passe_previsoes.parquet", index=False)
    print(f"{r['json']['tempo_s']:.0f} s; parâmetros:", r["json"]["parametros_calibrados"])


if __name__ == "__main__":
    main()


# Em linguagem corrente: para cada passe que as equipas fizeram, guarda-se a posição de todos os defesas no instante da escolha e
# os pontos da linha para o recetor. Com isso, a probabilidade de o passe chegar é uma conta que se refaz depressa para
# qualquer escolha de parâmetros (tempo de reação, velocidade, incerteza, tamanho do bloqueio). Procura-se os parâmetros que
# tornam mais prováveis os resultados observados (chegou / não chegou) e testa-se se, ajustados num jogo, acertam no outro.
