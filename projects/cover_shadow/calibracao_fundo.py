"""Fase 8g · calibração do fundo do modelo do passe com os 20 jogos abertos da SkillCorner (D41).

O fundo (`p_fundo = Π_d (1 − p_fora)`, todos os defesas, sem orientação nem bloqueio pelo corpo) não precisa de pose, por isso
usa-se só tracking (10 fps) e eventos. Para cada passe rasteiro (D37: não pelo ar, ≤ 35 m, sem fora de jogo) olha-se para o frame de
tracking do passe e para a linha do passador ao recetor visado.

Critérios de sucesso (fixos na D41, avaliados por aplicação mecânica):
1. log-loss do modelo < a da taxa base em ≥ 16 dos 20 jogos (dobras de um jogo de fora) e melhoria média positiva;
2. nenhum parâmetro no limite da busca e todos a ±20 % do ajuste com todos os jogos, em todas as dobras;
3. ECE (10 decis) ≤ 0,03 fora da amostra, juntando as 20 dobras.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd
from calibracao_passe import CAMPOS, curva_de_calibracao, erro_padrao, log_verosimilhanca, metricas
from entradas import exigir
from metrica.chegada import probabilidade_de_chegar_ao_ponto
from metrica.parametros import PADRAO, Parametros
from metrica.passe import pontos_das_linhas
from metrica.velocidade import velocidade_dos_jogadores
from scipy.optimize import minimize

from footy.ingest import skillcorner as sc
from footy.paths import DATA_PROCESSED

JOGOS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691, 1996435, 1996436,
         2006229, 2006363, 2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461)
PASTA_DADOS = DATA_PROCESSED / "cover_shadow"
LIMITE_COMPRIMENTO = 35.0  # D37
DISTANCIA_CORTE = 5.0  # m: o secundário «cortado» começa a menos disto da linha
N_DEFESAS = 11
PASSO_VELOCIDADE = 2  # frames de tracking de cada lado (0,2 s), como em `metrica.velocidade`
RESULTADOS = {"successful": 1, "unsuccessful": 0}
LIMITES = {n: (v[1], v[2]) for n, v in CAMPOS.items() if n != "bloqueio_nulo"}
INICIO = {n: CAMPOS[n][0] for n in LIMITES}
NOMES = list(LIMITES)
MIN_JOGOS_A_BATER = 16
TOLERANCIA_PARAMETROS = 0.20
LIMITE_ECE = 0.03


# ---------------------------------------------------------------- passes (eventos)

def passes_rasteiros(eventos: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Passes do jogo com recetor visado e resultado successful/unsuccessful, não pelo ar (`high_pass` ≠ True).
    O limite de 35 m aplica-se depois, com o comprimento medido no tracking. Devolve também o funil de contagens."""
    p = eventos[(eventos["event_type"] == "player_possession") & (eventos["end_type"] == "pass")]
    funil = {"passes": len(p), "fora_de_jogo": int((p["pass_outcome"] == "offside").sum())}
    p = p[p["pass_outcome"].isin(RESULTADOS)]
    funil["com_resultado"] = len(p)
    pelo_ar = p["high_pass"].map(lambda v: v is True or v == "True" or v is np.True_).fillna(False).astype(bool)
    funil["pelo_ar"] = int(pelo_ar.sum())
    p = p[~pelo_ar & p["player_targeted_id"].notna()]
    funil["sem_recetor_visado"] = funil["com_resultado"] - funil["pelo_ar"] - len(p)
    saida = pd.DataFrame({
        "posse_id": p["event_id"].to_numpy(), "periodo": p["period"].to_numpy(), "equipa": p["team_id"].to_numpy(),
        "frame": p["frame_end"].astype(int).to_numpy(), "passador": p["player_id"].astype(int).to_numpy(),
        "recetor": p["player_targeted_id"].astype(int).to_numpy(), "resultado": p["pass_outcome"].to_numpy(),
        "chegou": p["pass_outcome"].map(RESULTADOS).to_numpy(int), "indice": p.index.to_numpy()})
    return saida.reset_index(drop=True), funil


def posse_seguinte(eventos: pd.DataFrame, passes: pd.DataFrame) -> pd.DataFrame:
    """Para cada passe, equipa e ponto de início (x_start, y_start) da posse de jogador seguinte (a seguinte nos eventos, por frame)."""
    posses = eventos[eventos["event_type"] == "player_possession"].sort_values(["frame_start", "index"]).reset_index(drop=True)
    posicao = {e: i for i, e in enumerate(posses["event_id"])}
    seguinte = posses.shift(-1)
    i = passes["posse_id"].map(posicao).to_numpy()
    return pd.DataFrame({"equipa_seguinte": seguinte["team_id"].to_numpy()[i], "x_seguinte": seguinte["x_start"].to_numpy()[i],
                         "y_seguinte": seguinte["y_start"].to_numpy()[i]})


def distancia_ao_segmento(px, py, ax, ay, bx, by) -> tuple[np.ndarray, np.ndarray]:
    """Distância de pontos (px, py) ao segmento a→b e posição da projeção ao longo dele (0 em a, 1 em b; sem limitar)."""
    dx, dy = bx - ax, by - ay
    comprimento2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = ((px - ax) * dx + (py - ay) * dy) / comprimento2
    tc = np.clip(t, 0, 1)
    return np.hypot(px - (ax + tc * dx), py - (ay + tc * dy)), t


def marcar_cortados(passes: pd.DataFrame) -> pd.Series:
    """`cortado` (1/0, NaN se fica de fora): successful → 0; unsuccessful → 1 só se a posse seguinte é de quem defendia e começa a
    menos de `DISTANCIA_CORTE` da linha e antes do recetor; os restantes falhados ficam NaN (fora desta análise)."""
    d, t = distancia_ao_segmento(passes["x_seguinte"], passes["y_seguinte"], passes["x_passador"], passes["y_passador"],
                                 passes["x_recetor"], passes["y_recetor"])
    cortado = (passes["resultado"] == "unsuccessful") & (passes["equipa_seguinte"] != passes["equipa"]) & (d < DISTANCIA_CORTE) & (t < 1)
    saida = pd.Series(np.nan, index=passes.index)
    saida[passes["resultado"] == "successful"] = 0.0
    saida[cortado] = 1.0
    return saida


# ---------------------------------------------------------------- tracking → arrays

def frames_necessarios(frames: set[int]) -> set[int]:
    return {f + k * PASSO_VELOCIDADE for f in frames for k in (-1, 0, 1)}


def ler_tracking(jogo: int, quais: set[int]) -> pd.DataFrame:
    """Posição de todos os jogadores nos frames pedidos (extrapolada quando não detetados). Salta as linhas que não interessam
    sem as descodificar (o ficheiro tem ~100 mil frames)."""
    import json
    linhas = []
    with open(sc.caminho(jogo, "tracking_extrapolated.jsonl"), encoding="utf-8") as f:
        for linha in f:
            n = int(linha[9:linha.index(",")])
            if n not in quais:
                continue
            frame = json.loads(linha)
            for j in frame["player_data"]:
                if j["x"] is not None:
                    linhas.append((n, frame["period"], j["player_id"], j["x"], j["y"]))
    return pd.DataFrame(linhas, columns=["frame", "periodo", "jogador", "x", "y"])


class DadosFundo(NamedTuple):
    """Arrays por passe (N), defesa (n) e ponto da linha (P)."""
    jogo: np.ndarray
    chegou: np.ndarray  # (N,) 0/1 (resultado principal)
    cortado: np.ndarray  # (N,) 0/1/NaN (secundário)
    comprimento: np.ndarray
    dist_defesa_linha: np.ndarray  # (N,) distância do defesa mais próximo ao segmento da linha
    x: np.ndarray  # (N × n)
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    existe: np.ndarray  # (N × n) bool
    px: np.ndarray  # (N × P)
    py: np.ndarray
    dist_bola: np.ndarray  # (N × P)


def tabela_de_passes(jogo: int) -> tuple[pd.DataFrame, dict, dict]:
    """Passes rasteiros do jogo com posições (passador, recetor, defesas) no frame do passe; devolve a tabela, os defesas por passe
    (listas x, y, vx, vy) e o funil de contagens."""
    eventos, match = sc.ler_eventos(jogo), sc.ler_match(jogo)
    passes, funil = passes_rasteiros(eventos)
    passes = passes.join(posse_seguinte(eventos, passes))
    equipa_de = {p["id"]: p["team_id"] for p in match["players"]}
    trk = ler_tracking(jogo, frames_necessarios(set(passes["frame"])))
    v = velocidade_dos_jogadores(trk[["frame", "periodo", "jogador", "x", "y"]].drop_duplicates(["frame", "jogador"]))
    trk = trk.drop_duplicates(["frame", "jogador"]).merge(v, on=["frame", "jogador"], how="left")
    trk["equipa"] = trk["jogador"].map(equipa_de)
    por_frame = dict(tuple(trk.groupby("frame")))
    linhas, defesas = [], {}
    for i, r in passes.iterrows():
        g = por_frame.get(r["frame"])
        if g is None:
            continue
        pos = g.set_index("jogador")
        if r["passador"] not in pos.index or r["recetor"] not in pos.index:
            continue
        d = g[(g["equipa"] != r["equipa"]) & g["equipa"].notna()]
        if len(d) == 0:
            continue
        defesas[i] = d[["x", "y", "vx", "vy"]].to_numpy(float)
        linhas.append({**r.to_dict(), "_i": i, "x_passador": pos.at[r["passador"], "x"], "y_passador": pos.at[r["passador"], "y"],
                       "x_recetor": pos.at[r["recetor"], "x"], "y_recetor": pos.at[r["recetor"], "y"]})
    t = pd.DataFrame(linhas)
    funil["com_posicoes"] = len(t)
    t["comprimento"] = np.hypot(t["x_recetor"] - t["x_passador"], t["y_recetor"] - t["y_passador"])
    funil["mais_de_35_m"] = int((t["comprimento"] > LIMITE_COMPRIMENTO).sum())
    t = t[t["comprimento"] <= LIMITE_COMPRIMENTO].reset_index(drop=True)
    funil["rasteiros"] = len(t)
    return t, defesas, funil


def montar_dados(jogo: int, p: Parametros = PADRAO) -> tuple[DadosFundo, dict]:
    """Arrays do jogo: defesas (x, y, velocidade) do frame do passe e pontos da linha passador → recetor."""
    t, defesas, funil = tabela_de_passes(jogo)
    n = len(t)
    arr = {k: np.full((n, N_DEFESAS), np.nan) for k in ("x", "y", "vx", "vy")}
    perto = np.empty(n)
    for k, i in enumerate(t["_i"]):
        d = defesas[i][:N_DEFESAS]
        for c, nome in enumerate(("x", "y", "vx", "vy")):
            arr[nome][k, : len(d)] = d[:, c]
        r = t.iloc[k]
        perto[k] = distancia_ao_segmento(d[:, 0], d[:, 1], r["x_passador"], r["y_passador"], r["x_recetor"], r["y_recetor"])[0].min()
    existe = ~np.isnan(arr["x"])
    px, py = np.empty((n, p.pontos_por_linha)), np.empty((n, p.pontos_por_linha))
    origem = t[["x_passador", "y_passador"]].to_numpy(float)
    destino = t[["x_recetor", "y_recetor"]].to_numpy(float)
    for k in range(n):
        a, b = pontos_das_linhas(origem[k], destino[k:k + 1], p.pontos_por_linha)
        px[k], py[k] = a[0], b[0]
    cortado = marcar_cortados(t).to_numpy(float)
    dados = DadosFundo(
        jogo=np.full(n, jogo), chegou=t["chegou"].to_numpy(int), cortado=cortado, comprimento=t["comprimento"].to_numpy(float),
        dist_defesa_linha=perto, x=np.nan_to_num(arr["x"], nan=1e6), y=np.nan_to_num(arr["y"], nan=1e6),
        vx=np.nan_to_num(arr["vx"]), vy=np.nan_to_num(arr["vy"]), existe=existe, px=px, py=py,
        dist_bola=np.hypot(px - origem[:, 0:1], py - origem[:, 1:2]))
    funil["taxa_sucesso"] = float(dados.chegou.mean())
    funil["cortados"] = int((cortado == 1).sum())
    funil["secundario"] = int((~np.isnan(cortado)).sum())
    return dados, funil


def juntar(partes: list[DadosFundo]) -> DadosFundo:
    return DadosFundo(*(np.concatenate(c) for c in zip(*partes)))


def guardar_dados(dados: DadosFundo, funis: dict) -> None:
    import json
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PASTA_DADOS / "calibracao_fundo_dados.npz", **dados._asdict())
    with open(PASTA_DADOS / "calibracao_fundo_funis.json", "w", encoding="utf-8") as f:
        json.dump(funis, f, indent=1)


def ler_dados() -> DadosFundo:
    z = np.load(exigir(PASTA_DADOS / "calibracao_fundo_dados.npz", "calibracao_fundo.py preparar"))
    return DadosFundo(**{k: z[k] for k in DadosFundo._fields})


def subconjunto(d: DadosFundo, mascara: np.ndarray) -> DadosFundo:
    return DadosFundo(*(a[mascara] for a in d))


# ---------------------------------------------------------------- o modelo do fundo

def parametros_de(theta, nomes=NOMES, base: Parametros = PADRAO) -> Parametros:
    return replace(base, **dict(zip(nomes, map(float, theta))))


def p_fundo(d: DadosFundo, p: Parametros, bloco: int = 1500) -> np.ndarray:
    """(N,) Π_d (1 − p_fora): probabilidade de nenhum defesa (guarda-redes incluído) chegar à linha. Tempo de interceção de
    Spearman/Shaw sem orientação nem bloqueio pelo corpo; cada defesa conta com o ponto da linha onde chega com mais probabilidade.
    Calcula-se aos blocos de passes para os arrays (N × defesas × pontos) não esgotarem a memória."""
    tr = p.tempo_reacao
    saida = np.empty(len(d.chegou))
    for i in range(0, len(saida), bloco):
        s = slice(i, i + bloco)
        margem = d.dist_bola[s, None, :] / p.velocidade_bola - (
            tr + np.hypot(d.px[s, None, :] - (d.x[s] + d.vx[s] * tr)[:, :, None],
                          d.py[s, None, :] - (d.y[s] + d.vy[s] * tr)[:, :, None]) / p.velocidade_defesa)
        chega = np.where(d.existe[s, :, None], probabilidade_de_chegar_ao_ponto(margem, p.incerteza_tempo), 0.0).max(axis=2)
        saida[s] = np.prod(1 - chega, axis=1)
    return saida


# ---------------------------------------------------------------- ajuste

class Ajuste(NamedTuple):
    theta: np.ndarray
    erro_padrao: np.ndarray
    log_verosimilhanca: float
    n: int


def ajustar(d: DadosFundo, alvo: np.ndarray, pontos_iniciais: int = 3) -> Ajuste:
    """Máxima verosimilhança de (tempo de reação, velocidade do defesa, σ) para `alvo` (1 = chegou / não cortado), Nelder-Mead a
    partir de `pontos_iniciais` pontos. Erro-padrão pelo Hessiano numérico (NaN onde não se inverte)."""
    inicio = np.array([INICIO[n] for n in NOMES])
    baixo, alto = np.array([LIMITES[n][0] for n in NOMES]), np.array([LIMITES[n][1] for n in NOMES])

    def custo(theta):
        if np.any(theta < baixo) or np.any(theta > alto):
            return 1e9
        return -log_verosimilhanca(alvo, p_fundo(d, parametros_de(theta)))

    rng = np.random.default_rng(0)
    melhor = None
    for k in range(pontos_iniciais):
        x0 = inicio if k == 0 else np.clip(inicio * rng.uniform(0.7, 1.4, len(NOMES)), baixo + 1e-3, alto - 1e-3)
        r = minimize(custo, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 600})
        if melhor is None or r.fun < melhor.fun:
            melhor = r
    return Ajuste(melhor.x, erro_padrao(custo, melhor.x, baixo, alto), float(-melhor.fun), len(alvo))


def no_limite(theta: np.ndarray) -> list[str]:
    """Parâmetros a menos de 0,01 de um limite da busca (critério 2)."""
    return [n for n, v in zip(NOMES, theta) if v <= LIMITES[n][0] + 1e-2 or v >= LIMITES[n][1] - 1e-2]


# ---------------------------------------------------------------- referência logística

def caracteristicas_logistico(d: DadosFundo) -> np.ndarray:
    return np.column_stack([np.ones(len(d.chegou)), d.comprimento, d.dist_defesa_linha])


def ajustar_logistico(d: DadosFundo, alvo: np.ndarray) -> np.ndarray:
    """Regressão logística de `alvo` em (comprimento, distância do defesa mais próximo à linha), máxima verosimilhança, sem penalização."""
    x = caracteristicas_logistico(d)
    media, desvio = x[:, 1:].mean(axis=0), x[:, 1:].std(axis=0) + 1e-9
    z = np.column_stack([x[:, 0], (x[:, 1:] - media) / desvio])

    def custo(b):
        eta = z @ b
        return np.sum(np.logaddexp(0, eta) - alvo * eta)

    b = minimize(custo, np.zeros(z.shape[1]), method="BFGS").x
    return np.concatenate([[b[0] - np.sum(b[1:] * media / desvio)], b[1:] / desvio])


def prever_logistico(b: np.ndarray, d: DadosFundo) -> np.ndarray:
    return 1 / (1 + np.exp(-(caracteristicas_logistico(d) @ b)))


# ---------------------------------------------------------------- critérios (D41)

def ece(alvo: np.ndarray, prob: np.ndarray, grupos: int = 10) -> float:
    """Erro de calibração esperado: média ponderada de |observado − previsto| nos decis da probabilidade prevista."""
    c = curva_de_calibracao(alvo, prob, grupos)
    return float(np.sum(c["n"] * np.abs(c["observado"] - c["previsto"])) / c["n"].sum())


def criterios(melhorias: np.ndarray, thetas: np.ndarray, theta_total: np.ndarray, alvo: np.ndarray, prob: np.ndarray) -> dict:
    """Aplicação mecânica dos critérios da D41. `melhorias`: log-loss base − log-loss modelo, por jogo; `thetas`: (dobras × parâmetros)."""
    relativo = np.abs(thetas - theta_total) / np.abs(theta_total)
    c1 = {"jogos_a_bater_a_base": int((melhorias > 0).sum()), "n_jogos": len(melhorias), "melhoria_media": float(melhorias.mean())}
    c1["cumpre"] = bool(c1["jogos_a_bater_a_base"] >= MIN_JOGOS_A_BATER and c1["melhoria_media"] > 0)
    c2 = {"dobras_com_parametro_no_limite": int(sum(bool(no_limite(t)) for t in thetas)),
          "desvio_relativo_maximo": {n: float(relativo[:, k].max()) for k, n in enumerate(NOMES)},
          "dobras_fora_de_20_pct": int((relativo > TOLERANCIA_PARAMETROS).any(axis=1).sum())}
    c2["cumpre"] = bool(c2["dobras_com_parametro_no_limite"] == 0 and c2["dobras_fora_de_20_pct"] == 0)
    c3 = {"ece": ece(alvo, prob)}
    c3["cumpre"] = bool(c3["ece"] <= LIMITE_ECE)
    return {"criterio_1_bate_a_taxa_base": c1, "criterio_2_parametros_estaveis": c2, "criterio_3_calibracao": c3,
            "veredicto": "calibrável" if c1["cumpre"] and c2["cumpre"] and c3["cumpre"] else "não calibrável com estes dados"}


# ---------------------------------------------------------------- 20 dobras

def _dobra(args) -> dict:
    d, alvo, jogo = args
    treino, teste = d.jogo != jogo, d.jogo == jogo
    dt, de, yt, ye = subconjunto(d, treino), subconjunto(d, teste), alvo[treino], alvo[teste]
    a = ajustar(dt, yt)
    previsto = {"modelo": p_fundo(de, parametros_de(a.theta)), "literatura": p_fundo(de, PADRAO),
                "taxa_base": np.full(len(ye), yt.mean()), "logistico": prever_logistico(ajustar_logistico(dt, yt), de)}
    return {"jogo": int(jogo), "n": int(teste.sum()), "taxa": float(ye.mean()), "theta": a.theta,
            "metricas": {m: metricas(ye, q) for m, q in previsto.items()}, "previsto": previsto, "y": ye}


def validar(d: DadosFundo, alvo: np.ndarray, processos: int = 6) -> dict:
    """`d`/`alvo` já sem passes de fora da análise. Ajuste com todos os jogos, 20 dobras de um jogo de fora (em paralelo) e critérios."""
    jogos = sorted(set(d.jogo.tolist()))
    tarefas = [(d, alvo, j) for j in jogos]
    if processos > 1:
        with ProcessPoolExecutor(processos) as ex:
            futuro_total = ex.submit(ajustar, d, alvo)  # corre ao mesmo tempo que as dobras
            dobras = list(ex.map(_dobra, tarefas))
            total = futuro_total.result()
    else:
        total, dobras = ajustar(d, alvo), [_dobra(t) for t in tarefas]
    modelos = ["modelo", "literatura", "taxa_base", "logistico"]
    por_jogo = pd.DataFrame([{"jogo": f["jogo"], "passes": f["n"], "taxa_sucesso": f["taxa"],
                              **{f"log_loss_{m}": f["metricas"][m]["log_loss"] for m in modelos}} for f in dobras])
    thetas = np.array([f["theta"] for f in dobras])
    melhorias = (por_jogo["log_loss_taxa_base"] - por_jogo["log_loss_modelo"]).to_numpy()
    y = np.concatenate([f["y"] for f in dobras])
    previsto = {m: np.concatenate([f["previsto"][m] for f in dobras]) for m in modelos}
    return {
        "n_passes": len(alvo), "taxa_sucesso": float(alvo.mean()),
        "ajuste_total": {"parametros": dict(zip(NOMES, map(float, total.theta))),
                         "erro_padrao": {n: (None if np.isnan(e) else float(e)) for n, e in zip(NOMES, total.erro_padrao)},
                         "log_verosimilhanca": total.log_verosimilhanca, "no_limite": no_limite(total.theta)},
        "parametros_literatura": {n: float(getattr(PADRAO, n)) for n in NOMES},
        "parametros_por_dobra": {str(f["jogo"]): dict(zip(NOMES, map(float, f["theta"]))) for f in dobras},
        "por_jogo": por_jogo.to_dict("records"),
        "metricas_juntas": {m: {**metricas(y, previsto[m]), "ece": ece(y, previsto[m])} for m in modelos},
        "curva_de_calibracao": curva_de_calibracao(y, previsto["modelo"]).to_dict("records"),
        "criterios": criterios(melhorias, thetas, total.theta, y, previsto["modelo"]),
        "previsoes": pd.DataFrame({"jogo": np.concatenate([[f["jogo"]] * f["n"] for f in dobras]), "alvo": y,
                                   **{f"p_{m}": previsto[m] for m in modelos}}),
    }


def analisar(d: DadosFundo, processos: int = 6) -> dict:
    """Principal (`pass_outcome`) e secundário (`cortado`, descritivo): mesmas dobras e os mesmos critérios, calculados mecanicamente."""
    principal = validar(d, d.chegou.astype(float), processos)
    com_cortado = ~np.isnan(d.cortado)
    secundario = validar(subconjunto(d, com_cortado), 1 - d.cortado[com_cortado], processos)  # alvo 1 = não cortado
    return {"principal": principal, "secundario": secundario}


# ---------------------------------------------------------------- execução

def preparar(jogos=JOGOS, processos: int = 6) -> None:
    """Lê eventos e tracking dos jogos (uma vez) e guarda os arrays em data/processed."""
    inicio = time.time()
    with ProcessPoolExecutor(processos) as ex:
        resultados = list(ex.map(montar_dados, jogos))
    funis = {str(j): f for j, (_, f) in zip(jogos, resultados)}
    guardar_dados(juntar([r[0] for r in resultados]), funis)
    print(f"dados preparados em {time.time() - inicio:.0f} s")


def main(argumentos=None) -> None:
    import argparse
    import json

    from footy.paths import output_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("passo", choices=["preparar", "calibrar"])
    ap.add_argument("--processos", type=int, default=6)
    a = ap.parse_args(argumentos)
    if a.passo == "preparar":
        preparar(processos=a.processos)
        return
    inicio = time.time()
    r = analisar(ler_dados(), a.processos)
    with open(PASTA_DADOS / "calibracao_fundo_funis.json", encoding="utf-8") as f:
        funis = json.load(f)
    for k in ("principal", "secundario"):
        r[k].pop("previsoes").to_parquet(PASTA_DADOS / f"calibracao_20_jogos_previsoes_{k}.parquet", index=False)
    r["funis"] = funis
    r["tempo_s"] = time.time() - inicio
    with open(output_dir("cover_shadow") / "calibracao_20_jogos.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1, default=float)
    print(f"{r['tempo_s']:.0f} s; principal: {r['principal']['criterios']['veredicto']}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: para cada passe rasteiro dos 20 jogos guarda-se onde estavam todos os defesas no instante do passe e a linha
# do passador para o recetor. O modelo do «fundo» diz, só com distâncias e tempos de reação, que probabilidade tem algum defesa de chegar
# a essa linha; procuram-se o tempo de reação, a velocidade e a incerteza que melhor explicam quais passes chegaram. Cada jogo é previsto
# com parâmetros ajustados nos outros 19 e compara-se com a taxa média de sucesso. Os três critérios (bater a taxa média, parâmetros
# estáveis, probabilidades bem calibradas) estavam fixados antes de ver os resultados e aplicam-se à letra.
