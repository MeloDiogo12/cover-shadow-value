"""Fase 9a · xPass base calibrado com os 20 jogos (sem pose) e peso γ da sombra nos 2 jogos com pose (D42).

`p(chegar)` de uma linha passa a ser um logístico aprendido com os passes reais: a base (xPass) usa só o tracking (comprimento,
avanço, defesas junto à linha, pressão sobre recetor e portador); a sombra entra por cima, `logit p = base + γ · sombra`, com
γ ajustado nos 2 jogos com pose (a base fica como *offset*, sem termo independente). Critérios K1 e K2 da D42, fixos antes de calcular:
- K1: ECE ≤ 0,03 (10 decis, previsões das 20 dobras juntas) e log-loss melhor que a taxa base em ≥ 16 dos 20 jogos;
- K2: γ ajustado num dos jogos com pose e testado no outro (e ao contrário): ECE ≤ 0,04 em cada jogo de teste.

As variáveis calculam-se com a mesma função (`caracteristicas`) para os passes reais e para qualquer linha de qualquer momento.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import calibracao_fundo as cf
import calibracao_passe as cp
import modelo_escolha as me
import numpy as np
import pandas as pd
from calibracao_passe import metricas
from entradas import exigir
from escolha import PASTA_SAIDA
from metrica.parametros import PADRAO
from metrica.xt import sentido_de_ataque
from scipy.optimize import minimize

from footy.ingest import skillcorner as sc
from footy.paths import DATA_PROCESSED

PASTA_DADOS = DATA_PROCESSED / "cover_shadow"
NOMES = ["comprimento", "avanco", "dist_linha", "n_perto", "dist_recetor", "dist_portador"]
TETO = 15.0  # m: teto das distâncias (D42); sem defesa dentro do segmento, `dist_linha` vale o teto
RAIO_PERTO = 3.0  # m: «defesa perto da linha»
LONGE = 1e6  # defesa que não existe: tão longe que nunca conta
LIMITE_ECE_K1, MIN_JOGOS_K1 = 0.03, 16
LIMITE_ECE_K2 = 0.04
PROCESSOS = 6


# ---------------------------------------------------------------- variáveis (geometria)

def caracteristicas(origem, destinos: np.ndarray, adversarios: np.ndarray, sentido: int) -> np.ndarray:
    """(L × 6) variáveis de `NOMES` das linhas portador → cada destino. `adversarios`: (n × 2), todos os da equipa que defende
    (guarda-redes incluído; os que não existem com coordenadas `LONGE`). Distâncias com teto `TETO`.
    `dist_linha` e `n_perto` só contam defesas cuja projeção cai dentro do segmento (distância perpendicular)."""
    o = np.asarray(origem, float)
    destinos = np.atleast_2d(np.asarray(destinos, float))
    adv = np.asarray(adversarios, float).reshape(-1, 2)
    d = destinos - o  # (L × 2)
    comprimento = np.hypot(d[:, 0], d[:, 1])
    v = adv[None, :, :] - o  # (1 × n × 2)
    seguro = np.maximum(comprimento, 1e-6)[:, None]
    t = (v[..., 0] * d[:, None, 0] + v[..., 1] * d[:, None, 1]) / seguro**2  # posição da projeção ao longo do segmento
    perpendicular = np.abs(v[..., 0] * d[:, None, 1] - v[..., 1] * d[:, None, 0]) / seguro
    dentro = (t >= 0) & (t <= 1)
    dist_linha = np.minimum(np.where(dentro, perpendicular, TETO).min(axis=1), TETO) if adv.size else np.full(len(d), TETO)
    n_perto = (dentro & (perpendicular < RAIO_PERTO)).sum(axis=1)
    if adv.size:
        dist_recetor = np.minimum(np.hypot(adv[None, :, 0] - destinos[:, None, 0], adv[None, :, 1] - destinos[:, None, 1]).min(axis=1), TETO)
        dist_portador = np.full(len(d), min(float(np.hypot(adv[:, 0] - o[0], adv[:, 1] - o[1]).min()), TETO))
    else:
        dist_recetor = dist_portador = np.full(len(d), TETO)
    return np.column_stack([comprimento, sentido * d[:, 0], dist_linha, n_perto, dist_recetor, dist_portador])


# ---------------------------------------------------------------- logístico sem penalização

def sigmoide(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def ajustar_logistico(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coeficientes `[b0, b1…]` (na escala original) do logístico de `y` em `X`, máxima verosimilhança sem penalização
    (BFGS com gradiente, variáveis padronizadas só durante o ajuste)."""
    media, desvio = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - media) / desvio])

    def custo(b):
        eta = Z @ b
        return float(np.sum(np.logaddexp(0, eta) - y * eta))

    def gradiente(b):
        return Z.T @ (sigmoide(Z @ b) - y)

    b = minimize(custo, np.zeros(Z.shape[1]), jac=gradiente, method="BFGS", options={"gtol": 1e-7, "maxiter": 500}).x
    return np.concatenate([[b[0] - np.sum(b[1:] * media / desvio)], b[1:] / desvio])


def logit_xpass(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Logit da probabilidade de o passe chegar (a base)."""
    return coef[0] + np.asarray(X, float) @ coef[1:]


# ---------------------------------------------------------------- dados dos 20 jogos

class DadosXPass(NamedTuple):
    jogo: np.ndarray
    chegou: np.ndarray
    X: np.ndarray  # (N × 6)


def variaveis_do_jogo(jogo: int) -> tuple[DadosXPass, dict]:
    """Passes rasteiros do jogo (D37, como na 8g) e as variáveis de cada um no frame do passe."""
    t, defesas, funil = cf.tabela_de_passes(jogo)
    match = sc.ler_match(jogo)
    X = np.empty((len(t), len(NOMES)))
    for k, r in t.iterrows():
        sentido = sentido_de_ataque(match, int(r["equipa"]), int(r["periodo"]))
        X[k] = caracteristicas((r["x_passador"], r["y_passador"]), [[r["x_recetor"], r["y_recetor"]]],
                               defesas[r["_i"]][:, :2], sentido)[0]
    return DadosXPass(np.full(len(t), jogo), t["chegou"].to_numpy(int), X), funil


def preparar(jogos=cf.JOGOS, processos: int = PROCESSOS) -> DadosXPass:
    """Calcula as variáveis dos 20 jogos (em paralelo) e guarda-as em data/processed."""
    inicio = time.time()
    with ProcessPoolExecutor(processos) as ex:
        partes = list(ex.map(variaveis_do_jogo, jogos))
    d = DadosXPass(*(np.concatenate(c) for c in zip(*[p[0] for p in partes])))
    np.savez_compressed(PASTA_DADOS / "xpass_dados.npz", **d._asdict())
    print(f"variáveis dos {len(jogos)} jogos: {len(d.chegou)} passes em {time.time() - inicio:.0f} s")
    return d


def ler_dados() -> DadosXPass:
    z = np.load(exigir(PASTA_DADOS / "xpass_dados.npz", "xpass.py preparar"))
    return DadosXPass(**{k: z[k] for k in DadosXPass._fields})


# ---------------------------------------------------------------- K1: validação deixando um jogo de fora

def validar_k1(d: DadosXPass) -> dict:
    """20 dobras: ajusta nos outros 19 jogos e prevê o jogo de fora. Devolve métricas por jogo (modelo vs taxa base), as previsões, os
    coeficientes de cada dobra (a base dos jogos com pose usa a dobra desse jogo) e o veredicto de K1 por aplicação mecânica."""
    jogos = sorted(set(d.jogo.tolist()))
    previsto = np.empty(len(d.chegou))
    coef_por_jogo, linhas = {}, []
    for j in jogos:
        teste = d.jogo == j
        coef = ajustar_logistico(d.X[~teste], d.chegou[~teste])
        coef_por_jogo[int(j)] = coef
        previsto[teste] = sigmoide(logit_xpass(coef, d.X[teste]))
        base = np.full(teste.sum(), d.chegou[~teste].mean())
        linhas.append({"jogo": int(j), "passes": int(teste.sum()), "taxa_sucesso": float(d.chegou[teste].mean()),
                       "log_loss_modelo": metricas(d.chegou[teste], previsto[teste])["log_loss"],
                       "log_loss_taxa_base": metricas(d.chegou[teste], base)["log_loss"]})
    por_jogo = pd.DataFrame(linhas)
    a_bater = int((por_jogo["log_loss_modelo"] < por_jogo["log_loss_taxa_base"]).sum())
    e = cf.ece(d.chegou, previsto)
    ajuste_total = ajustar_logistico(d.X, d.chegou)
    return {"por_jogo": por_jogo, "previsto": previsto, "coef_por_jogo": coef_por_jogo, "coef_total": ajuste_total,
            "metricas_juntas": {**metricas(d.chegou, previsto), "ece": e},
            "curva": cp.curva_de_calibracao(d.chegou, previsto),
            "k1": {"ece": e, "limite_ece": LIMITE_ECE_K1, "jogos_a_bater_a_taxa_base": a_bater, "minimo_de_jogos": MIN_JOGOS_K1,
                   "cumpre": bool(e <= LIMITE_ECE_K1 and a_bater >= MIN_JOGOS_K1)}}


# ---------------------------------------------------------------- passes dos 2 jogos com pose

def passes_com_pose() -> pd.DataFrame:
    """Os passes do modelo da D37 nos 2 jogos com pose (1 408 da 8f): variáveis da base (`NOMES`), `sombra_tronco` (1 − p_sombras da v3,
    literatura), `sombra_bola` (cone virado para a bola, só defesas com pose) e `chegou`."""
    passes = cp.passes_do_modelo()
    d = cp.montar_dados(passes)
    esc = me.preparar_dados(pd.read_parquet(exigir(PASTA_SAIDA / "escolhas_v3.parquet", "escolha.py")))
    esc = esc[esc["escolhida"] == 1][["jogo", "posse_id", "sentido", "sombra_bola_pose", "sombra_tronco_pose"]]
    t = passes.merge(esc, on=["jogo", "posse_id"], how="left", validate="one_to_one")
    adv = np.stack([d.x, d.y], axis=2)  # defesas inexistentes: 1e6
    X = np.vstack([caracteristicas(d.origem[i], t[["x_recetor", "y_recetor"]].to_numpy(float)[i:i + 1], adv[i], int(t["sentido"].iloc[i]))
                   for i in range(len(t))])
    t = pd.concat([t.drop(columns="comprimento"), pd.DataFrame(X, columns=NOMES)], axis=1)  # o comprimento passa a ser o da base
    t["sombra_tronco"] = 1 - cp.probabilidades(d, PADRAO)["sombras"]
    t = t.rename(columns={"sombra_bola_pose": "sombra_bola", "sombra_tronco_pose": "sombra_tronco_escolhas"})
    return t


# ---------------------------------------------------------------- peso da sombra (γ)

class Gamma(NamedTuple):
    gamma: float
    erro_padrao: float
    n: int

    @property
    def ic95(self) -> tuple[float, float]:
        return self.gamma - 1.96 * self.erro_padrao, self.gamma + 1.96 * self.erro_padrao


def ajustar_gamma(offset: np.ndarray, sombra: np.ndarray, chegou: np.ndarray) -> Gamma:
    """γ de `logit p = offset + γ · sombra` por máxima verosimilhança de Bernoulli (Newton, sem termo independente). Erro-padrão pela
    informação de Fisher `Σ p(1 − p) sombra²`."""
    g = 0.0
    for _ in range(100):
        p = sigmoide(offset + g * sombra)
        passo = np.sum((chegou - p) * sombra) / max(np.sum(p * (1 - p) * sombra**2), 1e-12)
        g += passo
        if abs(passo) < 1e-10:
            break
    p = sigmoide(offset + g * sombra)
    return Gamma(float(g), float(1 / np.sqrt(max(np.sum(p * (1 - p) * sombra**2), 1e-12))), len(chegou))


def offsets_por_jogo(t: pd.DataFrame, coef_por_jogo: dict) -> np.ndarray:
    """Base (logit do xPass) de cada passe com pose, com o xPass ajustado nos outros 19 jogos (o do jogo do passe fica de fora)."""
    saida = np.empty(len(t))
    for j, g in t.groupby("jogo"):
        saida[g.index] = logit_xpass(coef_por_jogo[int(j)], g[NOMES].to_numpy(float))
    return saida


def validar_k2(t: pd.DataFrame, offset: np.ndarray, bootstrap: int = 2000) -> dict:
    """Ajusta γ num jogo e testa no outro (as duas direções), com a sombra pelo tronco e pela bola, e sem sombra (γ = 0).
    ECE em 10 decis por jogo de teste; K2 = ECE ≤ 0,04 nos dois. γ com IC 95 % do ajuste com os 2 jogos; diferença de log-loss tronco − bola
    (por passe, nas duas direções juntas) com IC 95 % por bootstrap."""
    jogos = sorted(set(t["jogo"]))
    y = t["chegou"].to_numpy(int)
    previsto = {k: np.empty(len(t)) for k in ("sem_sombra", "tronco", "bola")}
    direcoes = {}
    for treino in jogos:
        teste = next(j for j in jogos if j != treino)
        a, b = (t["jogo"] == treino).to_numpy(), (t["jogo"] == teste).to_numpy()
        g = {k: ajustar_gamma(offset[a], t[f"sombra_{k}"].to_numpy()[a], y[a]) for k in ("tronco", "bola")}
        previsto["sem_sombra"][b] = sigmoide(offset[b])
        for k in ("tronco", "bola"):
            previsto[k][b] = sigmoide(offset[b] + g[k].gamma * t[f"sombra_{k}"].to_numpy()[b])
        direcoes[f"{treino}->{teste}"] = {
            "treino": int(treino), "teste": int(teste), "n_teste": int(b.sum()),
            "gamma_tronco": g["tronco"].gamma, "gamma_bola": g["bola"].gamma,
            "metricas": {k: {**metricas(y[b], previsto[k][b]), "ece": cf.ece(y[b], previsto[k][b])} for k in previsto},
            "curvas": {k: cp.curva_de_calibracao(y[b], previsto[k][b]).to_dict("records") for k in previsto}}
    ambos = {k: ajustar_gamma(offset, t[f"sombra_{k}"].to_numpy(), y) for k in ("tronco", "bola")}
    q = {k: np.clip(v, cp.EPS, 1 - cp.EPS) for k, v in previsto.items()}
    perda = {k: -(y * np.log(v) + (1 - y) * np.log(1 - v)) for k, v in q.items()}
    diferenca = perda["tronco"] - perda["bola"]
    k2 = {"ece_por_jogo_de_teste": {d["teste"]: d["metricas"]["tronco"]["ece"] for d in direcoes.values()}, "limite_ece": LIMITE_ECE_K2}
    k2["cumpre"] = bool(all(e <= LIMITE_ECE_K2 for e in k2["ece_por_jogo_de_teste"].values()))
    corr = t[["sombra_tronco", "sombra_bola", *NOMES]].corr()["sombra_tronco"].drop("sombra_tronco")
    return {"direcoes": direcoes, "k2": k2,
            "gamma_dois_jogos": {k: {"gamma": v.gamma, "erro_padrao": v.erro_padrao, "ic95": list(v.ic95), "n": v.n} for k, v in ambos.items()},
            "tronco_menos_bola_log_loss": {"media": float(diferenca.mean()), **me.intervalo_bootstrap(diferenca, bootstrap)},
            "metricas_juntas": {k: {**metricas(y, v), "ece": cf.ece(y, v)} for k, v in previsto.items()},
            "correlacao_sombra_tronco": corr.to_dict(),
            "matriz_de_correlacao": t[["sombra_tronco", "sombra_bola", *NOMES]].corr().to_dict(),
            "sombra_tronco_vs_escolhas": float(np.abs(t["sombra_tronco"] - t["sombra_tronco_escolhas"]).max())}


FICHEIRO_POSE = PASTA_DADOS / "xpass_passes_pose.parquet"


def ler_ou_montar_passes_com_pose() -> pd.DataFrame:
    """Passes dos jogos com pose com as variáveis e as sombras; calcula-se uma vez e fica em data/processed."""
    if not FICHEIRO_POSE.exists():
        passes_com_pose().to_parquet(FICHEIRO_POSE, index=False)
    return pd.read_parquet(FICHEIRO_POSE)


def modelo_final() -> tuple[dict, float]:
    """({jogo: coeficientes do xPass ajustado nos outros 19}, γ do ajuste com os 2 jogos com pose, sombra pelo tronco): o que a v3.1 e a v4 usam."""
    k1 = validar_k1(ler_dados())
    t = ler_ou_montar_passes_com_pose()
    g = ajustar_gamma(offsets_por_jogo(t, k1["coef_por_jogo"]), t["sombra_tronco"].to_numpy(), t["chegou"].to_numpy(int))
    return k1["coef_por_jogo"], g.gamma


def main(argumentos=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("passo", choices=["preparar"])
    ap.add_argument("--processos", type=int, default=PROCESSOS)
    a = ap.parse_args(argumentos)
    if a.passo == "preparar":
        preparar(processos=a.processos)


if __name__ == "__main__":
    main()


# Em linguagem corrente: em vez de calcular à mão se um passe chega (tempos de reação, velocidades), aprende-se com 14 mil passes
# reais a probabilidade de chegar a partir de coisas simples que se veem no tracking: o comprimento, quanto avança, se há defesas
# junto à linha e quão apertados estão o portador e o recetor. É o «xPass». A sombra de um defesa (o cone atrás do tronco) entra depois,
# como um acrescento a esse xPass com um peso γ que se mede só nos dois jogos com orientação dos corpos. Testa-se sempre num jogo que
# não serviu para ajustar.
