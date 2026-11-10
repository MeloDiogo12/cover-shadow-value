"""Fase 9a · v3.1 e v4 com o xPass calibrado + peso da sombra γ, nas três moedas (D42).

Pressupõe a tabela da fase 3, a tabela de escolhas e `xpass.py preparar` (variáveis dos 20 jogos). Da raiz do projeto:
    uv run python projects/cover_shadow/calcular_xpass.py --jogo 1925299
    uv run python projects/cover_shadow/calcular_xpass.py --jogo 1996435

Por momento e linha l: `p_l(C) = σ(base_l + γ · sombra_l(C))`, `p_l(∅) = σ(base_l)`, com a base = logit do xPass (ajustado nos outros 19 jogos) e
`sombra_l(C) = 1 − Π_{d∈C}(1 − p_cone_{l,d})` (cone da v3, parâmetros da literatura). O contrafactual «sem sombra» é sombra a 0.
- **v3.1 xPass:** `negada = Σ_l (a_l + b_l)(p_l(∅) − p_l(C))`, Shapley exato por linha entre os defesas que a sombreiam (a soma sobre as linhas dá o Shapley do total):
  `v3_1_xpass/<jogo>_v3_1_<moeda>_{linhas,atribuicoes,momentos}`;
- **v4 xPass:** escolha M2 (cross-fitting, sombras da literatura), I/D simétrica, Shapley exato: `v4_xpass_<moeda>/<jogo>_v4_{linhas,momentos,atribuicoes}`.
O ponto de corte da moeda `risco` vem do modelo físico com os parâmetros da literatura (D42).
"""

from __future__ import annotations

import argparse
import time

import modelo_escolha as me
import numpy as np
import pandas as pd
import xpass as xp
from calcular_modelo_passe import arrays_dos_defesas, linhas_sem_pose
from calcular_v4 import ajuste_do_outro_jogo, guarda_redes_por_frame, variaveis_das_linhas
from entradas import exigir
from escolha import PASTA_SAIDA, distancia_ao_adversario_mais_proximo, recetores_marcados
from metrica.linhas import recetores_com_valor
from metrica.parametros import PADRAO
from metrica.passe import (
    MOEDAS,
    adversarios_ultrapassados,
    custo_da_perda,
    ganho_e_custo,
    probabilidades_das_linhas,
    valor_esperado,
)
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.v4 import MAX_DEFESAS, avaliar_momento, p_chegar_por_coligacao, shapley_exato
from metrica.xt import sentido_de_ataque

from footy.ingest import skillcorner as sc

LIMITE_COMPRIMENTO = me.LIMITE_COMPRIMENTO  # D37: 35 m
PASTA_V3_1 = PASTA_SAIDA / "v3_1_xpass"
PASTA_V3_1_LITERATURA = PASTA_SAIDA / "v3_1_literatura"  # v3.1 da D40 com os parâmetros da literatura (referência da 9a)


def pasta_v4(moeda: str):
    return PASTA_SAIDA / f"v4_xpass_{moeda}"


def p_do_xpass(base: np.ndarray, gamma: float, sombra: np.ndarray) -> np.ndarray:
    """`σ(base + γ · sombra)`; `base` (L,), `sombra` (K × L) ou (L,)."""
    return xp.sigmoide(np.asarray(base, float) + gamma * np.asarray(sombra, float))


def shapley_da_linha(escala: float, base: float, gamma: float, p_cone_linha: np.ndarray) -> np.ndarray:
    """Shapley (por defesa que sombreia a linha) de `escala · (σ(base) − σ(base + γ · sombra(S)))`; o valor da coligação vazia é 0."""
    ps = p_chegar_por_coligacao(p_cone_linha[None, :])[:, 0]  # (2^k,)
    valor = escala * (p_do_xpass(base, gamma, 0.0) - p_do_xpass(base, gamma, 1 - ps))
    return shapley_exato(valor)


def avaliar_frame_xpass(jogadores: pd.DataFrame, defesas: pd.DataFrame, sentido: int, campo: tuple, marcados: set, dist_adversario: dict,
                        coef: np.ndarray, coef_xpass: np.ndarray, gamma: float, p_lit=PADRAO,
                        guarda_redes: pd.DataFrame | None = None, limite: float = LIMITE_COMPRIMENTO, moedas=MOEDAS):
    """({moeda: v3.1}, {moeda: v4}) de um frame, ou None se não sobra nenhuma linha; cada um (linhas, momento, atribuições).
    `coef`: escolha M2 (cross-fitting); `coef_xpass`: xPass (outros 19 jogos); `gamma`: peso da sombra."""
    origem, recetores = recetores_com_valor(jogadores, sentido, *campo, p_lit)
    recetores["comprimento"] = np.hypot(recetores["x"] - origem[0], recetores["y"] - origem[1])
    recetores = recetores[~recetores["fora_de_jogo"] & ~recetores["cruzamento"]
                          & (recetores["comprimento"] <= limite)].reset_index(drop=True)
    if recetores.empty:
        return None
    xy = recetores[["x", "y"]].to_numpy(float)
    sem_pose = linhas_sem_pose(origem, xy, defesas, p_lit)
    d = arrays_dos_defesas(defesas, guarda_redes)
    lit = probabilidades_das_linhas(origem, xy, d, p_lit)
    em_c = (lit.p_cone > 0).any(axis=0)
    if em_c.sum() > MAX_DEFESAS:
        raise ValueError("mais de 10 defesas em C")
    jogadores_c = d["jogador"][em_c]
    pc = lit.p_cone[:, em_c]  # (L × n): a v3, parâmetros da literatura
    com_sombra = pc > 0
    base = xp.logit_xpass(coef_xpass, xp.caracteristicas(origem, xy, np.stack([d["x"], d["y"]], axis=1), sentido))
    p0 = p_do_xpass(base, gamma, 0.0)
    p_sombras = p_chegar_por_coligacao(pc)  # (2^n × L)
    p_col = p_do_xpass(base, gamma, 1 - p_sombras)  # (2^n × L): xPass com a sombra de cada coligação
    ultrapassados = adversarios_ultrapassados(origem, xy, defesas["x"].to_numpy(), sentido)
    custo_xt = custo_da_perda(lit.ponto_corte, sentido, *campo)
    primeiro = jogadores.iloc[0]
    frame = primeiro["frame"]
    contexto = {"frame": frame, "posse_id": primeiro["posse_id"], "portador": primeiro["portador"]}
    x = variaveis_das_linhas(origem, recetores, sentido, marcados, dist_adversario, sem_pose)
    eta = x[me.BASE].to_numpy() @ coef[:-1]
    quem = [",".join(str(j) for j in sorted(jogadores_c[linha])) for linha in com_sombra]
    avaliadas = ~sem_pose  # v3.1 mantém as exclusões da v3 (D13)
    v31, v4 = {}, {}
    for moeda in moedas:
        a, b = ganho_e_custo(moeda, x["valor_xt"].to_numpy(), ultrapassados, custo_xt)
        a = np.asarray(a, float)
        b_ = None if b is None else np.asarray(b, float)
        escala = a if b_ is None else a + b_
        pC = p_col[-1]
        negada = escala * (p0 - pC)
        atribs = []
        for l in np.flatnonzero(avaliadas & com_sombra.any(axis=1)):
            for j, s in zip(jogadores_c[com_sombra[l]], shapley_da_linha(escala[l], base[l], gamma, pc[l][com_sombra[l]])):
                atribs.append({"frame": frame, "recetor": recetores["recetor"].iloc[l], "defesa": j, "shapley": s})
        ev0, evc = valor_esperado(p0, a, b_), valor_esperado(pC, a, b_)
        l31 = pd.DataFrame({"recetor": recetores["recetor"].to_numpy(), "comprimento": recetores["comprimento"].to_numpy(),
                            "valor_xt": x["valor_xt"].to_numpy(), "ganho": a, "custo": np.nan if b_ is None else b_,
                            "base": base, "p0": p0, "pC": pC, "n_sombras": com_sombra.sum(axis=1), "quem_sombra": quem,
                            "excluida": ~avaliadas, "negada_linha": np.where(avaliadas, negada, 0.0),
                            "disponivel_linha": np.where(avaliadas, ev0, 0.0)}).assign(**contexto)
        m31 = {**contexto, "n_linhas": int(avaliadas.sum()), "E0": float(ev0[avaliadas].sum()), "EC": float(evc[avaliadas].sum()),
               "v": float(negada[avaliadas].sum()), "n_defesas": int(em_c.sum())}
        v31[moeda] = (l31, m31, pd.DataFrame(atribs, columns=["frame", "recetor", "defesa", "shapley"]))
        m = avaliar_momento(a, eta, coef[-1], pc, custo=b_, p_coligacoes=p_col)
        linhas = x.assign(**contexto, recetor=recetores["recetor"].to_numpy(), s0=m.s0, sC=m.sC, pC=m.pC, p0=m.p0, base=base,
                          ganho=a, custo=np.nan if b_ is None else b_, n_sombras=com_sombra.sum(axis=1), quem_sombra=quem)
        linhas["negada_linha"] = m.s0 * valor_esperado(m.p0, a, b_) - m.sC * valor_esperado(m.pC, a, b_)
        momento = {**contexto, "n_linhas": len(x), "n_linhas_sem_pose": int(sem_pose.sum()), "n_defesas": int(em_c.sum()),
                   "E0": m.E0, "EC": m.EC, "I": m.I, "D": m.D, "v": m.v,
                   **{f"{t}_{o}": getattr(dd, t) for o, dd in m.por_ordem.items() for t in ("I", "D")}}
        atrib4 = pd.DataFrame({"frame": frame, "defesa": jogadores_c, "shapley_I": m.shapley_I, "shapley_D": m.shapley_D,
                               "shapley": m.shapley_I + m.shapley_D,
                               **{f"shapley_{t}_{o}": getattr(dd, f"shapley_{t}") for o, dd in m.por_ordem.items() for t in ("I", "D")},
                               **{f"shapley_{o}": dd.shapley_I + dd.shapley_D for o, dd in m.por_ordem.items()}})
        v4[moeda] = (linhas, momento, atrib4)
    return v31, v4


def calcular_jogo_xpass(jogo: int, coef_xpass: np.ndarray, gamma: float, limite: float = LIMITE_COMPRIMENTO, moedas=MOEDAS,
                        max_frames: int | None = None, progresso: int = 2000) -> dict:
    """{"v3_1": {moeda: {linhas, momentos, atribuicoes}}, "v4": {moeda: {...}}} de um jogo."""
    match, tabela = sc.ler_match(jogo), carregar_tabela(jogo)
    defesas = preparar_defesas(tabela, match, PADRAO)
    campo = sc.dimensoes_campo(match)
    coef = ajuste_do_outro_jogo(jogo, "M2").coef
    mc = recetores_marcados(defesas)
    marcados_por_frame: dict = {}
    for frame, recetor in zip(mc["frame"], mc["recetor"]):
        marcados_por_frame.setdefault(frame, set()).add(recetor)
    dist = distancia_ao_adversario_mais_proximo(tabela[["frame", "jogador", "x", "y", "defende"]])
    dist_por_frame = {f: dict(zip(g["recetor"], g["dist_adversario"])) for f, g in dist.groupby("frame")}
    defesas_por_frame = dict(tuple(defesas.groupby("frame")))
    guardas = guarda_redes_por_frame(tabela, PADRAO)
    saida = {v: {m: ([], [], []) for m in moedas} for v in ("v3_1", "v4")}
    n, inicio = 0, time.time()
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        r = avaliar_frame_xpass(jogadores, defesas_por_frame[frame], sentido, campo, marcados_por_frame.get(frame, set()),
                                dist_por_frame[frame], coef, coef_xpass, gamma, PADRAO, guardas.get(frame), limite, moedas)
        if r is None:
            continue
        for versao, por_moeda in zip(("v3_1", "v4"), r):
            for moeda, (l, m, a) in por_moeda.items():
                saida[versao][moeda][0].append(l), saida[versao][moeda][1].append(m), saida[versao][moeda][2].append(a)
        n += 1
        if progresso and n % progresso == 0:
            print(f"  jogo {jogo}: {n} momentos, {time.time() - inicio:.0f} s", flush=True)
        if max_frames and n >= max_frames:
            break
    return {v: {m: {"linhas": pd.concat(ls, ignore_index=True), "momentos": pd.DataFrame(ms),
                    "atribuicoes": pd.concat(a, ignore_index=True)} for m, (ls, ms, a) in por_moeda.items()}
            for v, por_moeda in saida.items()}


def caminho_v3_1(jogo: int, moeda: str, tabela: str):
    return PASTA_V3_1 / f"{jogo}_v3_1_{moeda}_{tabela}.parquet"


def caminho_v4(jogo: int, moeda: str, tabela: str):
    return pasta_v4(moeda) / f"{jogo}_v4_{tabela}.parquet"


def ler_v3_1(jogo: int, moeda: str) -> dict:
    return {t: pd.read_parquet(exigir(caminho_v3_1(jogo, moeda, t), f"calcular_xpass.py --jogo {jogo}")) for t in ("linhas", "momentos", "atribuicoes")}


def ler_v4(jogo: int, moeda: str) -> dict:
    return {t: pd.read_parquet(exigir(caminho_v4(jogo, moeda, t), f"calcular_xpass.py --jogo {jogo}")) for t in ("linhas", "momentos", "atribuicoes")}


def guardar(resultado: dict, jogo: int) -> None:
    PASTA_V3_1.mkdir(parents=True, exist_ok=True)
    for moeda, tabelas in resultado["v3_1"].items():
        for nome, t in tabelas.items():
            t.to_parquet(caminho_v3_1(jogo, moeda, nome), index=False)
    for moeda, tabelas in resultado["v4"].items():
        pasta_v4(moeda).mkdir(parents=True, exist_ok=True)
        for nome, t in tabelas.items():
            t.to_parquet(caminho_v4(jogo, moeda, nome), index=False)


def calcular_v3_1_literatura(jogo: int) -> None:
    """v3.1 do modelo físico completo (D40) com os parâmetros da literatura, nas três moedas: a referência «v3.1 com a literatura» das tabelas da 9a.
    Só guarda a v3.1 (a v4 desse modelo não entra); pasta própria, sem tocar em `v3_1/`."""
    import calcular_modelo_passe as cmp
    r = cmp.calcular_jogo_completo(jogo, PADRAO)["v3_1"]
    PASTA_V3_1_LITERATURA.mkdir(parents=True, exist_ok=True)
    for moeda, tabelas in r.items():
        for nome, t in tabelas.items():
            t.to_parquet(PASTA_V3_1_LITERATURA / f"{jogo}_v3_1_{moeda}_{nome}.parquet", index=False)


def ler_v3_1_literatura(jogo: int, moeda: str) -> dict:
    corre = f"calcular_xpass.py --literatura --jogo {jogo}"
    return {t: pd.read_parquet(exigir(PASTA_V3_1_LITERATURA / f"{jogo}_v3_1_{moeda}_{t}.parquet", corre)) for t in ("linhas", "momentos", "atribuicoes")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--literatura", action="store_true", help="calcula só a v3.1 do modelo físico com os parâmetros da literatura")
    ap.add_argument("--jogo", type=int, default=sc.JOGO_PADRAO, choices=sc.JOGOS_COM_POSE)
    ap.add_argument("--max-frames", type=int, default=None, help="só os primeiros N momentos, sem guardar (para medir o tempo)")
    args = ap.parse_args()
    inicio = time.time()
    if args.literatura:
        calcular_v3_1_literatura(args.jogo)
        print(f"{time.time() - inicio:.0f} s")
        return
    coefs, gamma = xp.modelo_final()
    print(f"gamma = {gamma:.4f}", flush=True)
    r = calcular_jogo_xpass(args.jogo, coefs[args.jogo], gamma, max_frames=args.max_frames)
    print(f"{time.time() - inicio:.0f} s")
    if args.max_frames is None:
        guardar(r, args.jogo)
    for moeda in MOEDAS:
        print(moeda, "v3.1:", r["v3_1"][moeda]["momentos"][["E0", "EC", "v"]].sum().round(3).to_dict(),
              "| v4:", r["v4"][moeda]["momentos"][["E0", "EC", "I", "D", "v"]].sum().round(3).to_dict())


if __name__ == "__main__":
    main()


# Em linguagem corrente: para cada instante da posse e cada linha de passe calcula-se primeiro a probabilidade «de base» de o passe chegar
# (o xPass, que só olha para a geometria) e depois vê-se quanto essa probabilidade desce por causa das sombras, com o peso γ aprendido
# nos jogos com orientação dos corpos. A diferença é o que a sombra nega (v3.1). Na v4 conta-se também que o portador, ao ver a sombra,
# pode mudar de passe; o que cada defesa fez reparte-se por Shapley, como antes.
