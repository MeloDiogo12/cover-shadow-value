"""Fase 8f · v3.1 e v4 com o modelo completo de p (calibrado) e as três moedas (D40).

Pressupõe a tabela da fase 3 (`preparar_dados.py`), a tabela de escolhas (`escolha.py`) e os parâmetros calibrados
(`calibracao_passe.py`, que escreve `outputs/cover_shadow/calibracao_passe.json`). Da raiz do projeto:
    uv run python projects/cover_shadow/calcular_modelo_passe.py --jogo 1925299
    uv run python projects/cover_shadow/calcular_modelo_passe.py --jogo 1996435

Para cada momento da posse calcula-se uma vez as probabilidades (cone e fora do cone, com os parâmetros calibrados; e o cone
com os da literatura, que alimenta o modelo de escolha como no M2 ajustado) e avaliam-se as três moedas (`xt`, `linhas`, `risco`):
- **v3.1** = v3 (sem fora de jogo, cruzamentos nem linhas sem pose) + limite de 35 m, com `negada = Σ ev(p(∅)) − ev(p(C))`
  (para o xT, `Σ x (p(∅) − p(C))`) repartida por Shapley entre os defesas de C: `v3_1/<jogo>_v3_1_<moeda>_{linhas,atribuicoes,momentos}`;
- **v4 completo** (escolha M2 com cross-fitting, I/D simétrica, Shapley): `v4_completo_<moeda>/<jogo>_v4_{linhas,momentos,atribuicoes}`,
  com as mesmas colunas da v4.
O modelo de escolha não muda: `s_l` é função de `1 − p_sombras` com as sombras da literatura, a variável com que o M2 foi ajustado.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import modelo_escolha as me
import numpy as np
import pandas as pd
from calcular_v4 import ajuste_do_outro_jogo, guarda_redes_por_frame, variaveis_das_linhas
from escolha import PASTA_SAIDA, distancia_ao_adversario_mais_proximo, recetores_marcados
from metrica.chegada import valores_de_shapley
from metrica.linhas import distancia_a_segmento, recetores_com_valor
from metrica.parametros import PADRAO, Parametros
from metrica.passe import (
    MOEDAS,
    adversarios_ultrapassados,
    custo_da_perda,
    ganho_e_custo,
    p_fundo,
    probabilidades_das_linhas,
    valor_esperado,
)
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.v4 import MAX_DEFESAS, avaliar_momento
from metrica.xt import sentido_de_ataque

from footy.ingest import skillcorner as sc
from footy.paths import output_dir

LIMITE_COMPRIMENTO = me.LIMITE_COMPRIMENTO  # D37: 35 m
FICHEIRO_CALIBRACAO = output_dir("cover_shadow") / "calibracao_passe.json"
PASTA_V3_1 = PASTA_SAIDA / "v3_1"


def pasta_v4_completo(moeda: str):
    return PASTA_SAIDA / f"v4_completo_{moeda}"


def parametros_calibrados(ficheiro=FICHEIRO_CALIBRACAO) -> Parametros:
    """`PADRAO` com os parâmetros ajustados (os do ajuste com os dois jogos) lidos do json da calibração."""
    with open(ficheiro, encoding="utf-8") as f:
        ajuste = json.load(f)["parametros_calibrados"]
    return replace(PADRAO, **ajuste)


# ---------------------------------------------------------------- um momento

def arrays_dos_defesas(defesas: pd.DataFrame, guarda_redes: pd.DataFrame | None) -> dict:
    """Todos os defesas do momento (campo + guarda-redes): arrays `x`, `y`, `vx`, `vy`, `orientacao`, `sombra` (tem pose e não
    marca; o guarda-redes nunca) e `jogador`."""
    d = defesas.reset_index(drop=True)
    gr = guarda_redes if guarda_redes is not None and not guarda_redes.empty else d.iloc[:0]
    cat = lambda c, g=None: np.concatenate([d[c].to_numpy(float), (gr[c].to_numpy(float) if g is None else np.full(len(gr), g))])
    return {"x": cat("x"), "y": cat("y"), "vx": np.nan_to_num(cat("vx")), "vy": np.nan_to_num(cat("vy")),
            "orientacao": np.concatenate([d["orientacao"].to_numpy(float), np.full(len(gr), np.nan)]),
            "sombra": np.concatenate([(d["orientacao"].notna() & ~d["a_marcar"]).to_numpy(bool), np.zeros(len(gr), bool)]),
            "jogador": np.concatenate([d["jogador"].to_numpy(), gr["jogador"].to_numpy()]),
            "e_gr": np.concatenate([np.zeros(len(d), bool), np.ones(len(gr), bool)])}


def linhas_sem_pose(origem, recetores_xy: np.ndarray, defesas: pd.DataFrame, p: Parametros) -> np.ndarray:
    """(L,): algum defesa de campo que não marca, a menos de `comprimento_cone + largura/2` da linha, não tem pose (D13, D18)."""
    if defesas.empty:
        return np.zeros(len(recetores_xy), bool)
    raio = p.comprimento_cone + p.largura_junto_ao_corpo / 2
    dx, dy = defesas["x"].to_numpy()[None, :], defesas["y"].to_numpy()[None, :]
    d = distancia_a_segmento(dx, dy, origem[0], origem[1], recetores_xy[:, 0:1], recetores_xy[:, 1:2])
    sem = (defesas["orientacao"].isna() & ~defesas["a_marcar"]).to_numpy()[None, :]
    return ((d < raio) & sem).any(axis=1)


def avaliar_frame_completo(jogadores: pd.DataFrame, defesas: pd.DataFrame, sentido: int, campo: tuple, marcados: set,
                           dist_adversario: dict, coef: np.ndarray, p_cal: Parametros, p_lit: Parametros = PADRAO,
                           guarda_redes: pd.DataFrame | None = None, limite: float = LIMITE_COMPRIMENTO,
                           moedas=MOEDAS):
    """({moeda: v3.1}, {moeda: v4 completo}) de um frame, ou None se não sobra nenhuma linha. Cada v3.1 é
    (linhas, momento, atribuições); cada v4, (linhas, momento, atribuições) como em `calcular_v4.avaliar_frame_v4`."""
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
    cal = probabilidades_das_linhas(origem, xy, d, p_cal)
    em_c = (lit.p_cone > 0).any(axis=0)  # C: projetam sombra nalguma linha
    if em_c.sum() > MAX_DEFESAS:  # não acontece com 10 jogadores de campo
        raise ValueError("mais de 10 defesas em C")
    jogadores_c = d["jogador"][em_c]
    pc_cal, pc_lit = cal.p_cone[:, em_c], lit.p_cone[:, em_c]
    pf = p_fundo(cal.p_fora)
    adversarios_x = defesas["x"].to_numpy()
    ultrapassados = adversarios_ultrapassados(origem, xy, adversarios_x, sentido)
    custo_xt = custo_da_perda(cal.ponto_corte, sentido, *campo)
    primeiro = jogadores.iloc[0]
    frame = primeiro["frame"]
    contexto = {"frame": frame, "posse_id": primeiro["posse_id"], "portador": primeiro["portador"]}
    x = variaveis_das_linhas(origem, recetores, sentido, marcados, dist_adversario, sem_pose)
    eta = x[me.BASE].to_numpy() @ coef[:-1]
    com_sombra = pc_cal > 0
    quem = [",".join(str(j) for j in sorted(jogadores_c[linha])) for linha in com_sombra]
    p_sombras = np.prod(1 - pc_cal, axis=1)
    v31, v4 = {}, {}
    avaliadas = ~sem_pose  # v3.1 mantém as exclusões da v3 (D13): linhas com algum defesa sem pose ficam de fora
    shapley_unitario = {}  # Shapley de 1 xT por linha (linear no valor): reutiliza-se nas três moedas
    for moeda in moedas:
        a, b = ganho_e_custo(moeda, x["valor_xt"].to_numpy(), ultrapassados, custo_xt)
        a = np.asarray(a, float)
        b_ = None if b is None else np.asarray(b, float)
        escala = a if b_ is None else a + b_
        # ---- v3.1: negada = Σ [ev(p(∅)) − ev(p(C))] = Σ (a + b)·p_fundo·(1 − p_sombras)
        negada = escala * pf * (1 - p_sombras)
        atribs = []
        for l in np.flatnonzero(avaliadas & com_sombra.any(axis=1)):
            if l not in shapley_unitario:
                shapley_unitario[l] = valores_de_shapley(1.0, pc_cal[l][com_sombra[l]])
            for j, s in zip(jogadores_c[com_sombra[l]], shapley_unitario[l] * escala[l] * pf[l]):
                atribs.append({"frame": frame, "recetor": recetores["recetor"].iloc[l], "defesa": j, "shapley": s})
        ev0, evc = valor_esperado(pf, a, b_), valor_esperado(pf * p_sombras, a, b_)
        l31 = pd.DataFrame({"recetor": recetores["recetor"].to_numpy(), "comprimento": recetores["comprimento"].to_numpy(),
                            "valor_xt": x["valor_xt"].to_numpy(), "ganho": a, "custo": np.nan if b_ is None else b_,
                            "p_fundo": pf, "p_sombras": p_sombras, "n_sombras": com_sombra.sum(axis=1), "quem_sombra": quem,
                            "excluida": ~avaliadas, "negada_linha": np.where(avaliadas, negada, 0.0),
                            "disponivel_linha": np.where(avaliadas, ev0, 0.0)}).assign(**contexto)
        m31 = {**contexto, "n_linhas": int(avaliadas.sum()), "E0": float(ev0[avaliadas].sum()), "EC": float(evc[avaliadas].sum()),
               "v": float(negada[avaliadas].sum()), "disponivel_sem_fundo": float(a[avaliadas].sum())}
        v31[moeda] = (l31, m31, pd.DataFrame(atribs, columns=["frame", "recetor", "defesa", "shapley"]))
        # ---- v4 completo
        m = avaliar_momento(a, eta, coef[-1], pc_cal, p_fundo=pf, custo=b_, p_defesa_escolha=pc_lit)
        linhas = x.assign(**contexto, recetor=recetores["recetor"].to_numpy(), s0=m.s0, sC=m.sC, pC=m.pC, p_fundo=m.p0,
                          p_sombras=p_sombras, ganho=a, custo=np.nan if b_ is None else b_, n_sombras=com_sombra.sum(axis=1),
                          quem_sombra=quem)
        linhas["negada_linha"] = (m.s0 * valor_esperado(m.p0, a, b_)) - (m.sC * valor_esperado(m.pC, a, b_))
        momento = {**contexto, "n_linhas": len(x), "n_linhas_sem_pose": int(sem_pose.sum()), "n_defesas": int(em_c.sum()),
                   "n_defesas_cortados": 0, "E0": m.E0, "EC": m.EC, "I": m.I, "D": m.D, "v": m.v,
                   **{f"{t}_{o}": getattr(dd, t) for o, dd in m.por_ordem.items() for t in ("I", "D")}}
        atrib4 = pd.DataFrame({"frame": frame, "defesa": jogadores_c, "shapley_I": m.shapley_I, "shapley_D": m.shapley_D,
                               "shapley": m.shapley_I + m.shapley_D,
                               **{f"shapley_{t}_{o}": getattr(dd, f"shapley_{t}") for o, dd in m.por_ordem.items() for t in ("I", "D")},
                               **{f"shapley_{o}": dd.shapley_I + dd.shapley_D for o, dd in m.por_ordem.items()}})
        v4[moeda] = (linhas, momento, atrib4)
    return v31, v4


# ---------------------------------------------------------------- um jogo

def calcular_jogo_completo(jogo: int, p_cal: Parametros, limite: float = LIMITE_COMPRIMENTO, moedas=MOEDAS,
                           max_frames: int | None = None) -> dict:
    """{"v3_1": {moeda: {linhas, momentos, atribuicoes}}, "v4": {moeda: {...}}} de um jogo. `max_frames`: só os primeiros N frames
    (para medir o tempo)."""
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
    n = 0
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        r = avaliar_frame_completo(jogadores, defesas_por_frame[frame], sentido, campo, marcados_por_frame.get(frame, set()),
                                   dist_por_frame[frame], coef, p_cal, PADRAO, guardas.get(frame), limite, moedas)
        if r is None:
            continue
        for versao, por_moeda in zip(("v3_1", "v4"), r):
            for moeda, (l, m, a) in por_moeda.items():
                saida[versao][moeda][0].append(l), saida[versao][moeda][1].append(m), saida[versao][moeda][2].append(a)
        n += 1
        if max_frames and n >= max_frames:
            break
    return {v: {m: {"linhas": pd.concat(ls, ignore_index=True), "momentos": pd.DataFrame(ms),
                    "atribuicoes": pd.concat(a, ignore_index=True)} for m, (ls, ms, a) in por_moeda.items()}
            for v, por_moeda in saida.items()}


def caminho_v3_1(jogo: int, moeda: str, tabela: str):
    return PASTA_V3_1 / f"{jogo}_v3_1_{moeda}_{tabela}.parquet"


def caminho_v4_completo(jogo: int, moeda: str, tabela: str):
    return pasta_v4_completo(moeda) / f"{jogo}_v4_{tabela}.parquet"


def ler_v3_1(jogo: int, moeda: str) -> dict:
    return {t: pd.read_parquet(caminho_v3_1(jogo, moeda, t)) for t in ("linhas", "momentos", "atribuicoes")}


def ler_v4_completo(jogo: int, moeda: str) -> dict:
    return {t: pd.read_parquet(caminho_v4_completo(jogo, moeda, t)) for t in ("linhas", "momentos", "atribuicoes")}


def guardar(resultado: dict, jogo: int) -> None:
    PASTA_V3_1.mkdir(parents=True, exist_ok=True)
    for moeda, tabelas in resultado["v3_1"].items():
        for nome, t in tabelas.items():
            t.to_parquet(caminho_v3_1(jogo, moeda, nome), index=False)
    for moeda, tabelas in resultado["v4"].items():
        pasta_v4_completo(moeda).mkdir(parents=True, exist_ok=True)
        for nome, t in tabelas.items():
            t.to_parquet(caminho_v4_completo(jogo, moeda, nome), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogo", type=int, default=sc.JOGO_PADRAO, choices=sc.JOGOS_COM_POSE)
    ap.add_argument("--max-frames", type=int, default=None, help="só os primeiros N frames, sem guardar (para medir o tempo)")
    args = ap.parse_args()
    inicio = time.time()
    r = calcular_jogo_completo(args.jogo, parametros_calibrados(), max_frames=args.max_frames)
    print(f"{time.time() - inicio:.0f} s")
    if args.max_frames is None:
        guardar(r, args.jogo)
    for moeda in MOEDAS:
        print(moeda, "v3.1:", r["v3_1"][moeda]["momentos"][["E0", "EC", "v"]].sum().round(3).to_dict(),
              "| v4:", r["v4"][moeda]["momentos"][["E0", "EC", "I", "D", "v"]].sum().round(3).to_dict())


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script faz, para cada instante da posse, a conta com o modelo completo de passe. Para cada linha de
# passe calcula a chance de ser cortada por cada defesa, tanto pela zona da sombra como pelo resto da linha, com os parâmetros
# ajustados aos passes reais. Depois avalia as três "moedas" (ganho de xT, adversários ultrapassados, ganho menos o risco de
# perder a bola) na v3.1, que só conta o que a sombra tira, e na v4, que conta também o que muda na escolha do portador.
