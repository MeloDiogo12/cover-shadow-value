"""Fase 9a · critérios K1–K5 (D42) e tabelas do notebook 12, por aplicação mecânica da regra. Só números.

Versões (nomes das colunas): `v3` (parâmetros da literatura), `v4` (a v4 com fundo, D39), `v3_1_lit_<moeda>` (v3.1 do modelo físico da D40 com os
parâmetros da literatura), `v3_1_xp_<moeda>` e `v4_xp_<moeda>` (xPass + sombra, esta fase), com `<moeda>` em `xt`, `linhas`, `risco`.
K1 e K2 não dependem da moeda (vêm de `xpass.py`); K3, K4 e K5 avaliam a `v4_xp_<moeda>`. Cada função recebe tabelas já calculadas.
"""

from __future__ import annotations

import avaliacao as av
import avaliacao_modelo_passe as amp
import calcular_v4 as c4
import calcular_xpass as cx
import numpy as np
import pandas as pd
from metrica import agregacao
from metrica.passe import MOEDAS

TOLERANCIA = 1e-12
LIMITE_V_NEGATIVA, LIMITE_D_NEGATIVA = 10.0, 25.0  # % (K3)
RAZAO_MINIMA_K4 = 0.25
BLOCOS_ALTO_E_MEDIO = ("high_block", "medium_block")
BLOCO_BAIXO = "low_block"
DEFESA_ESPERADA = {"C06": "Jelacic", "C03": "Klimala"}  # K5
VERSOES = ("v3", "v4", *(f"v3_1_lit_{m}" for m in MOEDAS), *(f"v3_1_xp_{m}" for m in MOEDAS), *(f"v4_xp_{m}" for m in MOEDAS))
LANCES = amp.LANCES


def carregar_versoes(jogo: int) -> dict:
    """{versão: {momentos, atribuicoes, linhas}} de um jogo (na v3, `por_lance` com E0 = disponível e v = negada)."""
    linhas3, atrib3 = av.ler_variante(jogo, "base", av.PASTA_AVALIACAO_V3)
    momentos3 = agregacao.por_lance(linhas3).rename(columns={"disponivel": "E0", "negada": "v"})
    momentos3["EC"] = momentos3["E0"] - momentos3["v"]
    out = {"v3": {"momentos": momentos3, "atribuicoes": atrib3, "linhas": linhas3}, "v4": c4.ler_v4(jogo)}
    for m in MOEDAS:
        out[f"v3_1_lit_{m}"] = cx.ler_v3_1_literatura(jogo, m)
        out[f"v3_1_xp_{m}"] = cx.ler_v3_1(jogo, m)
        out[f"v4_xp_{m}"] = cx.ler_v4(jogo, m)
    return out


# ---------------------------------------------------------------- K3

def k3(momentos_por_jogo: dict) -> dict:
    """K3 sobre a v4 de uma moeda. `momentos_por_jogo`: {jogo: momentos (v, D, n_defesas)}. Percentagens com os dois jogos juntos, entre os
    momentos com pelo menos um defesa em C; D total por jogo."""
    todos = pd.concat([m[m["n_defesas"] >= 1] for m in momentos_por_jogo.values()], ignore_index=True)
    pct_v = float(100 * (todos["v"] < -TOLERANCIA).mean())
    pct_d = float(100 * (todos["D"] < -TOLERANCIA).mean())
    d_total = {int(j): float(m["D"].sum()) for j, m in momentos_por_jogo.items()}
    r = {"momentos_com_defesas_em_C": len(todos), "pct_v_negativa": pct_v, "limite_v": LIMITE_V_NEGATIVA,
         "pct_D_negativa": pct_d, "limite_D": LIMITE_D_NEGATIVA, "D_total_por_jogo": d_total}
    r["cumpre"] = bool(pct_v <= LIMITE_V_NEGATIVA and pct_d <= LIMITE_D_NEGATIVA and all(d >= 0 for d in d_total.values()))
    return r


# ---------------------------------------------------------------- K4

def momentos_com_contexto(momentos: pd.DataFrame, contexto: pd.DataFrame) -> pd.DataFrame:
    """Momentos com alguma linha avaliada (`n_linhas`/`n_avaliadas` > 0), com a fase defensiva e o terço de cada um."""
    coluna = "n_avaliadas" if "n_avaliadas" in momentos else "n_linhas"
    m = momentos[momentos[coluna] > 0]
    return m.merge(contexto[["frame", "terco", "fase"]], on="frame", how="left")


def k4(momentos_por_jogo: dict, contextos: dict) -> dict:
    """K4: média de `v` por momento nos blocos alto + médio a dividir pela do bloco baixo (dois jogos juntos)."""
    t = pd.concat([momentos_com_contexto(m, contextos[j]) for j, m in momentos_por_jogo.items()], ignore_index=True)
    alto_medio = t[t["fase"].isin(BLOCOS_ALTO_E_MEDIO)]["v"]
    baixo = t[t["fase"] == BLOCO_BAIXO]["v"]
    razao = float(alto_medio.mean() / baixo.mean()) if len(baixo) and baixo.mean() != 0 else float("nan")
    r = {"media_alto_e_medio": float(alto_medio.mean()), "momentos_alto_e_medio": len(alto_medio),
         "media_baixo": float(baixo.mean()), "momentos_baixo": len(baixo), "razao": razao, "razao_minima": RAZAO_MINIMA_K4}
    r["cumpre"] = bool(np.isfinite(razao) and baixo.mean() > 0 and razao >= RAZAO_MINIMA_K4)
    return r


# ---------------------------------------------------------------- K5

def k5(atribuicoes_por_jogo: dict, nomes_por_jogo: dict) -> dict:
    """K5: em cada lance de `DEFESA_ESPERADA`, o defesa com maior Shapley total (I + D) é o esperado. `atribuicoes_por_jogo`: {jogo: atribuições
    da v4 de uma moeda (frame, defesa, shapley)}; `nomes_por_jogo`: {jogo: {id: nome}}."""
    lances = {}
    for lance, esperado in DEFESA_ESPERADA.items():
        jogo, frame = LANCES[lance]
        a = atribuicoes_por_jogo[jogo]
        s = a[a["frame"] == frame].groupby("defesa")["shapley"].sum()
        s.index = [nomes_por_jogo[jogo].get(j, j) for j in s.index]
        s = s.sort_values(ascending=False)
        lances[lance] = {"esperado": esperado, "maior": s.index[0] if len(s) else None, "shapley": {n: float(v) for n, v in s.items()},
                         "cumpre": bool(len(s) and esperado in str(s.index[0]))}
    return {"lances": lances, "cumpre": all(v["cumpre"] for v in lances.values())}


# ---------------------------------------------------------------- veredicto

def veredicto(k1: dict, k2: dict, por_moeda: dict) -> dict:
    """A v4 é aceite se K1–K5 passarem em pelo menos uma moeda; se passarem em mais do que uma, a escolha é do autor; senão, o artigo segue com a v3.1."""
    passam = [m for m, r in por_moeda.items() if k1["cumpre"] and k2["cumpre"] and r["k3"]["cumpre"] and r["k4"]["cumpre"] and r["k5"]["cumpre"]]
    if not passam:
        decisao = "v4 não aceite: o artigo segue com a v3.1"
    elif len(passam) == 1:
        decisao = f"v4 aceite na moeda {passam[0]}"
    else:
        decisao = f"v4 aceite em mais do que uma moeda ({', '.join(passam)}): a escolha da moeda é do autor"
    return {"moedas_que_passam": passam, "decisao": decisao}


# ---------------------------------------------------------------- tabelas

def por_fase_ou_terco(versoes_por_jogo: dict, contextos: dict, por: str, versoes=VERSOES) -> pd.DataFrame:
    """Por fase defensiva ou terço, por versão: nº de momentos, média de `v` por momento e % negada (soma de v / soma de E0), dois jogos juntos."""
    partes = []
    for jogo, vs in versoes_por_jogo.items():
        for nome in versoes:
            m = momentos_com_contexto(vs[nome]["momentos"], contextos[jogo])
            partes.append(m.groupby(por, dropna=False).agg(momentos=("frame", "size"), negada=("v", "sum"), E0=("E0", "sum")).reset_index()
                          .assign(versao=nome))
    t = pd.concat(partes).groupby([por, "versao"], dropna=False)[["momentos", "negada", "E0"]].sum().reset_index()
    t["media_por_momento"] = t["negada"] / t["momentos"]
    t["pct_negada"] = np.where(t["E0"] > 0, 100 * t["negada"] / t["E0"], np.nan)
    return t


def pivotar(t: pd.DataFrame, por: str, valor: str, versoes=VERSOES) -> pd.DataFrame:
    return t.pivot(index=por, columns="versao", values=valor).reindex(columns=list(versoes))


def totais_por_versao(versoes_por_jogo: dict, versoes=VERSOES) -> pd.DataFrame:
    linhas = [{"jogo": jogo, "versao": nome, **amp.totais(vs[nome]["momentos"])} for jogo, vs in versoes_por_jogo.items() for nome in versoes]
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- tudo junto (notebook 12 e xpass_v4.json)

def tabela_json(t: pd.DataFrame) -> list:
    return t.reset_index().to_dict("records") if t.index.name or isinstance(t.index, pd.MultiIndex) else t.to_dict("records")


def calcular_tudo(bootstrap: int = 2000) -> dict:
    """K1–K5 por moeda, o veredicto e as tabelas (fase, terço, totais, lances, ρ por posição). Não escreve nada em disco."""
    import time

    import xpass as xp

    from footy.ingest import skillcorner as sc
    inicio = time.time()
    jogos = tuple(sc.JOGOS_COM_POSE)
    d = xp.ler_dados()
    r1 = xp.validar_k1(d)
    t = xp.ler_ou_montar_passes_com_pose()
    r2 = xp.validar_k2(t, xp.offsets_por_jogo(t, r1["coef_por_jogo"]), bootstrap)
    versoes_por_jogo = {j: carregar_versoes(j) for j in jogos}
    contextos = {j: amp.contexto_do_jogo(j) for j in jogos}
    presencas = {j: amp.presenca_do_jogo(j) for j in jogos}
    nomes = {j: presencas[j][2] for j in jogos}
    por_moeda = {}
    for m in MOEDAS:
        nome = f"v4_xp_{m}"
        por_moeda[m] = {"k3": k3({j: versoes_por_jogo[j][nome]["momentos"] for j in jogos}),
                        "k4": k4({j: versoes_por_jogo[j][nome]["momentos"] for j in jogos}, contextos),
                        "k5": k5({j: versoes_por_jogo[j][nome]["atribuicoes"] for j in jogos}, nomes)}
    k1, k2 = r1["k1"], r2["k2"]
    for m, r in por_moeda.items():
        r["todos_cumprem"] = bool(k1["cumpre"] and k2["cumpre"] and r["k3"]["cumpre"] and r["k4"]["cumpre"] and r["k5"]["cumpre"])
    fase = por_fase_ou_terco(versoes_por_jogo, contextos, "fase")
    terco = por_fase_ou_terco(versoes_por_jogo, contextos, "terco")
    por_versao = {n: pd.concat([amp.por_jogador(j, versoes_por_jogo[j], presencas[j][0], presencas[j][1])[n] for j in jogos], ignore_index=True)
                  for n in VERSOES}
    rho_media, rho_min = amp.matriz_de_rho(por_versao, VERSOES)
    lances = {}
    for nome_lance, (jogo, frame) in LANCES.items():
        tot, defesas = amp.lance(jogo, frame, {n: versoes_por_jogo[jogo][n] for n in VERSOES}, nomes[jogo])
        lances[nome_lance] = {"jogo": jogo, "frame": frame, "totais": tot.round(6).to_dict(), "shapley": defesas.round(6).to_dict()}
    return {
        "criterios": {"K1": k1, "K2": k2, "por_moeda": por_moeda, "veredicto": veredicto(k1, r2["k2"], por_moeda)},
        "xpass": {"nomes": xp.NOMES, "coeficientes_20_jogos": dict(zip(["intercepto", *xp.NOMES], map(float, r1["coef_total"]))),
                  "por_jogo": r1["por_jogo"].to_dict("records"), "metricas_juntas": r1["metricas_juntas"],
                  "curva_de_calibracao": r1["curva"].to_dict("records"),
                  "n_passes": len(d.chegou), "taxa_sucesso": float(d.chegou.mean())},
        "sombra": {k: v for k, v in r2.items() if k != "k2"},
        "tabelas": {"fase_media_por_momento": pivotar(fase, "fase", "media_por_momento").round(6).reset_index().to_dict("records"),
                    "fase_pct_negada": pivotar(fase, "fase", "pct_negada").round(3).reset_index().to_dict("records"),
                    "fase_momentos": pivotar(fase, "fase", "momentos").reset_index().to_dict("records"),
                    "terco_media_por_momento": pivotar(terco, "terco", "media_por_momento").round(6).reset_index().to_dict("records"),
                    "terco_pct_negada": pivotar(terco, "terco", "pct_negada").round(3).reset_index().to_dict("records"),
                    "terco_momentos": pivotar(terco, "terco", "momentos").reset_index().to_dict("records"),
                    "totais": totais_por_versao(versoes_por_jogo).round(6).to_dict("records")},
        "lances": lances,
        "rho_por_posicao": {"media": rho_media.round(3).to_dict(), "minimo": rho_min.round(3).to_dict()},
        "tempo_s": time.time() - inicio}


def main() -> None:
    import json

    from footy.paths import output_dir
    r = calcular_tudo()
    with open(output_dir("cover_shadow") / "xpass_v4.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=1, default=float)
    print(r["criterios"]["veredicto"], f"{r['tempo_s']:.0f} s")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este módulo aplica à letra as regras que ficaram escritas na D42 antes de se calcular qualquer coisa. Para cada
# «moeda» vê-se se a v4 com o xPass cumpre cinco testes: o xPass está calibrado; a sombra, com o peso aprendido num jogo, acerta no outro;
# o efeito de dissuasão não é absurdo; a construção de jogo (bloco alto e médio) conta pelo menos um quarto do que conta o bloco baixo; e nos dois
# lances que o treinador escolheu o defesa certo é o que mais valor tem. Se uma moeda passar tudo, a v4 é aceite; se nenhuma passar, o artigo
# fica com a v3.1. Não há afinação nem variantes depois de ver os números.
