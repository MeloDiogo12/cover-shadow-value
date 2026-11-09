"""Fase 9d · números do artigo na moeda oficial «risco» (D44). Só números, sem leitura.

Reúne no `artigo_v31.json` (secção `risco`, com `"moeda_oficial": "risco"`) o que antes estava em xT: totais por jogo, fase e terço, distribuição do
índice por fase, C3 a C6, W01–W08, momento da decisão, distribuição por comprimento, C06 e C03 por defesa e a sensibilidade às outras moedas.
As secções em xT e a `fase_9c` ficam como estavam (análise de sensibilidade). Pressupõe `numeros_v31.py` e `numeros_9c.py` já corridos
(tabelas da v3.1, custos de perda, `artigo_v31.json`). Da raiz do repositório:
    uv run python projects/cover_shadow/numeros_9d.py
"""

from __future__ import annotations

import argparse
import json
import time

import avaliacao as av
import avaliacao_risco as ar
import avaliacao_v31 as a31
import figuras_agregados as ag
import numeros_9c as n9c
import numeros_v31 as n31
import pandas as pd
from metrica import agregacao
from reprodutibilidade_v4 import comparar_tabelas

from footy.paths import output_dir

SAIDA = output_dir("cover_shadow")
JOGOS = ag.JOGOS
VERSAO = n31.VERSAO
CASO_PRESSAO_ALTA = {"jogo": 1996435, "frame": 22256}  # Goodwin sobre o guarda-redes (9c)


# ---------------------------------------------------------------- tabelas em risco

def risco_da_variante(jogo: int, variante: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabelas da v3.1 de uma variante, na moeda risco. O custo `c` é o da base (parâmetros da literatura): só as sombras variam (D44)."""
    linhas, atribuicoes = av.ler_variante(jogo, variante, av.PASTAS[VERSAO])
    return ar.moeda_risco(linhas, atribuicoes, ar.custos_guardados(jogo))


def jogadores_de(JG: dict, tabelas: dict) -> pd.DataFrame:
    """Ameaça negada por jogador (D17) dos dois jogos; `tabelas`: {jogo: (linhas, atribuições)}."""
    return pd.concat([av.negada_por_jogador(*tabelas[j], JG[j]["defesas"][["frame", "posse_id", "jogador"]], JG[j]["jogadores"]).assign(jogo=j)
                      for j in JOGOS], ignore_index=True)


def lances_de(tabelas: dict) -> pd.DataFrame:
    return pd.concat([agregacao.por_lance(tabelas[j][0]).assign(jogo=j) for j in JOGOS], ignore_index=True)


# ---------------------------------------------------------------- totais, fase e terço

def totais_da_moeda(T: dict) -> dict:
    """Lances (frames com alguma linha avaliada), ameaça disponível (g + c), negada e % negada: cada jogo e os dois juntos."""
    def soma(lances: pd.DataFrame) -> dict:
        a = lances[lances["n_avaliadas"] > 0]
        return {"lances": len(a), "disponivel": float(a["disponivel"].sum()), "negada": float(a["negada"].sum()),
                "pct_negada": float(100 * a["negada"].sum() / a["disponivel"].sum())}
    lances = lances_de({j: T[j]["risco"] for j in JOGOS})
    return {"juntos": soma(lances), **{str(j): soma(lances[lances["jogo"] == j]) for j in JOGOS}}


def fase_e_terco(T: dict, ctx: pd.DataFrame) -> dict:
    """% negada (sobre g + c) e parte da negada total, por fase defensiva e por terço da posse; dois jogos juntos. Os lances sem etiqueta
    ficam numa linha com a etiqueta vazia (contam no total)."""
    lances = n31.lances_por_moeda(T, "risco")
    saida = {}
    for por in ("fase", "terco"):
        g = a31.pct_por_contexto(lances, ctx, por)
        g["pct_da_negada_total"] = 100 * g["negada"] / g["negada"].sum()
        g["pct_dos_lances"] = 100 * g["lances"] / g["lances"].sum()
        saida[por] = n31.registos(g)
    return saida


def distribuicao_do_indice(J: dict) -> dict:
    """Índice por fase na moeda risco: mediana, p90 e máximo entre os jogadores com ≥ 100 momentos, nº de jogadores e os que ficam de fora."""
    indices = ar.indice_por_fase(J["atribuicoes"]["risco"], J["presenca"], J["contexto"], J["jogadores"])
    return {"indices": indices, "distribuicao_por_fase": n31.registos(ar.distribuicao_do_indice(indices)),
            "media_da_fase_negada_por_momento": {f: float(v) for f, v in indices.groupby("fase")["media_da_fase"].first().items()},
            "minimo_de_momentos": ar.MIN_MOMENTOS}


# ---------------------------------------------------------------- C4, C5

def c4_em_risco(JG: dict, T: dict) -> list[dict]:
    """ρ de Spearman por jogador (dentro da posição) e por lance de cada variante face à base, em risco; % negada por jogo."""
    base = jogadores_de(JG, {j: T[j]["risco"] for j in JOGOS})
    base_lances = lances_de({j: T[j]["risco"] for j in JOGOS})
    parametros = {**av.VALORES_ANTIGOS_V3, "raio_bloqueio_corpo": av.RAIOS_BLOQUEIO_V3, **n31.COMPRIMENTOS_C4}
    saida = []
    for parametro, valores in parametros.items():
        for valor in valores:
            nome = av.nome_da_variante(parametro, valor)
            variante = {j: risco_da_variante(j, nome) for j in JOGOS}
            rho = av.spearman_por_posicao(base, jogadores_de(JG, variante))
            saida.append({"parametro": parametro, "valor": valor, "variante": nome, "rho_jogador": rho["rho_media"], "rho_jogador_minimo": rho["rho_min"],
                          "rho_lance": av.spearman_por_lance(base_lances, lances_de(variante)),
                          **{f"pct_{j}": av.pct_negada(variante[j][0]) for j in JOGOS}})
    return saida


def c5_em_risco(JG: dict, T: dict) -> dict:
    """Cone pelo tronco contra cone virado para a bola, em risco: ρ por jogador, % de linhas em que muda quem faz sombra e % negada por jogo."""
    tronco = {j: T[j]["risco"] for j in JOGOS}
    bola = {j: risco_da_variante(j, "bola") for j in JOGOS}
    jogadores = jogadores_de(JG, tronco).merge(jogadores_de(JG, bola), on=["jogo", "jogador", "posicao", "nome"], suffixes=("_tronco", "_bola"))
    rho = av.spearman_por_posicao(jogadores_de(JG, tronco), jogadores_de(JG, bola))
    mudancas = []
    for j in JOGOS:
        base_l, base_a = av.ler_variante(j, "base", av.PASTAS[VERSAO])
        bola_l, bola_a = av.ler_variante(j, "bola", av.PASTAS[VERSAO])
        mudancas.append(av.linhas_que_mudam(base_l, bola_l, base_a, bola_a).assign(jogo=j))
    mudancas = pd.concat(mudancas, ignore_index=True)
    return {"rho_media": rho["rho_media"], "rho_minimo": rho["rho_min"], "pct_mudam": float(100 * mudancas["muda"].mean()), "linhas": len(mudancas),
            "pct_negada_tronco_bola": {str(j): [av.pct_negada(tronco[j][0]), av.pct_negada(bola[j][0])] for j in JOGOS}, "jogadores": len(jogadores),
            "jogadores_tabela": n31.registos(jogadores)}


# ---------------------------------------------------------------- C6, W, comprimento

def c6_em_risco(T: dict) -> dict:
    """Reprodutibilidade da moeda: volta a calcular os custos de perda a partir da tabela da fase 3 e compara com os guardados, e a moeda
    risco com a das tabelas guardadas (a v3.1 em si, a tabela da fase 3 e o resultado dos testes estão em `c6`, fase 9b, e não mudam)."""
    r = {"custos": {}, "moeda": {}}
    for j in JOGOS:
        guardado = ar.custos_guardados(j)
        r["custos"][str(j)] = comparar_tabelas(guardado, ar.custos_do_jogo(j, progresso=False))
        linhas, atribuicoes = av.ler_variante(j, "base", av.PASTAS[VERSAO])
        nova = ar.moeda_risco(linhas, atribuicoes, guardado)
        r["moeda"][str(j)] = {"linhas": comparar_tabelas(T[j]["risco"][0], nova[0]), "atribuicoes": comparar_tabelas(T[j]["risco"][1], nova[1])}
    r["todas_iguais"] = all(v["iguais"] for v in r["custos"].values()) and all(x["iguais"] for v in r["moeda"].values() for x in v.values())
    return r


def lances_w_em_risco(JG: dict, T: dict) -> dict:
    nomes = {j: JG[j]["jogadores"].set_index("jogador")["nome"].to_dict() for j in JOGOS}
    return {lance: {"jogo": j, "frame": f, "risco": a31.negada_por_defesa_no_lance(*T[j]["risco"], f, nomes[j])} for lance, (j, f) in n31.W.items()}


def negada_por_comprimento_em_risco(JG: dict, T: dict) -> dict:
    """Ameaça negada (risco) por banda de comprimento da linha, v3 (sem limite de 35 m) contra v3.1, dois jogos juntos."""
    v3 = []
    for j in JOGOS:
        linhas, atribuicoes = av.ler_variante(j, "base", av.PASTAS["v3"])
        v3.append(av.adicionar_comprimento(ar.moeda_risco(linhas, atribuicoes, ar.custos_guardados(j))[0], JG[j]["tabela"]))
    v31 = pd.concat([T[j]["risco"][0] for j in JOGOS])
    return {"v3": n31.registos(a31.negada_por_comprimento(pd.concat(v3))), "v3_1": n31.registos(a31.negada_por_comprimento(v31))}


# ---------------------------------------------------------------- C06 e C03 por defesa

def lance_por_defesa(JG: dict, T: dict, ctx: pd.DataFrame, indices: pd.DataFrame, jogo: int, frame: int) -> dict:
    """Por defesa com ameaça negada no lance: valor em risco, % do disponível no lance e índice do jogador na fase (e o seu percentil)."""
    nomes = JG[jogo]["jogadores"].set_index("jogador")["nome"].to_dict()
    ids = {n: i for i, n in nomes.items()}
    v = a31.negada_por_defesa_no_lance(*T[jogo]["risco"], frame, nomes)
    fase = ctx[(ctx["jogo"] == jogo) & (ctx["frame"] == frame)]["fase"].iloc[0]
    por_defesa = {}
    for nome, valor in v["shapley"].items():
        linha = indices[(indices["fase"] == fase) & (indices["jogador"] == ids[nome])]
        por_defesa[nome] = {"negada": valor, "pct_do_disponivel": float(100 * valor / v["perigo"]),
                            "momentos_na_fase": int(linha["momentos"].iloc[0]) if len(linha) else None,
                            "indice_na_fase": float(linha["indice"].iloc[0]) if len(linha) else None,
                            "percentil_na_fase": float(linha["percentil"].iloc[0]) if len(linha) else None}
    return {"jogo": jogo, "frame": frame, "fase": fase, "linhas_avaliadas": v["linhas_avaliadas"], "disponivel": v["perigo"], "negada": v["negada"],
            "pct_negada": float(100 * v["negada"] / v["perigo"]), "por_defesa": por_defesa}


def sensibilidade_entre_moedas() -> dict:
    """ρ por jogador (dentro da posição) dentro de cada fase entre risco e xT e entre risco e linhas, e a parte da negada total por fase em
    cada moeda: já calculados na 9c (`fase_9c`), reunidos aqui."""
    f = n31.ler_json("artigo_v31.json")["fase_9c"]["numeros_por_fase"]
    rho = {fase: {par: {k: v for k, v in d.items() if k != "por_posicao"} for par, d in por_par.items() if par in ("xt_risco", "linhas_risco")}
           for fase, por_par in f["rho_por_jogador_dentro_da_posicao"].items()}
    return {"rho_por_jogador_dentro_da_posicao": rho, "parte_da_negada_total_por_fase": f["parte_da_negada_total_por_fase"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so", nargs="*", help="só estes passos (os outros ficam como estão no ficheiro)")
    args = ap.parse_args()
    inicio = time.time()
    JG = ag.carregar_jogos()
    T = n9c.tabelas_das_tres_moedas(JG)
    ctx = n31.contextos(JG)
    J = n9c.juntar_jogos(JG, T, ctx)
    d = distribuicao_do_indice(J)
    indices = d.pop("indices")
    r = {"definicao": {"moeda": "risco (D44)", "negada_l": "(1 − p_chegar_l) · (g_l + c_l); g = ganho de xT, c = xT do adversário no ponto de corte; p(∅) = 1",
                       "disponivel_l": "g_l + c_l", "leitura": "índice por fase: 100 = média por momento dos jogadores nessa fase; percentil dentro da fase; "
                       f"só jogadores com ≥ {ar.MIN_MOMENTOS} momentos na fase", "variantes_c4_c5": "o custo c é o da base; só as sombras variam"}}
    passos = {"totais": lambda: totais_da_moeda(T), "fase_e_terco": lambda: fase_e_terco(T, ctx), "indice_por_fase": lambda: d,
              "c3": lambda: n31.avaliacao_c3(JG), "c4": lambda: c4_em_risco(JG, T), "c5": lambda: c5_em_risco(JG, T),
              "c6_moeda": lambda: c6_em_risco(T), "lances_w01_w08": lambda: lances_w_em_risco(JG, T),
              "momento_da_decisao": lambda: n31.momento_da_decisao(JG, T, ("risco",)),
              "negada_por_comprimento": lambda: negada_por_comprimento_em_risco(JG, T),
              "c06": lambda: lance_por_defesa(JG, T, ctx, indices, **n31.C06), "c03": lambda: lance_por_defesa(JG, T, ctx, indices, **n31.C03),
              "caso_pressao_alta": lambda: lance_por_defesa(JG, T, ctx, indices, **CASO_PRESSAO_ALTA),
              "sensibilidade_entre_moedas": sensibilidade_entre_moedas}
    antigo = n31.ler_json("artigo_v31.json")
    if args.so:
        r = antigo["risco"]
    for nome, f in passos.items():
        if args.so and nome not in args.so:
            continue
        t0 = time.time()
        r[nome] = f()
        print(f"  {nome}: {time.time() - t0:.0f} s", flush=True)
    chaves = [k for k in ("n", "dif_controlada", "dif_sem_controlo") if k in antigo["c3"]]
    if "c3" in r:
        r["c3_igual_ao_da_9b"] = {k: r["c3"][k] for k in chaves} == {k: antigo["c3"][k] for k in chaves}
    antigo["moeda_oficial"] = "risco"
    antigo["risco"] = r
    destino = SAIDA / "artigo_v31.json"
    destino.write_text(json.dumps(antigo, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{destino} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script refaz em risco, a moeda oficial, os números que o artigo tinha em xT: quanto se nega por jogo, por fase e por
# terço, como se distribui o índice dos jogadores em cada fase, se a ordem dos jogadores aguenta as variações dos parâmetros e o cone para a bola,
# os lances de exemplo e a comparação com as outras duas moedas. Guarda tudo numa secção nova do mesmo ficheiro, sem apagar o que lá estava.
