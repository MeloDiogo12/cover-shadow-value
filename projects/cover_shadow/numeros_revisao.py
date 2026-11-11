"""Revisão do artigo · números acrescentados na revisão final, na moeda risco. Só números, sem leitura.

Junta ao `artigo_v31.json` uma secção `revisao` com:
- `costas`: defesas de costas para o portador (frente do tronco a mais de 90° da direção do portador), em percentagem das
  observações e da ameaça negada (secções 6.3 e 9.2 do artigo);
- `c5_por_momento`: ρ de Spearman entre o cone pelo tronco e o cone virado para a bola, com a ameaça negada por momento de
  cada jogador em vez do total (secção 7.3);
- `c4_por_momento`: o mesmo ρ para cada variante dos parâmetros e do comprimento das linhas (secção 7.4).
A ameaça negada por momento é a ameaça negada atribuída a cada jogador (Shapley, moeda risco) a dividir pelo número de momentos
em que ele está em campo na equipa sem bola, o que retira o peso dos minutos jogados. Pressupõe `numeros_9d.py` já corrido.
Da raiz do repositório:
    uv run python projects/cover_shadow/numeros_revisao.py
"""

from __future__ import annotations

import argparse
import json
import time

import avaliacao as av
import figuras_agregados as ag
import numeros_9d as n9d
import numeros_v31 as n31
import orientacao_defesas as od
import pandas as pd

from footy.paths import output_dir

SAIDA = output_dir("cover_shadow")
JOGOS = ag.JOGOS


def negada_por_momento(JG: dict, tabelas: dict) -> pd.DataFrame:
    """Ameaça negada por momento de cada jogador, por jogo: soma do Shapley a dividir pelos momentos em que defende."""
    partes = []
    for jogo in JOGOS:
        _, atribuicoes = tabelas[jogo]
        momentos = JG[jogo]["defesas"].groupby("jogador").size().rename("momentos")
        negada = atribuicoes.groupby("defesa")["shapley"].sum().rename("negada")
        t = momentos.to_frame().join(negada).fillna({"negada": 0.0}).reset_index().rename(columns={"index": "jogador"})
        t["negada"] = t["negada"] / t["momentos"]
        partes.append(t.merge(JG[jogo]["jogadores"], on="jogador", how="left").assign(jogo=jogo))
    return pd.concat(partes, ignore_index=True)


def defesas_de_costas(JG: dict, base: dict) -> dict:
    """Percentagem das observações e da ameaça negada em que a frente do tronco está a mais de 90° (e 135°) da direção do portador."""
    d = od.defesas_com_angulo(JG)
    negada = pd.concat([base[j][1].groupby(["frame", "defesa"], as_index=False)["shapley"].sum().assign(jogo=j) for j in JOGOS])
    negada = negada.rename(columns={"defesa": "jogador"}).merge(d[["jogo", "frame", "jogador", "angulo"]], on=["jogo", "frame", "jogador"], how="left")
    total = negada["shapley"].sum()
    return {"observacoes": int(len(d)),
            "pct_observacoes_acima_de_90": float(100 * (d["angulo"] > 90).mean()),
            "pct_negada_acima_de_90": float(100 * negada.loc[negada["angulo"] > 90, "shapley"].sum() / total),
            "pct_negada_acima_de_135": float(100 * negada.loc[negada["angulo"] > 135, "shapley"].sum() / total),
            "pct_negada_sem_angulo": float(100 * negada.loc[negada["angulo"].isna(), "shapley"].sum() / total)}


def rho_por_momento(JG: dict, base_jogadores: pd.DataFrame, variante: str) -> dict:
    """ρ de Spearman (dentro da posição) da ameaça negada por momento entre a base e uma variante."""
    tabelas = {j: n9d.risco_da_variante(j, variante) for j in JOGOS}
    rho = av.spearman_por_posicao(base_jogadores, negada_por_momento(JG, tabelas))
    return {"variante": variante, "rho_media": rho["rho_media"], "rho_minimo": rho["rho_min"]}


def variantes_c4() -> list[str]:
    """As mesmas variantes do C4 da `numeros_9d.py`: parâmetros, raio do bloqueio pelo corpo e comprimento das linhas."""
    parametros = {**av.VALORES_ANTIGOS_V3, "raio_bloqueio_corpo": av.RAIOS_BLOQUEIO_V3, **n31.COMPRIMENTOS_C4}
    return [av.nome_da_variante(p, v) for p, valores in parametros.items() for v in valores]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so", nargs="*", help="só estes passos (os outros ficam como estão no ficheiro)")
    args = ap.parse_args()
    inicio = time.time()
    JG = ag.carregar_jogos()
    base = {j: n9d.risco_da_variante(j, "base") for j in JOGOS}
    base_jogadores = negada_por_momento(JG, base)
    passos = {"costas": lambda: defesas_de_costas(JG, base),
              "c5_por_momento": lambda: rho_por_momento(JG, base_jogadores, "bola"),
              "c4_por_momento": lambda: [rho_por_momento(JG, base_jogadores, v) for v in variantes_c4()]}
    antigo = n31.ler_json("artigo_v31.json")
    r = antigo.get("revisao", {}) if args.so else {}
    r["definicao"] = {"negada_por_momento": "soma do Shapley (risco) de cada jogador a dividir pelos momentos em que está na equipa sem bola",
                      "costas": "frente do tronco a mais de 90° da direção do portador (sem guarda-redes, só com pose)"}
    for nome, f in passos.items():
        if args.so and nome not in args.so:
            continue
        t0 = time.time()
        r[nome] = f()
        print(f"  {nome}: {time.time() - t0:.0f} s", flush=True)
    antigo["revisao"] = r
    destino = SAIDA / "artigo_v31.json"
    destino.write_text(json.dumps(antigo, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{destino} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script calcula os números que entraram na revisão final do artigo. Primeiro, quantas vezes os defesas estão
# de costas para quem tem a bola e quanta da ameaça negada vem desses momentos. Depois, se a ordem dos jogadores se mantém quando cada um
# é medido por momento em campo, e não pelo total, o que tira a vantagem de quem jogou mais minutos. Guarda tudo numa secção nova do mesmo
# ficheiro, sem apagar o que lá estava.
