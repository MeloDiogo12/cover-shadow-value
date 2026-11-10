"""Fase 8c · calcula a métrica v4 (D38) de um jogo e guarda as tabelas em data/processed/cover_shadow/v4/.

Pressupõe a tabela da fase 3 (`preparar_dados.py`) e a tabela de escolhas da fase 8a/8b (`escolha.py`). A v3 não
se altera. Com os parâmetros de `metrica/parametros.py`, da raiz do projeto:
    uv run python projects/cover_shadow/calcular_v4.py --jogo 1925299
    uv run python projects/cover_shadow/calcular_v4.py --jogo 1996435

Com o fundo (D39, omissão), `p_l(∅)` já não é 1: os defesas que não projetam sombra (a marcar, sem pose, guarda-redes) também
cortam passes; `--fundo sem_guarda_redes` tira o guarda-redes e `--fundo sem_fundo` dá a v4 anterior (D38), cada um na sua pasta.

Os coeficientes do modelo de escolha (M2) de cada jogo vêm do ajuste no outro jogo (cross-fitting), para não
avaliar um jogo com o seu próprio ajuste. Escreve três tabelas por jogo:
- `<jogo>_v4_linhas.parquet`: uma linha por frame e linha de passe (com escolha e perigo, sem e com sombras);
- `<jogo>_v4_momentos.parquet`: uma por frame (perigo esperado, interceção, dissuasão);
- `<jogo>_v4_atribuicoes.parquet`: Shapley de I e de D por frame e defesa.
`I`, `D` e `shapley_*` são os da ordem por omissão (simétrica); as três ordens da decomposição (D38) estão também em
colunas com o sufixo `_ordem1`, `_ordem2` e `_simetrica`.
"""

from __future__ import annotations

import argparse
import time

import avaliacao as av
import modelo_escolha as me
import numpy as np
import pandas as pd
from entradas import exigir
from escolha import PASTA_SAIDA, distancia_ao_adversario_mais_proximo, recetores_marcados
from metrica import agregacao
from metrica.fundo import defesas_de_fundo, p_fundo_das_linhas
from metrica.linhas import pode_afetar, pontos_da_linha, probabilidade_na_linha, recetores_com_valor
from metrica.parametros import PADRAO, Parametros
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.v4 import MAX_DEFESAS, ORDEM_PADRAO, ORDENS, avaliar_momento, avaliar_total
from metrica.velocidade import velocidade_dos_jogadores
from metrica.xt import sentido_de_ataque

from footy.ingest import skillcorner as sc

PASTA_V4 = PASTA_SAIDA / "v4"  # com fundo (D39); a v4 sem fundo está em `v4_sem_fundo/`
FUNDOS = ("completo", "sem_guarda_redes", "sem_fundo")  # F com o guarda-redes (omissão), sem ele, ou sem fundo (v4 inicial)
LIMITE_COMPRIMENTO = me.LIMITE_COMPRIMENTO  # D37: nenhuma linha acima de 35 m


def ajuste_do_outro_jogo(jogo: int, modelo: str = "M2") -> me.Ajuste:
    """Ajuste do modelo de escolha (`me.BASE` + as variáveis do `modelo`) no jogo que não é `jogo` (cross-fitting)."""
    dados = me.preparar_dados(pd.read_parquet(exigir(PASTA_SAIDA / "escolhas_v3.parquet", "escolha.py")))
    return me.ajustar(dados[dados["jogo"] != jogo], me.BASE + me.MODELOS[modelo])


def coeficientes_do_outro_jogo(jogo: int, modelo: str = "M2") -> np.ndarray:
    """Coeficientes do `modelo` (por omissão o M2: `me.BASE` + `sombra_tronco`) ajustados no outro jogo."""
    return ajuste_do_outro_jogo(jogo, modelo).coef


def coeficientes_dos_dois_jogos(modelo: str = "M2") -> np.ndarray:
    """Coeficientes do `modelo` ajustados com os dois jogos juntos (a avaliar no próprio jogo: serve só para a
    comparação com o cross-fitting na estabilidade, C4)."""
    dados = me.preparar_dados(pd.read_parquet(exigir(PASTA_SAIDA / "escolhas_v3.parquet", "escolha.py")))
    return me.ajustar(dados, me.BASE + me.MODELOS[modelo]).coef


def conjuntos_de_coeficientes(ajuste: me.Ajuste, n: int, semente: int = me.SEMENTE) -> np.ndarray:
    """`n` conjuntos de coeficientes tirados da distribuição normal estimada no ajuste (média = coeficientes,
    covariância = inverso do Hessiano), para a incerteza dos coeficientes (C4). Linha k = conjunto k."""
    return np.random.default_rng(semente).multivariate_normal(ajuste.coef, ajuste.cov, size=n)


# ---------------------------------------------------------------- um momento

def probabilidades_dos_defesas(origem, recetores: pd.DataFrame, defesas: pd.DataFrame, p: Parametros) -> tuple:
    """(P, nomes, sem_pose): P (linhas × defesas com sombra) é a probabilidade de cada defesa chegar a cada linha
    (D19, D30, como na v3); `nomes` são esses defesas; `sem_pose[l]` diz se algum defesa que podia afetar a linha
    não tem pose (a linha entra na mesma, só com a sombra dos que têm: D37, opção B)."""
    por_linha, sem_pose = [], []
    for _, recetor in recetores.iterrows():
        candidatos = defesas[pode_afetar(defesas, origem, recetor, p)]
        sem_pose.append(bool(candidatos["orientacao"].isna().any()))
        px, py = pontos_da_linha(origem, recetor, p)
        com_pose = candidatos[candidatos["orientacao"].notna()]
        por_linha.append({j: probabilidade_na_linha(d, origem, px, py, p) for j, (_, d) in zip(com_pose["jogador"], com_pose.iterrows())})
    probs = pd.DataFrame(por_linha).fillna(0.0)
    probs = probs.loc[:, (probs > 0).any(axis=0)]  # C: só os defesas que projetam sombra nalguma linha
    return probs.reindex(range(len(recetores))).fillna(0.0), np.array(sem_pose)


def variaveis_das_linhas(origem, recetores: pd.DataFrame, sentido: int, marcados: set, dist_adversario: dict,
                         sem_pose: np.ndarray) -> pd.DataFrame:
    """As variáveis da linha do modelo de escolha (`me.BASE`), calculadas como na fase 8a."""
    return pd.DataFrame({
        "comprimento": recetores["comprimento"].to_numpy(),
        "avanco": (recetores["x"].to_numpy() - origem[0]) * sentido,
        "valor_xt": recetores["valor_xt"].to_numpy(),
        "recetor_marcado": [int(r in marcados) for r in recetores["recetor"]],
        "dist_adversario": [dist_adversario[r] for r in recetores["recetor"]],
        "linha_sem_pose": sem_pose.astype(int)})


def avaliar_frame_v4(jogadores: pd.DataFrame, defesas: pd.DataFrame, sentido: int, campo: tuple, marcados: set,
                     dist_adversario: dict, coef: np.ndarray, p: Parametros = PADRAO,
                     limite: float = LIMITE_COMPRIMENTO, alternativos: dict | None = None,
                     guarda_redes: pd.DataFrame | None = None, fundo: str = "completo"):
    """(linhas, momento, atribuições, alternativos) de um frame, ou None se não sobra nenhuma linha. Linhas: as da v3,
    sem fora de jogo (D26), sem cruzamentos (D27) e sem as de mais de `limite` m (D37: 35 m). `fundo` (D39): `"completo"`
    junta aos defesas fora de C o `guarda_redes` do frame (x, y, vx, vy); `"sem_guarda_redes"` usa só os defesas fora de C;
    `"sem_fundo"` é a v4 inicial (`p = 1` sem sombras). `alternativos`
    ({nome: coeficientes}) acrescenta, para cada um, o perigo esperado e o Shapley do total (só a ameaça negada total;
    mesmas linhas e sombras, só muda a escolha): devolve-se como {nome: (E0, v, defesas, shapley)}."""
    origem, recetores = recetores_com_valor(jogadores, sentido, *campo, p)
    recetores["comprimento"] = np.hypot(recetores["x"] - origem[0], recetores["y"] - origem[1])
    recetores = recetores[~recetores["fora_de_jogo"] & ~recetores["cruzamento"]
                          & (recetores["comprimento"] <= limite)].reset_index(drop=True)
    if recetores.empty:
        return None
    probs, sem_pose = probabilidades_dos_defesas(origem, recetores, defesas, p)
    x = variaveis_das_linhas(origem, recetores, sentido, marcados, dist_adversario, sem_pose)
    cortados = max(0, probs.shape[1] - MAX_DEFESAS)
    if cortados:  # não acontece com 10 jogadores de campo; se acontecesse, ficavam os que mais perigo cobrem
        probs = probs[(probs.mul(x["valor_xt"], axis=0)).sum().nlargest(MAX_DEFESAS).index]
    eta = x[me.BASE].to_numpy() @ coef[:-1]
    if fundo not in FUNDOS:
        raise ValueError(f"fundo deve ser um de {FUNDOS}")
    p_fundo = None
    if fundo != "sem_fundo":
        f = defesas_de_fundo(defesas, probs.columns, guarda_redes if fundo == "completo" else None)
        p_fundo = p_fundo_das_linhas(origem, recetores, f, p)
    m = avaliar_momento(x["valor_xt"], eta, coef[-1], probs.to_numpy(), p_fundo=p_fundo)
    primeiro = jogadores.iloc[0]
    frame = primeiro["frame"]
    com_sombra = probs.to_numpy() > 0
    quem = [",".join(str(j) for j in sorted(probs.columns[linha])) for linha in com_sombra]  # defesas com sombra na linha
    linhas = x.assign(frame=frame, posse_id=primeiro["posse_id"], portador=primeiro["portador"],
                      recetor=recetores["recetor"].to_numpy(), s0=m.s0, sC=m.sC, pC=m.pC, p_fundo=m.p0, n_sombras=com_sombra.sum(axis=1),
                      quem_sombra=quem)
    linhas["negada_linha"] = linhas["valor_xt"] * (m.s0 * m.p0 - m.sC * m.pC)  # parte de cada linha em v = E(∅) − E(C)
    momento = {"frame": frame, "posse_id": primeiro["posse_id"], "portador": primeiro["portador"], "n_linhas": len(x),
               "n_linhas_sem_pose": int(sem_pose.sum()), "n_defesas": probs.shape[1], "n_defesas_cortados": cortados,
               "E0": m.E0, "EC": m.EC, "I": m.I, "D": m.D, "v": m.v,
               **{f"{t}_{o}": getattr(d, t) for o, d in m.por_ordem.items() for t in ("I", "D")}}
    atribuicoes = pd.DataFrame({"frame": frame, "defesa": probs.columns, "shapley_I": m.shapley_I,
                                "shapley_D": m.shapley_D, "shapley": m.shapley_I + m.shapley_D,
                                **{f"shapley_{t}_{o}": getattr(d, f"shapley_{t}") for o, d in m.por_ordem.items() for t in ("I", "D")},
                                **{f"shapley_{o}": d.shapley_I + d.shapley_D for o, d in m.por_ordem.items()}})
    extra = {}
    for nome, c in (alternativos or {}).items():
        t = avaliar_total(x["valor_xt"], x[me.BASE].to_numpy() @ c[:-1], c[-1], probs.to_numpy(), p_fundo)
        extra[nome] = (t.E0, t.v, probs.columns.to_numpy(), t.shapley)
    return linhas, momento, atribuicoes, extra


# ---------------------------------------------------------------- um jogo

def guarda_redes_por_frame(tabela: pd.DataFrame, p: Parametros = PADRAO) -> dict:
    """{frame: guarda-redes da equipa que defende (x, y, vx, vy)}: parte do fundo `"completo"` (D39). A velocidade é a de
    `metrica.velocidade`, como a dos outros defesas."""
    gr = tabela[tabela["defende"] & (tabela["posicao"] == "GK")][["frame", "jogador", "x", "y"]]
    gr = gr.merge(velocidade_dos_jogadores(tabela, p.velocidade_maxima_tracking), on=["frame", "jogador"], how="left")
    return {frame: g for frame, g in gr.groupby("frame")}


def calcular_jogo_v4(jogo: int, p: Parametros = PADRAO, orientacao: str = "tronco", limite: float = LIMITE_COMPRIMENTO,
                     alternativos: dict | None = None, fundo: str = "completo") -> dict:
    """Tabelas da v4 de um jogo (com o fundo de D39; `fundo`: ver `avaliar_frame_v4`). `orientacao="bola"` vira as defesas para o portador (C5) e usa, na escolha, os
    coeficientes do M1 (sombra virada para a bola) em vez dos do M2; `limite` é o comprimento máximo das linhas (D37);
    `alternativos` ({nome: coeficientes}) acrescenta as tabelas leves `alternativos[nome]` (momentos e atribuições do total)."""
    match = sc.ler_match(jogo)
    tabela = carregar_tabela(jogo)
    defesas = preparar_defesas(tabela, match, p, orientacao)
    campo = sc.dimensoes_campo(match)
    coef = coeficientes_do_outro_jogo(jogo, "M1" if orientacao == "bola" else "M2")

    marcados = set(zip(recetores_marcados(defesas)["frame"], recetores_marcados(defesas)["recetor"]))
    dist = distancia_ao_adversario_mais_proximo(tabela[["frame", "jogador", "x", "y", "defende"]])
    dist_por_frame = {f: dict(zip(g["recetor"], g["dist_adversario"])) for f, g in dist.groupby("frame")}
    defesas_por_frame = dict(tuple(defesas.groupby("frame")))
    guardas_por_frame = guarda_redes_por_frame(tabela, p) if fundo == "completo" else {}
    marcados_por_frame: dict = {}
    for frame, recetor in marcados:
        marcados_por_frame.setdefault(frame, set()).add(recetor)

    linhas, momentos, atribuicoes, extras = [], [], [], {n: ([], []) for n in alternativos or {}}
    for frame, jogadores in tabela.groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any() or frame not in defesas_por_frame:
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        r = avaliar_frame_v4(jogadores, defesas_por_frame[frame], sentido, campo, marcados_por_frame.get(frame, set()),
                             dist_por_frame[frame], coef, p, limite, alternativos,
                             guardas_por_frame.get(frame), fundo)
        if r is not None:
            linhas.append(r[0]), momentos.append(r[1]), atribuicoes.append(r[2])
            for nome, (e0, v, quem, shapley) in r[3].items():
                extras[nome][0].append({"frame": frame, "E0": e0, "v": v})
                extras[nome][1].append(pd.DataFrame({"frame": frame, "defesa": quem, "shapley": shapley}))
    resultado = {"linhas": pd.concat(linhas, ignore_index=True), "momentos": pd.DataFrame(momentos),
                 "atribuicoes": pd.concat(atribuicoes, ignore_index=True), "defesas": defesas, "coef": coef}
    resultado["alternativos"] = {n: {"momentos": pd.DataFrame(m), "atribuicoes": pd.concat(a, ignore_index=True)}
                                 for n, (m, a) in extras.items()}
    return resultado


def pasta_v4(fundo: str = "completo"):
    """Pasta das tabelas: `v4/` (com fundo), `v4_sem_guarda_redes/` ou `v4_sem_fundo/` (a v4 anterior, D38)."""
    return PASTA_V4 if fundo == "completo" else PASTA_SAIDA / f"v4_{fundo}"


def caminho_v4(jogo: int, tabela: str, fundo: str = "completo"):
    return pasta_v4(fundo) / f"{jogo}_v4_{tabela}.parquet"


def ler_v4(jogo: int, fundo: str = "completo") -> dict:
    return {t: pd.read_parquet(exigir(caminho_v4(jogo, t, fundo), f"calcular_v4.py --jogo {jogo} --fundo {fundo}")) for t in ("linhas", "momentos", "atribuicoes")}


# ---------------------------------------------------------------- agregação (igual à v3)

def jogadores_do_jogo(match: dict) -> pd.DataFrame:
    return pd.DataFrame([{"jogador": q["id"], "nome": q["short_name"], "equipa": q["team_id"],
                          "posicao": q["player_role"]["acronym"]} for q in match["players"]])


def negada_por_jogador_v4(momentos: pd.DataFrame, atribuicoes: pd.DataFrame, presenca: pd.DataFrame,
                          jogadores: pd.DataFrame, ordem: str = ORDEM_PADRAO) -> pd.DataFrame:
    """Por jogador (média por posse, soma por jogador, como na v3): `negada` total e as partes `negada_I` e
    `negada_D`, na ordem da decomposição pedida; `disponivel` = E(∅) (perigo esperado sem sombras) e `pct_negada`. `presenca`: frame, posse_id, jogador
    (todos os frames da posse: os sem linhas contam com zero, como na v3)."""
    lances = momentos.rename(columns={"E0": "disponivel"})[["frame", "disponivel"]]
    partes = {}
    for nome, coluna in (("negada", f"shapley_{ordem}"), ("negada_I", f"shapley_I_{ordem}"), ("negada_D", f"shapley_D_{ordem}")):
        a = atribuicoes[["frame", "defesa"]].assign(shapley=atribuicoes[coluna])
        por_posse = agregacao.por_posse_e_defesa(a, presenca, lances)
        partes[nome] = agregacao.por_jogador(por_posse, jogadores)
    t = partes["negada"].merge(partes["negada_I"][["jogador", "negada"]].rename(columns={"negada": "negada_I"}), on="jogador")
    return t.merge(partes["negada_D"][["jogador", "negada"]].rename(columns={"negada": "negada_D"}), on="jogador")


# ---------------------------------------------------------------- números para o notebook

def verificar_identidades(v4: dict) -> dict:
    """Erros máximos (em xT) das identidades que a v4 tem de cumprir nos dados reais: I + D = v por momento, soma dos
    Shapley de I e de D por momento, e soma por linha da ameaça negada = v."""
    m = v4["momentos"].set_index("frame")
    soma = v4["atribuicoes"].groupby("frame")[["shapley_I", "shapley_D"]].sum().reindex(m.index).fillna(0.0)
    por_linhas = v4["linhas"].groupby("frame")["negada_linha"].sum().reindex(m.index)
    return {"I_mais_D_menos_v": float((m["I"] + m["D"] - m["v"]).abs().max()),
            "shapley_I_menos_I": float((soma["shapley_I"] - m["I"]).abs().max()),
            "shapley_D_menos_D": float((soma["shapley_D"] - m["D"]).abs().max()),
            "soma_das_linhas_menos_v": float((por_linhas - m["v"]).abs().max())}


def lance_v3_e_v4(jogo: int, frame: int, v4: dict, jogadores: pd.DataFrame,
                  ordens: tuple = (ORDEM_PADRAO,)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(totais, por defesa) de um frame, v3 e v4 lado a lado. Totais em xT; por defesa: Shapley da v3 e da v4 (I, D,
    total). Com várias `ordens` há uma coluna (totais) ou três (por defesa) por ordem: `v4 <ordem>`, `<ordem> I`..."""
    linhas3, atrib3 = av.ler_variante(jogo, "base", av.PASTA_AVALIACAO_V3)
    l3, a3 = linhas3[(linhas3["frame"] == frame) & ~linhas3["excluida"]], atrib3[atrib3["frame"] == frame]
    m4 = v4["momentos"].set_index("frame").loc[frame]
    colunas = {"v3": {"linhas avaliadas": len(l3), "disponível": l3["valor_xt"].sum(), "E(∅)": np.nan, "E(C)": np.nan,
                      "ameaça negada": l3["ameaca_negada"].sum(), "interceção (I)": np.nan, "dissuasão (D)": np.nan}}
    for o in ordens:
        colunas[f"v4 {o}" if len(ordens) > 1 else "v4"] = {
            "linhas avaliadas": m4["n_linhas"], "disponível": m4["E0"], "E(∅)": m4["E0"], "E(C)": m4["EC"],
            "ameaça negada": m4["v"], "interceção (I)": m4[f"I_{o}"], "dissuasão (D)": m4[f"D_{o}"]}
    totais = pd.DataFrame(colunas)
    a4 = v4["atribuicoes"][v4["atribuicoes"]["frame"] == frame].set_index("defesa")
    partes = [a3.groupby("defesa")["shapley"].sum().rename("v3")]
    for o in ordens:
        rotulo = f"{o} " if len(ordens) > 1 else ""
        partes.append(a4[[f"shapley_I_{o}", f"shapley_D_{o}", f"shapley_{o}"]].rename(
            columns={f"shapley_I_{o}": f"v4 {rotulo}I", f"shapley_D_{o}": f"v4 {rotulo}D", f"shapley_{o}": f"v4 {rotulo}total"}))
    por_defesa = pd.concat(partes, axis=1).fillna(0.0)
    por_defesa.index = por_defesa.index.map(jogadores.set_index("jogador")["nome"])
    return totais, por_defesa.sort_values(por_defesa.columns[-1], ascending=False)


def jogadores_v3_e_v4(jogo: int, v4: dict, defesas: pd.DataFrame, jogadores: pd.DataFrame,
                      ordem: str = ORDEM_PADRAO) -> pd.DataFrame:
    """Por jogador do jogo: ameaça negada da v3 e da v4 (total, I e D, na `ordem` pedida), em xT."""
    linhas3, atrib3 = av.ler_variante(jogo, "base", av.PASTA_AVALIACAO_V3)
    presenca = defesas[["frame", "posse_id", "jogador"]]
    t3 = av.negada_por_jogador(linhas3, atrib3, presenca, jogadores)[["jogador", "posicao", "negada", "disponivel"]]
    t4 = negada_por_jogador_v4(v4["momentos"], v4["atribuicoes"], presenca, jogadores, ordem)[
        ["jogador", "negada", "negada_I", "negada_D", "disponivel", "pct_negada"]]
    return t3.merge(t4, on="jogador", suffixes=("_v3", "_v4")).assign(jogo=jogo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogo", type=int, default=sc.JOGO_PADRAO, choices=sc.JOGOS_COM_POSE)
    ap.add_argument("--fundo", choices=FUNDOS, default="completo", help="defesas de fundo (D39); `sem_fundo` é a v4 anterior")
    args = ap.parse_args()
    jogo, fundo = args.jogo, args.fundo
    inicio = time.time()
    r = calcular_jogo_v4(jogo, fundo=fundo)
    pasta_v4(fundo).mkdir(parents=True, exist_ok=True)
    for tabela in ("linhas", "momentos", "atribuicoes"):
        r[tabela].to_parquet(caminho_v4(jogo, tabela, fundo), index=False)
    m = r["momentos"]
    print(f"{time.time() - inicio:.0f} s; {len(m)} momentos, {len(r['linhas'])} linhas; defesas por momento: máx. {m['n_defesas'].max()}, "
          f"cortados: {int(m['n_defesas_cortados'].sum())}")
    print(m[["E0", "EC", "I", "D", "v"] + [f"{t}_{o}" for o in ORDENS for t in ("I", "D")]].sum())


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script faz, para cada instante da posse, a conta da v4. Lista as linhas de passe
# (sem fora de jogo, cruzamentos nem passes de mais de 35 m), calcula a probabilidade de o portador escolher
# cada uma (com um modelo ajustado no outro jogo), e vê quanto perigo se perde quando as sombras dos
# defesas entram: uma parte porque os passes já não chegam e outra porque o portador muda de escolha. Depois
# reparte cada parte pelos defesas, e guarda as tabelas para os notebooks não terem de recalcular.
