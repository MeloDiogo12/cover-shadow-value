"""Fase 9c · moeda «risco» na v3.1, índice por fase, casos concretos e números novos (D43). Só números, sem leitura.

Pressupõe as tabelas da v3.1 (`avaliar_variantes.py --versao v3_1`), `artigo_v31.json` da 9b (de onde saem os 4 momentos que já existem) e
as tabelas auxiliares de `avaliacao_v31` (ultrapassados) e `avaliacao_risco` (custos de perda), que se calculam na primeira execução.
Escreve `outputs/cover_shadow/fase9c.json`, acrescenta a secção `fase_9c` a `artigo_v31.json` e guarda as imagens em
`outputs/cover_shadow/momentos_9c/` e o índice por jogador em `outputs/cover_shadow/indice_por_fase.csv`. Da raiz do repositório:
    uv run python projects/cover_shadow/numeros_9c.py
"""

from __future__ import annotations

import json
import time

import avaliacao as av
import avaliacao_risco as ar
import avaliacao_v31 as a31
import figuras_agregados as ag
import gerar_candidatos as gc
import numeros_v31 as n31
import numpy as np
import pandas as pd
from figuras_9c import figura_momento_3
from metrica.xt import sentido_de_ataque

from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

SAIDA = output_dir("cover_shadow")
PASTA_IMAGENS = SAIDA / "momentos_9c"
JOGOS = ag.JOGOS
VERSAO = n31.VERSAO
MOEDAS = ar.MOEDAS_3
FASES_PERCENTIL = ("high_block", "medium_block")
AVANCADOS = ("CF", "LW", "RW")  # quem tem a sombra nos casos de bloco alto
PORTADORES = ("GK", "LCB", "RCB")  # quem tem a bola nesses casos
MARGEM_AREA_M = 5.0  # «perto da própria área»: até 5 m para lá da área grande
RECETOR_JUNTO_A_AREA_M = 10.0  # recetor «junto à área»: até 10 m para lá da linha da área grande
FRAME_DA_ETIQUETA = {"jogo": 1996435, "frame": 28188}
DESLOCAMENTO = 1_000_000  # separa os frames de cada jogo quando se juntam os dois


def registos(t: pd.DataFrame) -> list[dict]:
    return n31.registos(t)


def tabelas_das_tres_moedas(JG: dict) -> dict:
    """{jogo: {moeda: (linhas, atribuições)}} da base v3.1: xT, linhas quebradas e risco."""
    T = n31.tabelas_das_moedas(JG)
    for j in JOGOS:
        T[j]["risco"] = ar.moeda_risco(*T[j]["xt"], ar.custos_guardados(j))
    return T


# ---------------------------------------------------------------- 1. a moeda risco

def numeros_da_moeda_risco(JG: dict, T: dict) -> dict:
    """Totais das três moedas, custo da perda e o que acontece quando o ponto de corte está na área de quem tem a bola."""
    r = {"totais": {}, "custo_da_perda": {}, "corte_na_area_propria": {}}
    for moeda in MOEDAS:
        lances = n31.lances_por_moeda(T, moeda)
        av_ = lances[lances["n_avaliadas"] > 0]
        r["totais"][moeda] = {"lances": len(av_), "disponivel": float(av_["disponivel"].sum()), "negada": float(av_["negada"].sum()),
                              "pct_negada": float(100 * av_["negada"].sum() / av_["disponivel"].sum())}
    risco = pd.concat([T[j]["risco"][0].assign(jogo=j) for j in JOGOS], ignore_index=True)
    xt = pd.concat([T[j]["xt"][0].assign(jogo=j) for j in JOGOS], ignore_index=True)
    a = risco[~risco["excluida"]]
    r["custo_da_perda"] = {"linhas_avaliadas": len(a), "custo_medio": float(a["custo"].mean()), "percentis_custo": {
        str(q): float(a["custo"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "ganho_medio": float(a["ganho"].mean()), "percentis_ganho": {str(q): float(a["ganho"].quantile(q)) for q in (0.5, 0.75, 0.9, 0.99)},
        "linhas_com_ganho_zero": int((a["ganho"] == 0).sum()), "pct_linhas_com_ganho_zero": float(100 * (a["ganho"] == 0).mean()),
        "pct_linhas_com_custo_maior_que_ganho": float(100 * (a["custo"] > a["ganho"]).mean())}
    r["corte_na_area_propria"] = {"juntos": ar.custo_por_zona_do_corte(risco),
                                  **{str(j): ar.custo_por_zona_do_corte(risco[risco["jogo"] == j]) for j in JOGOS}}
    # as mesmas linhas, na moeda xT: o que valiam antes
    dentro = risco[~risco["excluida"] & risco["corte_na_area_propria"]]
    mesmas = dentro[["jogo", "frame", "recetor", "ameaca_negada"]].merge(
        xt[["jogo", "frame", "recetor", "ameaca_negada"]], on=["jogo", "frame", "recetor"], suffixes=("_risco", "_xt"))
    r["corte_na_area_propria"]["mesmas_linhas_em_xt"] = {
        "linhas": len(mesmas), "negada_risco": float(mesmas["ameaca_negada_risco"].sum()), "negada_xt": float(mesmas["ameaca_negada_xt"].sum()),
        "negada_media_risco_por_linha_com_sombra": float(mesmas.loc[mesmas["ameaca_negada_risco"] > 0, "ameaca_negada_risco"].mean()),
        "negada_media_xt_por_linha_com_sombra": float(mesmas.loc[mesmas["ameaca_negada_risco"] > 0, "ameaca_negada_xt"].mean())}
    sem_ganho = a[(a["ganho"] == 0) & (a["ameaca_negada"] > 0)]
    r["linhas_sem_ganho_de_xt_com_sombra"] = {
        "linhas": len(sem_ganho), "pct_com_corte_na_area_propria": float(100 * sem_ganho["corte_na_area_propria"].mean()),
        "negada_risco": float(sem_ganho["ameaca_negada"].sum()), "pct_da_negada_risco_total": float(100 * sem_ganho["ameaca_negada"].sum() / a["ameaca_negada"].sum())}
    return r


# ---------------------------------------------------------------- 2. índice por fase

def juntar_jogos(JG: dict, T: dict, ctx: pd.DataFrame) -> dict:
    """Presença, contexto, jogadores e atribuições dos dois jogos numa só tabela (frames deslocados por jogo para não se confundirem)."""
    desloc = {j: i * DESLOCAMENTO for i, j in enumerate(JOGOS)}
    presenca = pd.concat([JG[j]["defesas"][["frame", "jogador"]].assign(frame=lambda d, j=j: d["frame"] + desloc[j]) for j in JOGOS], ignore_index=True)
    contexto = pd.concat([ctx[ctx["jogo"] == j][["frame", "fase"]].assign(frame=lambda d, j=j: d["frame"] + desloc[j]) for j in JOGOS], ignore_index=True)
    jogadores = pd.concat([JG[j]["jogadores"].assign(jogo=j) for j in JOGOS], ignore_index=True)
    atrib = {m: pd.concat([T[j][m][1][["frame", "defesa", "shapley"]].assign(frame=lambda d, j=j: d["frame"] + desloc[j]) for j in JOGOS],
                          ignore_index=True) for m in MOEDAS}
    return {"presenca": presenca, "contexto": contexto, "jogadores": jogadores, "atribuicoes": atrib}


def indices_das_moedas(J: dict) -> dict:
    return {m: ar.indice_por_fase(J["atribuicoes"][m], J["presenca"], J["contexto"], J["jogadores"]) for m in MOEDAS}


def distribuicao_dos_indices(indices: dict) -> dict:
    saida = {}
    for moeda, t in indices.items():
        saida[moeda] = {"distribuicao_por_fase": registos(ar.distribuicao_do_indice(t)),
                        "media_da_fase_negada_por_momento": {f: float(v) for f, v in t.groupby("fase")["media_da_fase"].first().items()}}
        e = t[t["momentos"] >= ar.MIN_MOMENTOS]
        saida[moeda]["maiores_indices_por_fase"] = {
            fase: registos(g.nlargest(5, "indice")[["nome", "posicao", "jogo", "momentos", "indice", "percentil"]])
            for fase, g in e.groupby("fase") if fase in a31.BLOCOS}
    return saida


def guardar_indices(indices: dict) -> None:
    tabela = pd.concat([t.assign(moeda=m) for m, t in indices.items()], ignore_index=True)
    tabela.to_csv(SAIDA / "indice_por_fase.csv", index=False)


# ---------------------------------------------------------------- 3. casos concretos

def candidatos_de_bloco_alto(JG: dict, T: dict, ctx: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, dict]:
    """Momentos de bloco alto em que um avançado (`AVANCADOS`) tem sombra sobre a linha do guarda-redes ou de um central (`PORTADORES`),
    com a bola na área própria ou até `MARGEM_AREA_M` para lá dela, para um colega junto à área (até `RECETOR_JUNTO_A_AREA_M` para lá da área
    grande). Ordenados pela ameaça negada (Shapley na moeda risco) do avançado nessa linha; uma só posse por momento."""
    partes = []
    for j in JOGOS:
        match, tabela = JG[j]["match"], JG[j]["tabela"]
        comprimento, _ = a31.sc.dimensoes_campo(match)
        frames = ctx[(ctx["jogo"] == j) & (ctx["fase"] == "high_block")]["frame"]
        t = tabela[tabela["frame"].isin(frames)]
        port = t[t["jogador"] == t["portador"]][["frame", "posse_id", "jogador", "x", "y", "posicao", "equipa_com_bola", "periodo"]]
        port = port.assign(sentido=[sentido_de_ataque(match, e, p) for e, p in zip(port["equipa_com_bola"], port["periodo"])])
        port["dist_baliza"] = port["x"] * port["sentido"] + comprimento / 2
        port = port[port["posicao"].isin(PORTADORES) & (port["dist_baliza"] <= ar.AREA_PROFUNDIDADE_M + MARGEM_AREA_M)
                    & (port["y"].abs() <= ar.AREA_LARGURA_M / 2 + MARGEM_AREA_M)]
        rec = t[["frame", "jogador", "x", "y", "posicao"]].rename(columns={"jogador": "recetor", "x": "rx", "y": "ry", "posicao": "pos_recetor"})
        nomes = JG[j]["jogadores"].set_index("jogador")
        linhas, atrib = T[j]["risco"]
        _, atrib_xt = T[j]["xt"]
        _, atrib_li = T[j]["linhas"]
        x = atrib.merge(linhas[["frame", "recetor", "comprimento", "ganho", "custo", "p_chegar", "corte_na_area_propria"]], on=["frame", "recetor"])
        x = x.merge(port.rename(columns={"jogador": "portador", "x": "px", "y": "py", "posicao": "pos_portador"}), on="frame").merge(rec, on=["frame", "recetor"])
        x = x.assign(pos_defesa=x["defesa"].map(nomes["posicao"]), defesa_nome=x["defesa"].map(nomes["nome"]), recetor_nome=x["recetor"].map(nomes["nome"]),
                     portador_nome=x["portador"].map(nomes["nome"]),
                     dist_baliza_recetor=x["rx"] * x["sentido"] + comprimento / 2)
        x = x[x["pos_defesa"].isin(AVANCADOS) & (x["dist_baliza_recetor"] <= ar.AREA_PROFUNDIDADE_M + RECETOR_JUNTO_A_AREA_M)
              & (x["recetor"] != x["portador"])]
        chave = ["frame", "recetor", "defesa"]
        x = x.merge(atrib_xt[chave + ["shapley"]].rename(columns={"shapley": "shapley_xt"}), on=chave, how="left")
        x = x.merge(atrib_li[chave + ["shapley"]].rename(columns={"shapley": "shapley_linhas"}), on=chave, how="left")
        partes.append(x.rename(columns={"shapley": "shapley_risco"}).assign(jogo=j))
    c = pd.concat(partes, ignore_index=True).sort_values("shapley_risco", ascending=False, ignore_index=True)
    escolhidos, posses = [], set()
    for _, r in c.iterrows():
        if (r["jogo"], r["posse_id"]) not in posses:
            escolhidos.append(r)
            posses.add((r["jogo"], r["posse_id"]))
        if len(escolhidos) == n:
            break
    resumo = {"linhas_candidatas": len(c), "momentos_candidatos": int(c[["jogo", "frame"]].drop_duplicates().shape[0]),
              "posses_candidatas": len(set(zip(c["jogo"], c["posse_id"]))),
              "criterios": {"fase": "high_block", "portador": list(PORTADORES), "portador_a_menos_de_m_da_propria_baliza": ar.AREA_PROFUNDIDADE_M + MARGEM_AREA_M,
                            "sombra_de": list(AVANCADOS), "recetor_a_menos_de_m_da_baliza_do_portador": ar.AREA_PROFUNDIDADE_M + RECETOR_JUNTO_A_AREA_M,
                            "ordem": "Shapley do avançado nessa linha na moeda risco"}}
    return pd.DataFrame(escolhidos).reset_index(drop=True), resumo


def percentil_do_momento(lances: dict, ctx: pd.DataFrame, jogo: int, frame: int) -> dict:
    """Percentil (0–100) do momento, em cada moeda, entre os momentos de blocos alto e médio (dois jogos juntos; empates com a média)."""
    saida = {}
    for m, t in lances.items():
        base = t[t["n_avaliadas"] > 0].merge(ctx[["jogo", "frame", "fase"]], on=["jogo", "frame"])
        base = base[base["fase"].isin(FASES_PERCENTIL)]["negada"]
        v = t[(t["jogo"] == jogo) & (t["frame"] == frame)]["negada"].iloc[0]
        saida[m] = float(100 * ((base < v).sum() + 0.5 * (base == v).sum()) / len(base))
    return saida


def vista_junto_a_baliza(JG: dict, jogo: int, frame: int, meia_largura: float = 21.0, meia_altura: float = 18.2) -> tuple:
    """Vista ((x0, x1), (y0, y1)) centrada no portador, empurrada para dentro do campo para não passar da linha de fundo, na proporção do painel."""
    comprimento, largura = a31.sc.dimensoes_campo(JG[jogo]["match"])
    t = JG[jogo]["tabela"]
    p = t[(t["frame"] == frame) & (t["jogador"] == t["portador"])].iloc[0]
    cx = float(np.clip(p["x"], -comprimento / 2 + meia_largura - 3, comprimento / 2 - meia_largura + 3))
    cy = float(np.clip(p["y"], -largura / 2 + meia_altura, largura / 2 - meia_altura))
    return (cx - meia_largura, cx + meia_largura), (cy - meia_altura, cy + meia_altura)


def imagem_do_momento(JG, T, ctx, lances, indices, brand, cache: dict, jogo: int, frame: int, rotulo: str, nome_img: str,
                      aproximar: bool = False) -> dict:
    """Guarda a imagem do momento e devolve os números que ela mostra."""
    if jogo not in cache:
        cache[jogo] = gc.carregar(jogo)
    nomes = JG[jogo]["jogadores"].set_index("jogador")["nome"].to_dict()
    valores = {m: a31.negada_por_defesa_no_lance(*T[jogo][m], frame, nomes) for m in MOEDAS}
    fase = ctx[(ctx["jogo"] == jogo) & (ctx["frame"] == frame)]["fase"].iloc[0]
    por_nome = {m: {nomes[r["jogador"]]: float(r["indice"]) for _, r in indices[m][indices[m]["fase"] == fase].iterrows() if r["jogador"] in nomes}
                for m in MOEDAS}
    percentis = percentil_do_momento(lances, ctx, jogo, frame)
    custos = ar.custos_guardados(jogo)
    risco = T[jogo]["risco"][0]
    com_negada = risco[(risco["frame"] == frame) & (risco["ameaca_negada"] > 0)][["recetor"]]
    cortes = custos[custos["frame"] == frame].merge(com_negada, on="recetor")
    figura_momento_3(cache[jogo], frame, rotulo, valores, percentis, por_nome, cortes, brand, PASTA_IMAGENS / f"{nome_img}.png",
                      vista_junto_a_baliza(JG, jogo, frame) if aproximar else None)
    return {"jogo": jogo, "frame": frame, "fase": fase, "imagem": f"outputs/cover_shadow/momentos_9c/{nome_img}.png",
            **{m: valores[m] for m in MOEDAS}, "percentil": percentis,
            "indice_na_fase_dos_defesas_com_ameaca_negada": {m: {n: por_nome[m].get(n) for n in valores[m]["shapley"]} for m in MOEDAS}}


def casos_concretos(JG: dict, T: dict, ctx: pd.DataFrame) -> dict:
    brand = load_brand()
    apply_theme(brand)
    lances = {m: n31.lances_por_moeda(T, m) for m in MOEDAS}
    J = juntar_jogos(JG, T, ctx)
    indices = indices_das_moedas(J)
    cache: dict = {}
    r = {"bloco_alto": [], "momentos_9b": {}, "C06_e_C03": {}}
    escolhidos, resumo = candidatos_de_bloco_alto(JG, T, ctx)
    r["bloco_alto_criterios"] = resumo
    for k, c in escolhidos.iterrows():
        info = imagem_do_momento(JG, T, ctx, lances, indices, brand, cache, int(c["jogo"]), int(c["frame"]),
                                 f"bloco alto · {c['portador_nome']} ({c['pos_portador']}) → {c['recetor_nome']} · sombra de {c['defesa_nome']} · {k + 1}",
                                 f"bloco_alto_{k + 1}_{int(c['jogo'])}_{int(c['frame'])}", aproximar=True)
        info["linha"] = {x: (c[x].item() if hasattr(c[x], "item") else c[x]) for x in (
            "portador_nome", "pos_portador", "recetor_nome", "pos_recetor", "defesa_nome", "pos_defesa", "comprimento", "ganho", "custo",
            "p_chegar", "p_defesa", "corte_na_area_propria", "shapley_xt", "shapley_linhas", "shapley_risco", "dist_baliza", "dist_baliza_recetor")}
        r["bloco_alto"].append(info)
    antigos = n31.ler_json("artigo_v31.json")["moedas_por_fase"]["momentos_que_mais_discordam"]
    for grupo, lista in antigos.items():
        r["momentos_9b"][grupo] = []
        for k, m in enumerate(lista):
            rotulo = f"{'xT alto, linhas baixo' if grupo == 'xt_alto' else 'linhas alto, xT baixo'} · {k + 1}"
            r["momentos_9b"][grupo].append(imagem_do_momento(JG, T, ctx, lances, indices, brand, cache, m["jogo"], m["frame"], rotulo,
                                                              f"momento_9b_{grupo}_{k + 1}_{m['jogo']}_{m['frame']}"))
    for nome, lance in (("C06", n31.C06), ("C03", n31.C03)):
        r["C06_e_C03"][nome] = imagem_do_momento(JG, T, ctx, lances, indices, brand, cache, lance["jogo"], lance["frame"], nome,
                                                 f"{nome}_{lance['jogo']}_{lance['frame']}")
    return r


# ---------------------------------------------------------------- 4. números novos

def numeros_por_fase(JG: dict, T: dict, ctx: pd.DataFrame) -> dict:
    """ρ por jogador dentro da posição entre as três moedas, em cada fase; parte da ameaça negada total por fase em cada moeda."""
    pares = (("xt", "linhas"), ("xt", "risco"), ("linhas", "risco"))
    r = {"rho_por_jogador_dentro_da_posicao": {}, "parte_da_negada_total_por_fase": {}}
    for fase in a31.BLOCOS:
        por_moeda = {m: n31.jogadores_por_moeda(JG, T, m, lambda j, fase=fase: a31.presenca_na_fase(
            JG[j]["defesas"][["frame", "posse_id", "jogador"]], ctx[ctx["jogo"] == j], fase)) for m in MOEDAS}
        r["rho_por_jogador_dentro_da_posicao"][fase] = {}
        for a, b in pares:
            rho = av.spearman_por_posicao(por_moeda[a], por_moeda[b])
            r["rho_por_jogador_dentro_da_posicao"][fase][f"{a}_{b}"] = {
                "media": rho["rho_media"], "minimo": rho["rho_min"], "n_jogadores": int(rho["posicoes"]["n"].sum()), "por_posicao": registos(rho["posicoes"])}
    for m in MOEDAS:
        r["parte_da_negada_total_por_fase"][m] = registos(a31.parte_da_negada_por_fase(n31.lances_por_moeda(T, m), ctx))
    return r


def etiqueta_de_fase(JG: dict, ctx: pd.DataFrame, jogo: int, frame: int) -> dict:
    """O que a SkillCorner atribui ao frame (`phases_of_play`): fase defensiva e de posse, a coluna de onde vem, a duração da fase, onde
    está a bola ao longo dela e a que distância da baliza de quem defende."""
    match, tabela, fases = JG[jogo]["match"], JG[jogo]["tabela"], JG[jogo]["fases"]
    comprimento, _ = a31.sc.dimensoes_campo(match)
    f = tabela[tabela["frame"] == frame].iloc[0]
    sentido = sentido_de_ataque(match, f["equipa_com_bola"], f["periodo"])
    linha = fases[(fases["frame_start"] <= frame) & (fases["frame_end"] >= frame)].iloc[0]
    fase_ctx = ctx[(ctx["jogo"] == jogo) & (ctx["frame"] == frame)].iloc[0]
    percurso = tabela[(tabela["frame"] >= linha["frame_start"]) & (tabela["frame"] <= linha["frame_end"])].groupby("frame")[["bola_x", "bola_y"]].first()
    percurso = percurso.iloc[:: max(len(percurso) // 12, 1)]
    dist = comprimento / 2 - percurso["bola_x"] * sentido  # distância à baliza de quem defende
    colunas_fase = [c for c in JG[jogo]["eventos"].columns if "phase" in c]
    ev = JG[jogo]["eventos"]
    ev_frame = ev[(ev["frame_start"] <= frame) & (ev["frame_end"] >= frame)] if {"frame_start", "frame_end"} <= set(ev.columns) else ev.iloc[:0]
    return {
        "jogo": jogo, "frame": frame, "equipa_com_bola": int(f["equipa_com_bola"]), "equipa_com_bola_nome": linha["team_in_possession_shortname"],
        "sentido_de_ataque": int(sentido), "bola_xy": [float(f["bola_x"]), float(f["bola_y"])],
        "bola_a_m_da_baliza_de_quem_defende": float(comprimento / 2 - f["bola_x"] * sentido),
        "bola_dentro_da_area_de_quem_defende": bool(ar.dentro_da_area_propria(f["bola_x"], f["bola_y"], -sentido, comprimento)),
        "fase_atribuida_pelo_frame": {"coluna_skillcorner": "team_out_of_possession_phase_type", "valor": fase_ctx["fase"],
                                      "coluna_do_nosso_contexto": "fase (agregacao.fase_defensiva_por_frame, por frame_start ≤ frame ≤ frame_end)"},
        "fase_de_posse_da_mesma_linha": linha["team_in_possession_phase_type"],
        "linha_de_phases_of_play": {c: (linha[c].item() if hasattr(linha[c], "item") else linha[c]) for c in (
            "index", "frame_start", "frame_end", "time_start", "time_end", "duration", "n_player_possessions_in_phase", "team_possession_loss_in_phase",
            "team_possession_lead_to_shot", "team_possession_lead_to_goal", "team_in_possession_phase_type", "team_out_of_possession_phase_type",
            "x_start", "x_end", "third_start", "third_end", "team_out_of_possession_length_start", "team_out_of_possession_length_end")},
        "bola_ao_longo_da_fase": [{"frame": int(i), "bola_x": float(r["bola_x"]), "bola_y": float(r["bola_y"]),
                                   "a_m_da_baliza_de_quem_defende": float(dist.loc[i])} for i, r in percurso.iterrows()],
        "colunas_de_fase_nos_eventos": colunas_fase,
        "eventos_com_o_frame": registos(ev_frame[["event_type"] + [c for c in colunas_fase if c in ev_frame.columns]]) if len(ev_frame) else [],
    }


def main() -> None:
    inicio = time.time()
    JG = ag.carregar_jogos()
    T = tabelas_das_tres_moedas(JG)
    ctx = n31.contextos(JG)
    r = {"definicao": {"moeda_risco": "negada_l = (1 − p_chegar_l) · (g_l + c_l); g_l = ganho de xT; c_l = xT do adversário no ponto de corte "
                       "(p(∅) = 1 na v3.1; ponto = onde o defesa com maior probabilidade de corte a corta com mais probabilidade, todos os defesas, "
                       "GR incluído, parâmetros da literatura); mesmo Shapley e mesma agregação; ameaça disponível da linha = g + c",
                       "indice_por_fase": "100 · (ameaça negada por momento do jogador na fase) / (média por momento de todos os jogadores na fase); "
                       f"percentil entre os jogadores com ≥ {ar.MIN_MOMENTOS} momentos na fase; um momento = um frame alinhado (0,2 s)"}}
    passos = {"moeda_risco": lambda: numeros_da_moeda_risco(JG, T), "etiqueta_de_fase_1996435_28188": lambda: etiqueta_de_fase(
        JG, ctx, FRAME_DA_ETIQUETA["jogo"], FRAME_DA_ETIQUETA["frame"]), "numeros_por_fase": lambda: numeros_por_fase(JG, T, ctx)}
    for nome, f in passos.items():
        t0 = time.time()
        r[nome] = f()
        print(f"  {nome}: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    indices = indices_das_moedas(juntar_jogos(JG, T, ctx))
    guardar_indices(indices)
    r["indice_por_fase"] = distribuicao_dos_indices(indices)
    print(f"  indice_por_fase: {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    r["casos"] = casos_concretos(JG, T, ctx)
    print(f"  casos: {time.time() - t0:.0f} s", flush=True)
    destino = SAIDA / "fase9c.json"
    destino.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    artigo = n31.ler_json("artigo_v31.json")
    artigo["fase_9c"] = r
    (SAIDA / "artigo_v31.json").write_text(json.dumps(artigo, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{destino} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script calcula a moeda «risco» na v3.1 (o valor de uma linha passa a ser o ganho de xT mais o perigo
# que o adversário teria se a bola fosse cortada onde a linha seria cortada), mostra o que muda quando esse ponto está dentro da área
# de quem tem a bola, dá a cada jogador um índice por fase defensiva (100 = média da fase, nas três moedas), desenha os casos concretos
# e verifica a etiqueta de fase de um momento. Não interpreta nada: cada número vem de uma tabela.
