"""Fase 6 · gera os candidatos a lance de exemplo (v3, dois jogos) em outputs/cover_shadow/candidatos/.

Da raiz do projeto:
    uv run python projects/cover_shadow/gerar_candidatos.py

Critérios (por frame alinhado, definição v3): quem tem a bola está na 1.ª ou 2.ª fase de construção;
nenhuma linha excluída por falta de pose, nem por cruzamento, e fora da zona de cruzamento (D27); pelo menos 3 linhas
avaliadas; ameaça negada de pelo menos `NEGADA_MINIMA` xT e `PCT_NEGADA_MINIMA`% da disponível, vinda de 1 a 3 defesas, sem mais de 2 sombras na mesma
linha; nenhuma marcação (de um defesa que afete uma linha) perto de um limite (D33). Entre os que cumprem fica o
frame de maior diferença corpo vs bola de cada posse, e dos 10 de maior diferença; a lista final ordena-se só pelo
minuto de jogo. Escreve a figura e o clip de cada um, `candidatos.csv` e `candidatos.md`.
"""

from __future__ import annotations

import avaliacao as av
import pandas as pd
import revisao as rev
from candidatos import (
    FASES_DE_CONSTRUCAO,
    ROTULOS_FASE_POSSE,
    defesas_com_mais_valor,
    defesas_que_afetam_alguma_linha,
    frames_concentrados,
    marcacao_estavel,
    minuto_do_frame,
    perto_de_um_limiar,
)
from metrica import agregacao
from metrica.cruzamento import em_zona_de_cruzamento
from metrica.parametros import PADRAO
from metrica.pipeline import carregar_tabela, preparar_defesas

from footy.ingest import skillcorner as sc
from footy.paths import output_dir
from footy.viz.theme import load_brand

FONTE = "SkillCorner Open Data (MIT)"
N_CANDIDATOS = 10
NEGADA_MINIMA = 0.005  # xT no frame (na construção o ganho de xT é pequeno: o limiar é baixo)
PCT_NEGADA_MINIMA = 25.0  # % da ameaça disponível nas linhas avaliadas do frame
JANELA_CLIP = 20  # frames de tracking para cada lado: ~4 s
ROTULOS_FASE_DEFESA = {"high_block": "bloco alto", "medium_block": "bloco médio", "low_block": "bloco baixo",
                       "chaotic": "caótica", "defending_direct": "defesa de jogo direto",
                       "defending_set_play": "bola parada", "defending_quick_break": "contra-ataque",
                       "defending_transition": "transição defensiva"}


def pt(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def carregar(jogo: int) -> dict:
    """Tudo o que os filtros e as figuras precisam de um jogo (v3)."""
    match = sc.ler_match(jogo)
    tabela = carregar_tabela(jogo)
    jogadores = pd.DataFrame([{"jogador": p["id"], "nome": p["short_name"], "posicao": p["player_role"]["acronym"]}
                              for p in match["players"]])
    fases = sc.ler_fases(jogo)
    linhas, atrib = av.ler_variante(jogo, "base", av.PASTA_AVALIACAO_V3)
    linhas_bola, atrib_bola = av.ler_variante(jogo, "bola", av.PASTA_AVALIACAO_V3)
    return {"jogo": jogo, "match": match, "tabela": tabela, "defesas": preparar_defesas(tabela, match, PADRAO),
            "campo": sc.dimensoes_campo(match), "nomes": jogadores.set_index("jogador")["nome"].to_dict(),
            "posicoes": jogadores.set_index("jogador")["posicao"].to_dict(), "fases": fases,
            "linhas": linhas, "atrib": atrib, "linhas_bola": linhas_bola, "atrib_bola": atrib_bola,
            "contexto": agregacao.contexto_dos_frames(tabela, fases),
            "titulo": f"{match['home_team']['short_name']} – {match['away_team']['short_name']}"}


def marcacao_no_limiar(J: dict, frame: int) -> bool:
    """True se, neste frame, algum defesa que pode afetar uma linha está perto de um limite da marcação."""
    d = J["defesas"][J["defesas"]["frame"] == frame]
    jogadores = J["tabela"][J["tabela"]["frame"] == frame]
    portador = jogadores[jogadores["jogador"] == jogadores["portador"]].iloc[0]
    colegas = jogadores[(jogadores["equipa"] == portador["equipa"]) & (jogadores["jogador"] != portador["jogador"])]
    afeta = defesas_que_afetam_alguma_linha(d, (portador["x"], portador["y"]), list(zip(colegas["x"], colegas["y"])), PADRAO)
    return bool((perto_de_um_limiar(d, PADRAO) & afeta).any())


def frames_elegiveis(J: dict) -> pd.DataFrame:
    """Frames que cumprem os critérios fixos (tudo menos a estabilidade da marcação no clip), com os números
    de cada um e a diferença corpo vs bola. Imprime quantos sobram a cada critério."""
    lances = agregacao.por_lance(J["linhas"]).merge(J["contexto"], on="frame", how="left")
    n = {"frames com linhas": len(lances)}
    lances = lances[lances["fase_posse"].isin(FASES_DE_CONSTRUCAO)]
    n["1.ª ou 2.ª fase de construção"] = len(lances)
    lances = lances[(lances["n_sem_pose"] == 0) & (lances["n_avaliadas"] >= 3)]
    n["sem linhas excluídas por falta de pose"] = len(lances)
    sem_cruzamento = [(c == 0) and not em_zona_de_cruzamento(cor, t)
                      for c, cor, t in zip(lances["n_cruzamento"], lances["corredor"], lances["terco"])]
    lances = lances[sem_cruzamento]
    n["sem cruzamento nem zona de cruzamento"] = len(lances)
    lances = lances[(lances["negada"] >= NEGADA_MINIMA) & (lances["pct_negada"] >= PCT_NEGADA_MINIMA)]
    n[f"ameaça negada >= {NEGADA_MINIMA} xT e >= {PCT_NEGADA_MINIMA:g}% da disponível"] = len(lances)
    lances = lances[lances["frame"].isin(frames_concentrados(J["atrib"]))]
    n["1 a 3 defesas, até 2 sombras por linha"] = len(lances)
    lances = lances[[not marcacao_no_limiar(J, f) for f in lances["frame"]]]
    n["nenhuma marcação (que afete uma linha) perto de um limite"] = len(lances)
    mudancas = av.linhas_que_mudam(J["linhas"], J["linhas_bola"], J["atrib"], J["atrib_bola"])
    disc = av.discordancia_por_lance(mudancas)
    lances = lances.merge(disc, on="frame", how="left").fillna({"discordancia": 0.0, "linhas_que_mudam": 0})
    negada_bola = agregacao.por_lance(J["linhas_bola"]).set_index("frame")["negada"]
    lances["negada_bola"] = lances["frame"].map(negada_bola)
    print(J["titulo"], n)
    return lances.assign(jogo=J["jogo"])


def escolher(elegiveis: pd.DataFrame) -> pd.DataFrame:
    """Um frame por posse (o de maior diferença corpo vs bola) e, por essa ordem de diferença, os primeiros
    `N_CANDIDATOS`."""
    melhores = elegiveis.sort_values("discordancia", ascending=False).drop_duplicates(["jogo", "posse_id"])
    return melhores.head(N_CANDIDATOS)


def descrever(r, J: dict) -> dict:
    """Uma linha da tabela de um candidato."""
    minuto = minuto_do_frame(J["fases"], r.frame)
    valor = defesas_com_mais_valor(J["atrib"], r.frame)
    quem = "; ".join(f"{J['nomes'][d]} ({J['posicoes'][d]}) {pt(v, 3)}" for d, v in zip(valor["defesa"], valor["negada"]))
    periodo = int(J["tabela"].loc[J["tabela"]["frame"] == r.frame, "periodo"].iloc[0])
    estavel = marcacao_estavel(J["defesas"], rev.frames_do_clip(J["tabela"], r.frame, JANELA_CLIP))
    return {"jogo": J["jogo"], "partida": J["titulo"], "parte": periodo, "minuto": minuto, "frame": r.frame,
            "fase_de_construcao": ROTULOS_FASE_POSSE[r.fase_posse],
            "fase_defensiva": ROTULOS_FASE_DEFESA.get(r.fase, r.fase),
            "negada_xt": r.negada, "defesas_com_mais_valor": quem, "negada_bola_xt": r.negada_bola,
            "dif_corpo_menos_bola_xt": r.negada - r.negada_bola, "discordancia_xt": r.discordancia,
            "linhas_que_mudam": int(r.linhas_que_mudam), "marcacao_igual_no_clip": "sim" if estavel else "não"}


def main() -> None:
    brand = load_brand()
    jogos = {j: carregar(j) for j in sc.JOGOS_COM_POSE}
    elegiveis = pd.concat([frames_elegiveis(J) for J in jogos.values()], ignore_index=True)
    escolhidos = escolher(elegiveis)
    tabela = pd.DataFrame([descrever(r, jogos[r.jogo]) for r in escolhidos.itertuples()])
    tabela = tabela.sort_values(["minuto", "jogo"], ignore_index=True)  # só por minuto de jogo
    tabela.insert(0, "candidato", [f"C{i + 1:02d}" for i in range(len(tabela))])
    pasta = output_dir("cover_shadow") / "candidatos"
    for r in tabela.itertuples():
        J = jogos[r.jogo]
        nome = f"{r.candidato}_{r.jogo}_{r.frame}"
        titulo = f"{J['titulo']} · {r.parte}.ª parte · minuto {r.minuto} · {r.fase_de_construcao}"
        rev.guardar_figura(J, r.frame, r.candidato, titulo, brand, FONTE, pasta, nome)
        rev.guardar_clip(J, r.frame, r.candidato, titulo, brand, FONTE, pasta, nome, JANELA_CLIP)
    tabela.to_csv(pasta / "candidatos.csv", index=False, sep=";", encoding="utf-8-sig")
    escrever_markdown(tabela, pasta / "candidatos.md")
    print(tabela.drop(columns=["partida"]).to_string(index=False))


def escrever_markdown(t: pd.DataFrame, destino) -> None:
    nota = ("Ordenados só por minuto de jogo. Ameaça negada em xT; diferença = tronco menos bola (cone virado para a bola). "
            "\"Marcação igual\" = o conjunto de defesas a marcar não muda durante os ~4 s do clip.")
    cabecalho = ("| # | Jogo | Parte · minuto | Fase de construção (fase defensiva) | Negada | Defesas com mais valor (xT) "
                 "| Bola | Diferença | Marcação igual |")
    linhas = ["# Candidatos a lance de exemplo (v3)", "", nota, "", cabecalho, "|---|---|---|---|---|---|---|---|---|"]
    for r in t.itertuples():
        linhas.append(f"| {r.candidato} | {r.partida} | {r.parte}.ª · {r.minuto}' | {r.fase_de_construcao} "
                      f"({r.fase_defensiva}) | {pt(r.negada_xt, 3)} | {r.defesas_com_mais_valor} | "
                      f"{pt(r.negada_bola_xt, 3)} | {pt(r.dif_corpo_menos_bola_xt, 3)} | {r.marcacao_igual_no_clip} |")
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

# Em linguagem corrente: este script percorre todos os instantes dos dois jogos e deixa só aqueles em
# que o adversário está a construir, todos os defesas relevantes têm dados do corpo, a ameaça negada é
# clara e vem de poucos defesas, e nenhuma marcação está no limite da regra. Desses, mostra os dez em
# que o corpo mais muda a sombra, por ordem de minuto de jogo, cada um com figura e clip de 4 segundos.
