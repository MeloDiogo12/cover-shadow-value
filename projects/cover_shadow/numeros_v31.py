"""Fase 9b · todos os números do artigo (v3.1) num só ficheiro: `outputs/cover_shadow/artigo_v31.json`.

Pressupõe as tabelas da v3.1 (`avaliar_variantes.py --versao v3_1`), a tabela da fase 3 e os resultados já calculados de 8b/8g/9a
(`modelo_escolha.json`, `xpass_v4.json`, `reprodutibilidade_v31.json`, que não se recalculam). Só números, sem leitura. Da raiz do projeto:
    uv run python projects/cover_shadow/numeros_v31.py
"""

from __future__ import annotations

import json
import time

import avaliacao as av
import avaliacao_v31 as a31
import figuras_agregados as ag
import gerar_candidatos as gc
import pandas as pd
from entradas import exigir
from figuras_9b import figura_momento
from metrica import agregacao
from metrica.parametros import PADRAO

from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

SAIDA = output_dir("cover_shadow")
VERSAO = "v3_1"
JOGOS = ag.JOGOS
C06 = {"jogo": 1925299, "frame": 12660}
C03 = {"jogo": 1996435, "frame": 39646}
W = {"W01": (1925299, 14478), "W02": (1925299, 50612), "W03": (1925299, 57378), "W04": (1996435, 45252),
     "W05": (1996435, 13394), "W06": (1925299, 38118), "W07": (1996435, 32452), "W08": (1996435, 38680)}
COMPRIMENTOS_C4 = {"comprimento_maximo_linha": av.LIMITES_COMPRIMENTO_V31}


def registos(t: pd.DataFrame) -> list[dict]:
    return json.loads(t.to_json(orient="records", force_ascii=False))


def tabelas_das_moedas(JG: dict) -> dict:
    """{jogo: {moeda: (linhas, atribuições)}} da base v3.1."""
    saida = {}
    for j in JOGOS:
        linhas, atribuicoes = av.ler_variante(j, "base", av.PASTAS[VERSAO])
        saida[j] = {"xt": (linhas, atribuicoes),
                    "linhas": a31.moeda_linhas(linhas, atribuicoes, a31.ultrapassados_guardados(j))}
    return saida


def jogadores_por_moeda(JG: dict, T: dict, moeda: str, presenca_de=None) -> pd.DataFrame:
    """Ameaça negada por jogador (D17) dos dois jogos na moeda dada; `presenca_de(jogo)` troca os frames de presença (janelas)."""
    partes = []
    for j in JOGOS:
        presenca = JG[j]["defesas"][["frame", "posse_id", "jogador"]] if presenca_de is None else presenca_de(j)
        partes.append(av.negada_por_jogador(*T[j][moeda], presenca, JG[j]["jogadores"]).assign(jogo=j))
    return pd.concat(partes, ignore_index=True)


def lances_por_moeda(T: dict, moeda: str) -> pd.DataFrame:
    return pd.concat([agregacao.por_lance(T[j][moeda][0]).assign(jogo=j) for j in JOGOS], ignore_index=True)


def contextos(JG: dict) -> pd.DataFrame:
    return pd.concat([agregacao.contexto_dos_frames(JG[j]["tabela"], JG[j]["fases"]).assign(jogo=j) for j in JOGOS], ignore_index=True)


def segunda_leitura(JG: dict, T: dict) -> dict:
    """§2: % negada por fase defensiva e por terço nas duas moedas (com o nº de momentos), jogadores lado a lado, ρ entre moedas, C06 e C03."""
    ctx = contextos(JG)
    r = {"fase_e_terco": {}, "totais": {}}
    for moeda in a31.MOEDAS:
        lances = lances_por_moeda(T, moeda)
        r["fase_e_terco"][moeda] = {por: registos(a31.pct_por_contexto(lances, ctx, por)) for por in ("fase", "terco")}
        av_ = lances[lances["n_avaliadas"] > 0]
        r["totais"][moeda] = {"lances": len(av_), "disponivel": float(av_["disponivel"].sum()), "negada": float(av_["negada"].sum()),
                              "pct_negada": float(100 * av_["negada"].sum() / av_["disponivel"].sum())}
    xt, linhas = jogadores_por_moeda(JG, T, "xt"), jogadores_por_moeda(JG, T, "linhas")
    lado = xt.merge(linhas, on=["jogo", "jogador", "posicao", "nome", "equipa"], suffixes=("_xt", "_linhas"))
    r["jogadores"] = registos(lado[["jogo", "jogador", "nome", "equipa", "posicao", "posses_xt", "negada_xt", "pct_negada_xt",
                                    "negada_linhas", "pct_negada_linhas"]].sort_values(["jogo", "negada_xt"], ascending=[True, False]))
    rho = av.spearman_por_posicao(xt, linhas)
    r["rho_entre_moedas"] = {"media": rho["rho_media"], "minimo": rho["rho_min"], "por_posicao": registos(rho["posicoes"])}
    nomes = {j: JG[j]["jogadores"].set_index("jogador")["nome"].to_dict() for j in JOGOS}
    for nome, lance in (("C06", C06), ("C03", C03)):
        j, f = lance["jogo"], lance["frame"]
        r[nome] = {moeda: a31.negada_por_defesa_no_lance(*[T[j][moeda][0], T[j][moeda][1]], f, nomes[j]) for moeda in a31.MOEDAS}
    return r


def momento_da_decisao(JG: dict, T: dict, moedas: tuple = a31.MOEDAS) -> dict:
    """§3: média só dos momentos dos últimos 1 s e 2 s antes do fim da posse individual, em vez da posse inteira."""
    fim = {j: a31.fim_das_posses(JG[j]["eventos"]) for j in JOGOS}
    janelas = {s: (lambda j, s=s: a31.presenca_na_janela(JG[j]["defesas"][["frame", "posse_id", "jogador"]], fim[j], s)) for s in a31.JANELAS_S}
    r = {"definicao": "frames alinhados (0,2 s) com frame >= frame_end da posse individual − janela (10 frames de tracking por segundo)",
         "posses_por_tipo_de_fim": {}, "rho_por_posicao": {}, "sobem_e_descem": {}}
    for j in JOGOS:
        frames = JG[j]["tabela"][["posse_id"]].drop_duplicates()
        r["posses_por_tipo_de_fim"][str(j)] = a31.posses_por_tipo_de_fim(
            frames, fim[j], {s: janelas[s](j) for s in a31.JANELAS_S})
    for moeda in moedas:
        atual = jogadores_por_moeda(JG, T, moeda)
        for s in a31.JANELAS_S:
            janela = jogadores_por_moeda(JG, T, moeda, janelas[s])
            rho = av.spearman_por_posicao(atual, janela)
            chave = f"{moeda}_{s:g}s"
            r["rho_por_posicao"][chave] = {"media": rho["rho_media"], "minimo": rho["rho_min"], "jogadores": len(janela)}
            r["sobem_e_descem"][chave] = a31.sobem_e_descem(atual, janela)
    return r


def moedas_por_fase(JG: dict, T: dict) -> dict:
    """Comparação das duas moedas dentro de cada fase: ρ por jogador (dentro da posição) em bloco alto, médio e baixo; parte da ameaça
    negada total por fase em cada moeda; os 2 + 2 momentos de bloco alto ou médio em que mais discordam (com imagem de cada um)."""
    ctx = contextos(JG)
    r = {"rho_por_jogador_dentro_da_posicao": {}, "parte_da_negada_total_por_fase": {}}
    for fase in a31.BLOCOS:
        por_moeda = {}
        for moeda in a31.MOEDAS:
            por_moeda[moeda] = jogadores_por_moeda(JG, T, moeda, lambda j, fase=fase: a31.presenca_na_fase(
                JG[j]["defesas"][["frame", "posse_id", "jogador"]], ctx[ctx["jogo"] == j], fase))
        rho = av.spearman_por_posicao(por_moeda["xt"], por_moeda["linhas"])
        p = rho["posicoes"]
        r["rho_por_jogador_dentro_da_posicao"][fase] = {
            "media": rho["rho_media"], "minimo": rho["rho_min"], "n_jogadores": int(p["n"].sum()), "n_posicoes": len(p),
            "por_posicao": registos(p)}
    lances = {m: lances_por_moeda(T, m) for m in a31.MOEDAS}
    for moeda in a31.MOEDAS:
        r["parte_da_negada_total_por_fase"][moeda] = registos(a31.parte_da_negada_por_fase(lances[moeda], ctx))
        g = a31.parte_da_negada_por_fase(lances[moeda], ctx).set_index("fase")["pct_da_negada_total"]
        r["parte_da_negada_total_por_fase"][f"{moeda}_bloco_alto_mais_medio"] = float(g.reindex(["high_block", "medium_block"]).sum())
    fases = ("high_block", "medium_block")
    escolhidos = a31.momentos_que_mais_discordam(lances["xt"], lances["linhas"], ctx, fases)
    brand = load_brand()
    apply_theme(brand)
    pasta = SAIDA / "momentos_moedas"
    r["momentos_que_mais_discordam"] = {}
    for grupo, t in escolhidos.items():
        r["momentos_que_mais_discordam"][grupo] = []
        for k, m in t.iterrows():
            j, f = int(m["jogo"]), int(m["frame"])
            nomes = JG[j]["jogadores"].set_index("jogador")["nome"].to_dict()
            valores = {mo: a31.negada_por_defesa_no_lance(*T[j][mo], f, nomes) for mo in a31.MOEDAS}
            rotulo = f"{'xT alto, linhas baixo' if grupo == 'xt_alto' else 'linhas alto, xT baixo'} · {k + 1}"
            nome_img = f"{grupo}_{k + 1}_{j}_{f}"
            figura_momento(gc.carregar(j), f, rotulo, valores, {"xt": m["percentil_xt"], "linhas": m["percentil_linhas"]}, brand,
                           pasta / f"{nome_img}.png")
            r["momentos_que_mais_discordam"][grupo].append({
                "jogo": j, "frame": f, "fase": m["fase"], "imagem": f"outputs/cover_shadow/momentos_moedas/{nome_img}.png",
                "xt": valores["xt"], "linhas": valores["linhas"],
                "percentil_xt": float(m["percentil_xt"]), "percentil_linhas": float(m["percentil_linhas"])})
    return r


def avaliacao_c3(JG: dict) -> dict:
    partes = []
    for j in JOGOS:
        linhas = av.ler_variante(j, "base", av.PASTAS[VERSAO])[0]
        com_linha = av.passes_com_linha(av.passes_do_jogo(JG[j]["eventos"]), linhas)
        partes.append(av.com_comparaveis(com_linha, linhas).assign(jogo=j))
    passes = pd.concat(partes, ignore_index=True)
    r = av.comparar_com_comparaveis(passes)
    baixa = passes[passes["p_chegar"] < 0.5]
    taxa, inf, sup = av.intervalo_wilson(int(baixa["completo"].sum()), len(baixa))
    return {**r, "n_baixa": len(baixa), "completos_baixa": int(baixa["completo"].sum()), "taxa_baixa": taxa,
            "taxa_baixa_ic95": [inf, sup]}


def avaliacao_c4(JG: dict) -> list[dict]:
    base = ag.jogadores_da_variante(JG, "base", VERSAO)
    base_lances = ag.lances_da_variante("base", VERSAO)
    parametros = {**av.VALORES_ANTIGOS_V3, "raio_bloqueio_corpo": av.RAIOS_BLOQUEIO_V3, **COMPRIMENTOS_C4}
    saida = []
    for parametro, valores in parametros.items():
        for valor in valores:
            nome = av.nome_da_variante(parametro, valor)
            variante = ag.jogadores_da_variante(JG, nome, VERSAO)
            rho = av.spearman_por_posicao(base, variante)
            saida.append({"parametro": parametro, "valor": valor, "variante": nome, "rho_jogador": rho["rho_media"],
                          "rho_jogador_minimo": rho["rho_min"],
                          "rho_lance": av.spearman_por_lance(base_lances, ag.lances_da_variante(nome, VERSAO)),
                          **{f"pct_{j}": av.pct_negada(ag.ler(j, nome, VERSAO)[0]) for j in JOGOS}})
    return saida


def avaliacao_c5(JG: dict) -> dict:
    n = ag.numeros_c5(JG, VERSAO)
    return {"rho_media": n["rho"]["rho_media"], "rho_minimo": n["rho"]["rho_min"], "pct_mudam": n["pct_mudam"], "linhas": n["linhas"],
            "pct_negada_tronco_bola": {str(j): list(v) for j, v in n["pct_negada"].items()}, "jogadores": len(n["jogadores"])}


def lances_w(JG: dict, T: dict) -> dict:
    """C1/C2: valores dos lances W01–W08 na v3.1 (e na v3, para ver o que muda), sem nova ronda."""
    nomes = {j: JG[j]["jogadores"].set_index("jogador")["nome"].to_dict() for j in JOGOS}
    saida = {}
    for lance, (j, f) in W.items():
        v3_l, v3_a = av.ler_variante(j, "base", av.PASTAS["v3"])
        saida[lance] = {"jogo": j, "frame": f, "v3": a31.negada_por_defesa_no_lance(v3_l, v3_a, f, nomes[j]),
                        **{f"v3_1_{m}": a31.negada_por_defesa_no_lance(*T[j][m], f, nomes[j]) for m in a31.MOEDAS},
                        "linhas_de_mais_de_35m_no_lance": int(
                            ((T[j]["xt"][0]["frame"] == f) & (T[j]["xt"][0]["motivo"] == "mais_de_35m")).sum())}
    return saida


def distribuicao_por_comprimento(JG: dict, T: dict) -> dict:
    """Ameaça negada (xT) por banda de comprimento da linha, v3 contra v3.1, dois jogos juntos."""
    v3 = pd.concat([av.adicionar_comprimento(av.ler_variante(j, "base", av.PASTAS["v3"])[0], JG[j]["tabela"]) for j in JOGOS])
    v31 = pd.concat([T[j]["xt"][0] for j in JOGOS])
    return {"v3": registos(a31.negada_por_comprimento(v3)), "v3_1": registos(a31.negada_por_comprimento(v31))}


def ler_json(nome: str) -> dict:
    ficheiro = SAIDA / nome
    return json.loads(ficheiro.read_text(encoding="utf-8")) if ficheiro.exists() else {}


def c8_e_xpass() -> dict:
    """C8 e o xPass da 9a, tal como já calculados (não se recalcula nada)."""
    m, x = ler_json("modelo_escolha.json"), ler_json("xpass_v4.json")
    c8 = {"coeficientes_M3": m.get("coeficientes_conjunto", {}).get("M3"), "n_passes": m.get("passes_por_jogo"),
          "log_verosimilhanca_conjunto": m.get("log_verosimilhanca"), "ao_acaso_conjunto": m.get("ao_acaso_conjunto"),
          "fora_da_amostra": m.get("fora_da_amostra")}
    xp = {"K1": x.get("criterios", {}).get("K1"), "K2": x.get("criterios", {}).get("K2"), "veredicto": x.get("criterios", {}).get("veredicto"),
          "metricas_20_jogos": x.get("xpass", {}).get("metricas_juntas"), "n_passes_20_jogos": x.get("xpass", {}).get("n_passes"),
          "gamma_dois_jogos": x.get("sombra", {}).get("gamma_dois_jogos"), "direcoes": {
              k: {c: v.get(c) for c in ("treino", "teste", "n_teste", "gamma_tronco", "gamma_bola")}
              for k, v in x.get("sombra", {}).get("direcoes", {}).items()},
          "log_loss_com_e_sem_sombra": x.get("sombra", {}).get("metricas_juntas"),
          "tronco_menos_bola_log_loss": x.get("sombra", {}).get("tronco_menos_bola_log_loss")}
    return {"c8": c8, "xpass": xp}


def confirmar_v3_1_literatura(T: dict) -> dict:
    """§1: a pasta `v3_1_literatura/` (D40: com o fundo `p_fundo`) não é «v3 + 35 m»; mostra-se a diferença de totais na moeda xT."""
    saida = {}
    for j in JOGOS:
        lit = pd.read_parquet(exigir(a31.PASTA_SAIDA / "v3_1_literatura" / f"{j}_v3_1_xt_linhas.parquet",
                                 f"calcular_xpass.py --literatura --jogo {j}"))
        base = T[j]["xt"][0]
        ok = base[~base["excluida"]]
        saida[str(j)] = {"linhas_avaliadas_v3_1": len(ok), "linhas_avaliadas_literatura": int((~lit["excluida"]).sum()),
                         "negada_v3_1": float(ok["ameaca_negada"].sum()), "negada_literatura": float(lit["negada_linha"].sum()),
                         "disponivel_v3_1": float(ok["valor_xt"].sum()), "disponivel_literatura": float(lit["disponivel_linha"].sum()),
                         "p_fundo_media_literatura": float(lit.loc[~lit["excluida"], "p_fundo"].mean())}
    return saida


def main() -> None:
    inicio = time.time()
    JG = ag.carregar_jogos()
    T = tabelas_das_moedas(JG)
    passos = {"v3_1_literatura_vs_definicao": lambda: confirmar_v3_1_literatura(T), "segunda_leitura_moedas": lambda: segunda_leitura(JG, T),
              "momento_da_decisao": lambda: momento_da_decisao(JG, T), "moedas_por_fase": lambda: moedas_por_fase(JG, T), "c3": lambda: avaliacao_c3(JG),
              "c4": lambda: avaliacao_c4(JG), "c5": lambda: avaliacao_c5(JG), "lances_w01_w08": lambda: lances_w(JG, T),
              "negada_por_comprimento": lambda: distribuicao_por_comprimento(JG, T), "ja_calculado": c8_e_xpass}
    r = {"definicao": {"versao": "v3.1 (D43)", "regra": "v3 + exclusão das linhas com mais de 35 m (D37)",
                       "comprimento_maximo_linha_m": PADRAO.comprimento_maximo_linha,
                       "valor_da_sombra": "índice de fecho da linha, não probabilidade calibrada de o passe chegar"}}
    for nome, f in passos.items():
        t0 = time.time()
        r[nome] = f()
        print(f"  {nome}: {time.time() - t0:.0f} s", flush=True)
    r["c6"] = ler_json("reprodutibilidade_v31.json")
    destino = SAIDA / "artigo_v31.json"
    destino.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"{destino} ({time.time() - inicio:.0f} s)")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script junta num só ficheiro todos os números que o artigo usa para a v3.1: onde se nega mais ameaça
# nas duas moedas, se a ordem dos jogadores muda quando só se conta o fim da posse, como se comportam os critérios de avaliação
# (C3 a C6, C8 e o xPass) e como se distribui a ameaça pelo comprimento das linhas. Não interpreta nada: cada número vem de uma tabela.
