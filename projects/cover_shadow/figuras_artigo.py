"""Fase 6 · figuras finais do artigo e legendas, em outputs/cover_shadow/artigo/.

Da raiz do repositório:
    uv run python projects/cover_shadow/figuras_artigo.py            # tudo
    uv run python projects/cover_shadow/figuras_artigo.py --so lances # só os lances (C06 e C03)
    uv run python projects/cover_shadow/figuras_artigo.py --so agregados

Lances: C06 (Brisbane–Perth, 1.ª parte, 21') em 7 passos + resumo e corpo vs bola; C03 (Sydney–Adelaide, 2.ª parte,
19'). Agregados: dois jogos, definição v3.1 (ver `figuras_agregados`); novas da fase 9b em `figuras_9b`. Os números de cada figura ficam em
`numeros_das_figuras.json` e a legenda de cada uma em `legendas.md`.
"""

from __future__ import annotations

import argparse
import json

import figuras_9b
import figuras_agregados as ag
import figuras_lance as fl
import gerar_candidatos as gc
from orientacao_defesas import acrescentar_a_legendas

from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

C06 = {"jogo": 1925299, "frame": 12660, "rotulo": "C06 · 1.ª parte · minuto 21"}
C03 = {"jogo": 1996435, "frame": 39646, "rotulo": "C03 · 2.ª parte · minuto 19"}
NB = "notebook 05"


def pt(v: float, casas: int = 1) -> str:
    return ag.pt(v, casas)


def lances(brand, pasta) -> dict:
    n = {}
    J6 = gc.carregar(C06["jogo"])
    n["c06_passos"] = fl.c06_passo_a_passo(J6, C06["frame"], C06["rotulo"], brand, pasta)
    n["c06_corpo_bola"] = fl.c06_corpo_vs_bola(J6, C06["frame"], C06["rotulo"], brand, pasta)
    J3 = gc.carregar(C03["jogo"])
    n["c03"] = fl.c03_lance(J3, C03["frame"], C03["rotulo"], brand, pasta, "Klimala", "Isaias")
    return n


def agregados(brand, pasta) -> dict:
    JG = ag.carregar_jogos()
    ag.JG_TITULOS.update({j: JG[j]["titulo"] for j in JG})
    n = {}
    n["fase_terco"] = ag.figura_fase_e_terco(JG, brand, pasta / "agregado_fase_e_terco.png")
    n["revisoes"] = ag.figura_revisoes(brand, pasta / "agregado_revisoes.png")
    n["versoes"] = ag.figura_versoes(brand, pasta / "agregado_versoes.png")
    n["c3"] = ag.figura_c3(JG, brand, pasta / "agregado_c3_sombra_escolhida.png")
    c4 = ag.figura_c4(JG, brand, pasta / "agregado_c4_estabilidade.png")
    n["c4"] = c4.to_dict("records")
    n["c5"] = ag.figura_c5(JG, brand, pasta / "agregado_c5_tronco_vs_bola.png")
    n["novas"] = figuras_9b.todas(JG, brand, pasta)
    return n


def legendas(n: dict) -> str:
    """Legenda curta e fonte de cada figura, com os números que nela aparecem."""
    risco = n.get("moeda") == "risco"
    U = "risco" if risco else "xT"  # unidade dos valores nas legendas
    L = ["# Legendas das figuras do artigo", "",
         ("Interno, não publicar. Cada entrada: legenda curta, números da figura e de onde vêm. Fonte de todas as figuras: "
         "SkillCorner Open Data (MIT), dois jogos com body pose (Brisbane Roar – Perth Glory, 21/12/2024; Sydney FC – Adelaide United, "
         "01/02/2025), definição v3.1 da métrica (v3 sem as linhas com mais de 35 m). "
         + ("Valores em risco, a moeda oficial (D44): valor da linha = ganho de xT + xT do adversário no ponto onde a linha seria cortada; "
            "as figuras em xT estão em `artigo_v31_xt/`. " if risco else "Valores em xT (ganho de xT), salvo indicação. ")
         + "O valor da sombra é um índice de fecho da linha, não uma probabilidade calibrada de o passe chegar (D43)."), ""]
    if "c06_passos" in n:
        p = n["c06_passos"]
        s = p[8]["shapley"]
        partes = ", ".join(f"{k} {pt(v, 4)}" for k, v in s.items())
        total, perigo = p[8]["negada_total_xt"], p[8]["perigo_total_xt"]
        base = ("Brisbane Roar – Perth Glory, 1.ª parte, minuto 21 (frame 12 660). Fonte: tracking, body pose e eventos SkillCorner; "
                "cálculo em `metrica.linhas` (o mesmo do notebook 04). Candidato C06 de `candidatos.md`.")
        L += ["## C06 · passo a passo (7 figuras e resumo)", "", base, ""]
        textos = [
            ("c06_passo1_linhas_de_passe.png", (f"Passo 1. Linhas do portador (Pennington) para os {p[1]['linhas']} colegas a 35 m ou menos (as linhas mais longas não se avaliam, D43); espessura = {'valor da linha (ganho de xT + xT do adversário no ponto de corte)' if risco else 'perigo da linha (ganho de xT)'}. "
             f"Perigo total das linhas: {pt(perigo, 3)} {U}. {fl.NOTA_SOMA}")),
            ("c06_passo2_quem_marca.png", (f"Passo 2. Só os defesas que acompanham um atacante não projetam sombra: "
             f"{', '.join(p[2]['marcam'])} ({len(p[2]['marcam'])} de 10 defesas, sem o guarda-redes). "
             "Quem pressiona o portador não marca e conta para a sombra (D20, D28).")),
            ("c06_passo3_quem_pode_afetar_cada_linha.png", (f"Passo 3. Defesas que podem afetar alguma linha ({p[3]['podem_afetar']} de 10) e quantos afetam cada linha. "
             f"Linhas excluídas neste lance: {p[3]['excluidas']}.")),
            ("c06_passo4_cones_pelo_tronco.png", f"Passo 4. Cone de {pt(60, 0)}° e {pt(10, 0)} m por trás do tronco de cada defesa que não marca; seta = frente do tronco."),
            ("c06_passo5_probabilidade_de_o_passe_chegar.png", (f"Passo 5. Índice de chegada {'na única linha' if len(p[5]['p_com_sombra']) == 1 else 'nas ' + str(len(p[5]['p_com_sombra'])) + ' linhas'} com sombra (de {p[5]['avaliadas']} avaliadas): "
             f"{', '.join(pt(v, 2) for v in p[5]['p_com_sombra'])}. Laranja = com sombra (índice abaixo de 0,95), "
             f"cinzento = sem sombra; {p[5]['cruzam_cone'] - len(p[5]['p_com_sombra'])} linha cruza um cone mas com "
             "índice ≥ 0,95, por isso fica cinzenta. Lê-se como índice, não como probabilidade calibrada.")),
            ("c06_passo6_ameaca_negada_por_linha.png", (f"Passo 6. Ameaça negada em cada linha com sombra (laranja) face ao perigo da linha (cinzento). "
             f"Total negado: {pt(total, 4)} {U} de {pt(perigo, 3)} {U} disponíveis. {fl.NOTA_SOMA}")),
            ("c06_passo7_ameaca_negada_por_defesa.png", f"Passo 7. Repartição de Shapley da ameaça negada: {partes}. Soma: {pt(total, 4)} {U}. {fl.NOTA_SOMA}"),
            ("c06_passo8_resumo.png", f"Resumo. Ameaça negada {pt(total, 4)} {U} ({pt(100 * total / perigo, 1)} % do perigo disponível); {partes}. {fl.NOTA_SOMA}"),
        ]
        for ficheiro, texto in textos:
            L += [f"### {ficheiro}", "", texto, "", f"Números: lance C06; `metrica.linhas.avaliar_frame`; ver `{NB}`, secção 1 (desenho dos lances) para o formato.", ""]
        c = n["c06_corpo_bola"]
        t, b = c["Cone pelo tronco"], c["Cone virado para a bola"]
        pt_t = ", ".join(f"{k} {pt(v, 4)}" for k, v in t["por_defesa"].items())
        pt_b = ", ".join(f"{k} {pt(v, 4)}" for k, v in b["por_defesa"].items())
        L += ["### c06_corpo_vs_bola.png", "",
              (f"C06 com o cone pelo tronco (a métrica) e com o cone virado para o portador. Ameaça negada: {pt(t['total_xt'], 4)} {U} "
              f"({pt_t}) contra {pt(b['total_xt'], 4)} {U} ({pt_b}). {fl.NOTA_SOMA}"), "",
              f"Números: `pipeline.preparar_defesas(..., orientacao=\"bola\")`; é a comparação da secção 4 do {NB} (C5) aplicada a um lance.", ""]
    if "c03" in n:
        c = n["c03"]
        L += ["## C03", "", "### c03_lance.png", "",
              (f"Sydney FC – Adelaide United, 2.ª parte, minuto 19 (frame 39 646). O cone por trás do tronco do Klimala cobre a linha do portador (Vriends) "
              f"para o Isaías: índice de chegada {pt(c['p_chegar'], 2)}; valor da linha {pt(c['valor_xt'], 4)} {U}, "
              f"ameaça negada na linha {pt(c['negada_linha'], 4)} {U} (só o Klimala); ameaça negada no lance {pt(c['negada_lance'], 4)} {U}. {fl.NOTA_SOMA}"), "",
              "Números: candidato C03 de `candidatos.md`" + ("; `artigo_v31.json`, secção `risco.c03`." if risco else " (0,007 xT no lance)."), ""]
    if "fase_terco" in n and risco:
        ROT = ag.ROTULOS_CONTEXTO
        ft = n["fase_terco"]
        L += ["## Agregados", "", "### agregado_fase_e_terco.png", "",
              ("Ameaça negada em risco, por fase defensiva e por terço da posse, dois jogos juntos, v3.1: % da ameaça disponível (g + c) que se nega (laranja) e "
               "parte da ameaça negada total (azul)."), ""]
        for por, rot in (("fase", "Por fase"), ("terco", "Por terço")):
            txt = "; ".join(f"{ROT[r[por]]} {pt(r['pct_negada'])} % negada, {pt(r['pct_da_negada_total'])} % do total ({r['lances']} momentos)"
                            for r in ft[por] if r[por] is not None)
            L += [f"- {rot}: {txt}."]
        L += ["", f"{ft['sem_fase']} momentos sem fase atribuída ficam fora da figura mas contam no total.", "",
              "Números: `avaliacao_v31.pct_por_contexto` sobre a base v3.1 em risco; `outputs/cover_shadow/artigo_v31.json`, secção `risco.fase_e_terco`.", ""]
    elif "fase_terco" in n:
        ROT = ag.ROTULOS_CONTEXTO
        ft = n["fase_terco"]
        L += ["## Agregados", ""]
        L += ["### agregado_fase_e_terco.png", "",
              ("Ameaça negada em % da disponível por fase defensiva e por terço da posse, nas duas moedas (xT e linhas quebradas), dois jogos juntos, v3.1. "
               "Laranja: moeda xT; azul: linhas quebradas."), ""]
        for moeda in ("xt", "linhas"):
            txt = "; ".join(f"{ROT[r['fase']]} {pt(r['pct_negada'])} % ({r['lances']} momentos)" for r in ft[moeda]["fase"])
            tt = "; ".join(f"{ROT[r['terco']]} {pt(r['pct_negada'])} % ({r['lances']} momentos)" for r in ft[moeda]["terco"])
            L += [f"- Moeda {moeda}. Por fase: {txt}. Por terço: {tt}."]
        L += ["", f"{ft['sem_fase']} momentos sem fase atribuída ficam fora da figura.", "",
              "Números: `avaliacao_v31.pct_por_contexto` sobre a base v3.1; `outputs/cover_shadow/artigo_v31.json`, secção `segunda_leitura_moedas`.", ""]
    if "revisoes" in n:
        r = n["revisoes"]
        linhas = "; ".join(f"{k}: sombras {v['sombras']['sim']}/{v['sombras']['em parte']}/{v['sombras']['não']}, marcação "
                           f"{v['marcacao']['sim']}/{v['marcacao']['em parte']}/{v['marcacao']['não']}" for k, v in r.items())
        L += ["### agregado_revisoes.png", "",
              f"Respostas (sim / em parte / não) nas três rondas de revisão: {linhas}.", "",
              f"Números: folhas preenchidas (`revisao_lances_preenchida*.xlsx`) ou `referencias/revisao_contagens.csv`; `{NB}`, secção 1 (1.1 a 1.4).", ""]
    if "versoes" in n:
        v = n["versoes"]
        txt = "; ".join(f"{k.replace('v3_1', 'v3.1').replace('_', ' · ')}: {pt(x['pct_negada'])} % negada, {pt(x['pct_excluidas'])} % excluídas" for k, x in v.items())
        L += ["### agregado_versoes.png", "", f"Ameaça negada e linhas excluídas por versão e jogo: {txt}."
              + (" Todas as versões em xT (histórico); a v3.1 oficial usa a moeda risco." if risco else ""), "",
              f"Números: `av.pct_negada` e `excluida` da base de cada versão; `{NB}`, secções 3.1 e 3.2.", ""]
    if "c3" in n:
        c = n["c3"]
        txt = "; ".join(f"{k}: {pt(x['dif_controlada'], 3)} [{pt(x['dif_controlada_inf'], 3)}; {pt(x['dif_controlada_sup'], 3)}] com controlo, "
                        f"{pt(x['dif_sem_controlo'], 3)} [{pt(x['dif_sem_controlo_inf'], 3)}; {pt(x['dif_sem_controlo_sup'], 3)}] sem (n = {x['n']})"
                        for k, x in c.items())
        L += ["### agregado_c3_sombra_escolhida.png", "",
              f"C3: sombra (1 − índice de chegada) da linha seguida menos a das alternativas do mesmo momento, com IC a 95 % (bootstrap). {txt}.", "",
              (f"Números: `av.comparar_com_comparaveis`; `{NB}`, secção 2.1 e 3.2 (a). v3.1: taxa de completos com p < 0,5 = "
              f"{pt(100 * c['v3_1']['taxa_baixa'])} % ({c['v3_1']['completos_baixa']} de {c['v3_1']['n_baixa']})."), ""]
    if "c4" in n:
        c = n["c4"]
        rhos = [r["rho_jogador"] for r in c]
        L += ["### agregado_c4_estabilidade.png", "",
              (f"C4: ρ de Spearman por jogador (dentro da posição) entre a v3.1 e {len(c)} variantes dos parâmetros; mínimo {pt(min(rhos), 3)}, "
              f"máximo {pt(max(rhos), 3)}). Eixo a começar em {pt(ag.limite_inferior_c4(min(rhos)), 2)} (calculado a partir do mínimo)."), "",
              f"Números: `av.spearman_por_posicao`; `{NB}`, secções 3.2 (b) e 3.3.", ""]
    if "c5" in n:
        c = n["c5"]
        (a1, b1), (a2, b2) = [c["pct_negada"][k] for k in sorted(c["pct_negada"], key=str)]
        L += ["### agregado_c5_tronco_vs_bola.png", "",
              (f"C5: ameaça negada por jogador com o cone pelo tronco e com o cone virado para a bola ({c['jogadores']} pontos). ρ = {pt(c['rho'], 3)} "
              f"(mínimo {pt(c['rho_min'], 3)}); a sombra muda em {pt(c['pct_mudam'])} % das {c['linhas']} linhas avaliadas; "
              f"% negada tronco/bola: {pt(a1)} / {pt(b1)} e {pt(a2)} / {pt(b2)}. "
              "Os eixos são um índice de ameaça negada (soma do xT das linhas), não golos: a soma por jogador junta o xT de linhas "
              "alternativas do mesmo momento."), "",
              f"Números: `av.linhas_que_mudam`, `av.spearman_por_posicao`; `{NB}`, secção 4.", ""]
    if "novas" in n:
        k = n["novas"]
        c8, xp, mo = k["c8"], k["xpass"], k["moedas"]
        ll = "; ".join(f"{d}: " + ", ".join(f"{mod} {pt(v, 3)}" for mod, v in t.items()) for d, t in c8["log_loss"].items())
        m3 = c8["M3"]
        L += ["## Figuras novas da fase 9b", "", "### novo_c8_escolha_do_passe.png", "",
              (f"C8: log-loss por passe fora da amostra (ajuste no outro jogo) de M0 (sem sombra), M1 (sombra virada para a bola), M2 (tronco) e M3 (as duas), "
               f"nos dois sentidos. {ll}. Coeficientes do M3 (dois jogos): sombra virada para a bola {pt(m3['sombra_bola']['coef'], 2)} "
               f"[{pt(m3['sombra_bola']['ic95_inf'], 2)}; {pt(m3['sombra_bola']['ic95_sup'], 2)}], sombra pelo tronco {pt(m3['sombra_tronco']['coef'], 2)} "
               f"[{pt(m3['sombra_tronco']['ic95_inf'], 2)}; {pt(m3['sombra_tronco']['ic95_sup'], 2)}]."), "",
              "Números: `modelo_escolha.json` (fase 8b, não recalculado); `outputs/cover_shadow/artigo_v31.json`, secção `ja_calculado`.", "",
              "### novo_xpass_calibracao_e_gamma.png", "",
              (f"xPass: calibração com 20 jogos (um de fora de cada vez; ECE {pt(xp['ece_20_jogos'], 4)}; {xp['n_20_jogos']} passes), peso γ da sombra com IC 95 % "
               f"(tronco {pt(xp['gamma']['tronco']['gamma'], 2)} [{pt(xp['gamma']['tronco']['ic95'][0], 2)}; {pt(xp['gamma']['tronco']['ic95'][1], 2)}]; "
               f"bola {pt(xp['gamma']['bola']['gamma'], 2)} [{pt(xp['gamma']['bola']['ic95'][0], 2)}; {pt(xp['gamma']['bola']['ic95'][1], 2)}]) e log-loss fora da "
               f"amostra sem sombra {pt(xp['log_loss']['sem_sombra'], 4)}, com a sombra pelo tronco {pt(xp['log_loss']['tronco'], 4)} e com a virada para a bola "
               f"{pt(xp['log_loss']['bola'], 4)}."), "",
              "Números: `xpass_v4.json` (fase 9a, não recalculado); `outputs/cover_shadow/artigo_v31.json`, secção `ja_calculado`.", "",
              ]
        if risco:
            L += ["### novo_moedas_por_fase.png", "",
                  ("Três moedas para o valor de uma linha, por fase defensiva. Esquerda: parte da ameaça negada total que cada fase reúne (risco / xT / linhas): "
                   + "; ".join(f"{ag.ROTULOS_CONTEXTO[f]} {pt(v['risco'])} % / {pt(v['xt'])} % / {pt(v['linhas'])} %" for f, v in mo["fases"].items())
                   + ". Direita: ρ de Spearman por jogador (dentro da posição), por bloco, entre risco e xT / risco e linhas / xT e linhas: "
                   + "; ".join(f"{ag.ROTULOS_CONTEXTO[b]} {pt(v['xt_risco'], 2)} / {pt(v['linhas_risco'], 2)} / {pt(v['xt_linhas'], 2)}" for b, v in mo["rho"].items()) + "."), "",
                  "Números: `artigo_v31.json`, secção `fase_9c.numeros_por_fase` (`numeros_9c.py`, fase 9c).", ""]
            p9 = n["pressao_alta"]
            leitura = fl.leitura_do_indice(p9["indice_na_fase"], p9["percentil_na_fase"])
            L += ["### novo_pressao_alta_goodwin.png", "",
                  (f"Pressão alta ({p9['equipa_defesa']} a defender, bloco alto): {p9['defesa']} cobre a linha do guarda-redes para o central. Em xT a ameaça negada é {pt(p9['xt'], 3)}; "
                   f"em risco é {pt(p9['risco'], 3)}, porque cortar o passe ali dava ao adversário um perigo enorme perto da baliza de quem tem a bola. "
                   f"{p9['defesa']} no bloco alto: à frente de {leitura['percentil']}% dos jogadores, com {leitura['vezes']} vezes a média da fase "
                   f"(índice {leitura['indice']}; {p9['momentos_na_fase']} momentos na fase)."), "",
                  "Números: `artigo_v31.json`, secção `risco.caso_pressao_alta`; lance 1996435, frame 22 256 (a etiqueta de fase vem da fase da SkillCorner, não do momento).", ""]
        else:
            L += ["### novo_moedas_por_fase.png", "",
                  ("Ameaça negada por fase defensiva em % da disponível, nas duas moedas (xT e linhas quebradas): "
                   + "; ".join(f"{ag.ROTULOS_CONTEXTO[f]} {pt(v['xt'])} % / {pt(v['linhas'])} % ({v['momentos']} momentos)" for f, v in mo["fases"].items()) + "."), "",
                  "Números: `avaliacao_v31.pct_por_contexto`; `outputs/cover_shadow/artigo_v31.json`, secção `segunda_leitura_moedas`.", ""]
    return "\n".join(L) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so", choices=["lances", "agregados"], default=None)
    args = parser.parse_args()
    brand = load_brand()
    apply_theme(brand)
    pasta = output_dir("cover_shadow") / "artigo"
    pasta.mkdir(parents=True, exist_ok=True)
    ficheiro_numeros = pasta / "numeros_das_figuras.json"
    numeros = json.loads(ficheiro_numeros.read_text(encoding="utf-8")) if ficheiro_numeros.exists() else {}
    if args.so in (None, "lances"):
        numeros.update(lances(brand, pasta))
    if args.so in (None, "agregados"):
        numeros.update(agregados(brand, pasta))
    # as chaves dos passos voltam de texto a inteiro ao ler o JSON
    if "c06_passos" in numeros:
        numeros["c06_passos"] = {int(k): v for k, v in numeros["c06_passos"].items()}
    ficheiro_numeros.write_text(json.dumps(numeros, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (pasta / "legendas.md").write_text(legendas(numeros), encoding="utf-8")
    if (pasta / "numeros_orientacao.json").exists():
        acrescentar_a_legendas(pasta)  # secção da fase 6d (orientacao_defesas.py)
    print(f"Figuras, numeros_das_figuras.json e legendas.md em {pasta}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script gera as figuras finais do artigo de uma só vez e escreve ao lado a legenda
# de cada uma, com os números que aparecem nela. Os números ficam também num ficheiro JSON, para se poder
# verificar cada um contra o notebook 05 antes de os usar.
