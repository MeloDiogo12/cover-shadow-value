"""Fase 10a · `legendas_en.md`: as legendas das 16 figuras inglesas do artigo, no formato de `legendas.md` (legenda curta, números, origem).

Os números são os que as próprias figuras devolvem (`figuras_10a.gerar`), por isso a legenda e a figura não podem discordar.
Só têm entrada as 16 figuras do artigo; as figuras do passo a passo que não entram no artigo (passos 1, 2, 3, 6 e 7) ficam de fora.
"""

from __future__ import annotations

from pathlib import Path

import figuras_agregados as ag
import figuras_lance as fl
import idioma as idi

NB = "notebook 05"
CTX = idi.ROTULOS_CONTEXTO["en"]
NOTA = fl.TEXTOS["en"]["nota_soma"]


def legendas(n: dict) -> str:
    pt, pct, mil = idi.formatadores("en", menos=True)
    U = "risk"
    L = ["# Figure captions of the article (English)", "",
         ("Internal, not for publication. Each entry: short caption, the numbers in the figure and where they come from. Source of all figures: "
          "SkillCorner Open Data (MIT), two matches with body pose (Brisbane Roar – Perth Glory, 21/12/2024; Sydney FC – Adelaide United, "
          "01/02/2025), v3.1 definition of the metric (v3 without lanes longer than 35 m). Values in risk, the official currency (D44): "
          "lane value = xT gain + the opponent's xT at the point where the lane would be cut. The shadow value is a lane-closure index, "
          "not a calibrated probability of the pass arriving (D43). Figure numbers are those of the article; file names are the same as in the Portuguese folder."), ""]
    p = n["c06_passos"]
    s = p[8]["shapley"]
    partes = ", ".join(f"{k} {pt(v, 4)}" for k, v in s.items())
    total, perigo = p[8]["negada_total_xt"], p[8]["perigo_total_xt"]
    L += ["## C06 · step by step (steps 4, 5 and summary)", "",
          ("Brisbane Roar – Perth Glory, first half, minute 21 (frame 12,660). Source: SkillCorner tracking, body pose and events; "
           "computation in `metrica.linhas` (the same as notebook 04). Candidate C06 of `candidatos.md`."), ""]
    c5 = p[5]["p_com_sombra"]
    textos = [
        ("c06_passo4_cones_pelo_tronco.png", "Figure 2", f"Step 4. {pt(60, 0)}° cone, {pt(10, 0)} m long, behind the trunk of each defender who is not marking; arrow = front of the trunk."),
        ("c06_passo5_probabilidade_de_o_passe_chegar.png", "Figure 3",
         (f"Step 5. Arrival index on {'the only lane' if len(c5) == 1 else 'the ' + str(len(c5)) + ' lanes'} with a shadow (out of {p[5]['avaliadas']} evaluated): "
          f"{', '.join(pt(v, 2) for v in c5)}. Orange = shadow (index below 0.95), grey = no shadow; "
          f"{p[5]['cruzam_cone'] - len(c5)} lane crosses a cone but with an index ≥ 0.95, so it stays grey. It is an index, not a calibrated probability.")),
        ("c06_passo8_resumo.png", "Figure 4",
         f"Summary. Denied threat {pt(total, 4)} {U} ({pct(100 * total / perigo, 1)} of the available threat); {partes}. {NOTA}"),
    ]
    for ficheiro, fig, texto in textos:
        L += [f"### {ficheiro} ({fig})", "", texto, "", f"Numbers: play C06; `metrica.linhas.avaliar_frame`; see `{NB}`, section 1 (drawing of the plays) for the format.", ""]
    c = n["c06_corpo_bola"]
    t, b = c[fl.TRONCO], c[fl.BOLA]
    pt_t = ", ".join(f"{k} {pt(v, 4)}" for k, v in t["por_defesa"].items())
    pt_b = ", ".join(f"{k} {pt(v, 4)}" for k, v in b["por_defesa"].items())
    L += ["### c06_corpo_vs_bola.png (Figure 9)", "",
          (f"C06 with the trunk-based cone (the metric) and with the ball-facing cone. Denied threat: {pt(t['total_xt'], 4)} {U} "
           f"({pt_t}) against {pt(b['total_xt'], 4)} {U} ({pt_b}). {NOTA}"), "",
          f"Numbers: `pipeline.preparar_defesas(..., orientacao=\"bola\")`; it is the comparison of section 4 of {NB} (C5) applied to one play.", ""]
    c = n["c03"]
    L += ["## C03", "", "### c03_lance.png (Figure 15)", "",
          (f"Sydney FC – Adelaide United, second half, minute 19 (frame 39,646). The cone behind Klimala's trunk covers the ball carrier's (Vriends) lane "
           f"to Isaías: arrival index {pt(c['p_chegar'], 2)}; lane value {pt(c['valor_xt'], 4)} {U}, "
           f"denied threat on the lane {pt(c['negada_linha'], 4)} {U} (Klimala only); denied threat in the play {pt(c['negada_lance'], 4)} {U}. {NOTA}"), "",
          "Numbers: candidate C03 of `candidatos.md`; `artigo_v31.json`, section `risco.c03`.", ""]
    ft = n["fase_terco"]
    L += ["## Aggregates", "", "### agregado_fase_e_terco.png (Figure 6)", "",
          ("Denied threat in risk, by defensive phase and by third of the possession, both matches together, v3.1: % of the available threat (g + c) that is denied (orange) and "
           "share of the total denied threat (blue)."), ""]
    for por, rot in (("fase", "By phase"), ("terco", "By third")):
        txt = "; ".join(f"{CTX[r[por]]} {pct(r['pct_negada'])} denied, {pct(r['pct_da_negada_total'])} of the total ({mil(r['lances'])} moments)"
                        for r in ft[por] if r[por] is not None)
        L += [f"- {rot}: {txt}."]
    L += ["", f"{mil(ft['sem_fase'])} moments with no phase assigned are left out of the figure but count in the total.", "",
          "Numbers: `avaliacao_v31.pct_por_contexto` on the v3.1 base in risk; `outputs/cover_shadow/artigo_v31.json`, section `risco.fase_e_terco`.", ""]
    r = n["revisoes"]
    rondas = dict(zip(r, ag.TEXTOS["en"]["rv_rondas"]))
    linhas = "; ".join(f"{rondas[k]}: shadows {v['sombras']['sim']}/{v['sombras']['em parte']}/{v['sombras']['não']}, marking "
                       f"{v['marcacao']['sim']}/{v['marcacao']['em parte']}/{v['marcacao']['não']}" for k, v in r.items())
    L += ["### agregado_revisoes.png (Figure 5)", "",
          f"Coach's answers (yes / partly / no) in the three review rounds: {linhas}.", "",
          f"Numbers: filled-in sheets (`revisao_lances_preenchida*.xlsx`); `{NB}`, section 1 (1.1 to 1.4).", ""]
    v = n["versoes"]
    txt = "; ".join(f"{k.replace('v3_1', 'v3.1').replace('_', ' · ')}: {pct(x['pct_negada'])} denied, {pct(x['pct_excluidas'])} excluded" for k, x in v.items())
    L += ["### agregado_versoes.png (Figure 1)", "",
          f"Denied threat and excluded lanes by version and match: {txt}. All versions in xT (historical); the official v3.1 uses the risk currency.", "",
          f"Numbers: `av.pct_negada` and `excluida` of the base of each version; `{NB}`, sections 3.1 and 3.2.", ""]
    c = n["c3"]
    txt = "; ".join(f"{k}: {pt(x['dif_controlada'], 3)} [{pt(x['dif_controlada_inf'], 3)}; {pt(x['dif_controlada_sup'], 3)}] with control, "
                    f"{pt(x['dif_sem_controlo'], 3)} [{pt(x['dif_sem_controlo_inf'], 3)}; {pt(x['dif_sem_controlo_sup'], 3)}] without (n = {x['n']})"
                    for k, x in c.items())
    L += ["### agregado_c3_sombra_escolhida.png (Figure 7)", "",
          f"C3: shadow (1 − arrival index) on the lane followed minus that of the alternatives of the same moment, with a 95% CI (bootstrap). {txt}.", "",
          (f"Numbers: `av.comparar_com_comparaveis`; `{NB}`, sections 2.1 and 3.2 (a). v3.1: completion rate with p < 0.5 = "
           f"{pct(100 * c['v3_1']['taxa_baixa'])} ({c['v3_1']['completos_baixa']} of {c['v3_1']['n_baixa']})."), ""]
    c4 = n["c4"]
    rhos = [x["rho_jogador"] for x in c4]
    L += ["### agregado_c4_estabilidade.png (Figure 10)", "",
          (f"C4: Spearman ρ per player (within position) between v3.1 and {len(c4)} parameter variants; minimum {pt(min(rhos), 3)}, "
           f"maximum {pt(max(rhos), 3)}. Axis starting at {pt(ag.limite_inferior_c4(min(rhos)), 2)} (computed from the minimum)."), "",
          f"Numbers: `av.spearman_por_posicao`; `{NB}`, sections 3.2 (b) and 3.3.", ""]
    c = n["c5"]
    (a1, b1), (a2, b2) = [c["pct_negada"][k] for k in sorted(c["pct_negada"], key=str)]
    L += ["### agregado_c5_tronco_vs_bola.png (Figure 8)", "",
          (f"C5: denied threat per player with the trunk-based cone and with the ball-facing cone ({c['jogadores']} points). ρ = {pt(c['rho'], 3)} "
           f"(minimum {pt(c['rho_min'], 3)}); the shadow changes in {pct(c['pct_mudam'])} of the {mil(c['linhas'])} evaluated lanes; "
           f"denied % trunk / ball: {pt(a1)} / {pt(b1)} and {pt(a2)} / {pt(b2)}. "
           "The axes are a denied-threat index (sum of lane values), not goals: the per-player sum adds up the value of alternative lanes of the same moment."), "",
          f"Numbers: `av.linhas_que_mudam`, `av.spearman_por_posicao`; `{NB}`, section 4.", ""]
    k = n["novas"]
    c8, xp, mo = k["c8"], k["xpass"], k["moedas"]
    ll = "; ".join(f"{d}: " + ", ".join(f"{mod} {pt(x, 3)}" for mod, x in t.items()) for d, t in c8["log_loss"].items())
    m3 = c8["M3"]
    L += ["## New figures of phase 9b", "", "### novo_c8_escolha_do_passe.png (Figure 11)", "",
          (f"C8: out-of-sample log-loss per pass (fitted on the other match) of M0 (no shadow), M1 (ball-facing shadow), M2 (trunk-based) and M3 (both), "
           f"in both directions; 'ao_acaso' = random. {ll}. M3 coefficients (both matches): ball-facing shadow {pt(m3['sombra_bola']['coef'], 2)} "
           f"[{pt(m3['sombra_bola']['ic95_inf'], 2)}; {pt(m3['sombra_bola']['ic95_sup'], 2)}], trunk-based shadow {pt(m3['sombra_tronco']['coef'], 2)} "
           f"[{pt(m3['sombra_tronco']['ic95_inf'], 2)}; {pt(m3['sombra_tronco']['ic95_sup'], 2)}]."), "",
          "Numbers: `modelo_escolha.json` (phase 8b, not recomputed); `outputs/cover_shadow/artigo_v31.json`, section `ja_calculado`.", "",
          "### novo_xpass_calibracao_e_gamma.png (Figure 14)", "",
          (f"xPass: calibration with 20 matches (one held out at a time; ECE {pt(xp['ece_20_jogos'], 4)}; {mil(xp['n_20_jogos'])} passes), shadow weight γ with a 95% CI "
           f"(trunk {pt(xp['gamma']['tronco']['gamma'], 2)} [{pt(xp['gamma']['tronco']['ic95'][0], 2)}; {pt(xp['gamma']['tronco']['ic95'][1], 2)}]; "
           f"ball {pt(xp['gamma']['bola']['gamma'], 2)} [{pt(xp['gamma']['bola']['ic95'][0], 2)}; {pt(xp['gamma']['bola']['ic95'][1], 2)}]) and out-of-sample "
           f"log-loss with no shadow {pt(xp['log_loss']['sem_sombra'], 4)}, with the trunk-based shadow {pt(xp['log_loss']['tronco'], 4)} and with the ball-facing one "
           f"{pt(xp['log_loss']['bola'], 4)}."), "",
          "Numbers: `xpass_v4.json` (phase 9a, not recomputed); `outputs/cover_shadow/artigo_v31.json`, section `ja_calculado`.", "",
          "### novo_moedas_por_fase.png (Figure 12)", "",
          ("Three currencies for the value of a lane, by defensive phase. Left: share of the total denied threat that each phase accounts for (risk / xT / lines broken): "
           + "; ".join(f"{CTX[f]} {pct(x['risco'])} / {pct(x['xt'])} / {pct(x['linhas'])}" for f, x in mo["fases"].items())
           + ". Right: Spearman ρ per player (within position), by block, between risk and xT / risk and lines / xT and lines: "
           + "; ".join(f"{CTX[b]} {pt(x['xt_risco'], 2)} / {pt(x['linhas_risco'], 2)} / {pt(x['xt_linhas'], 2)}" for b, x in mo["rho"].items()) + "."), "",
          "Numbers: `artigo_v31.json`, section `fase_9c.numeros_por_fase` (`numeros_9c.py`, phase 9c).", ""]
    p9 = n["pressao_alta"]
    leitura = fl.leitura_do_indice(p9["indice_na_fase"], p9["percentil_na_fase"], "en")
    L += ["### novo_pressao_alta_goodwin.png (Figure 13)", "",
          (f"High press ({p9['equipa_defesa']} defending, high block): {p9['defesa']} covers the goalkeeper's lane to the centre-back. In xT the denied threat is {pt(p9['xt'], 3)}; "
           f"in risk it is {pt(p9['risco'], 3)}, because cutting the pass there would have given the opponent an enormous threat near the goal of the team in possession. "
           f"{p9['defesa']} in the high block: ahead of {leitura['percentil']}% of players, with {leitura['vezes']} times the phase average "
           f"(index {leitura['indice']}; {p9['momentos_na_fase']} moments in the phase)."), "",
          "Numbers: `artigo_v31.json`, section `risco.caso_pressao_alta`; match 1996435, frame 22,256 (the phase label comes from the SkillCorner phase, not from the moment).", ""]
    bf = n["bola_frente"]
    sep = "; ".join(f"separation {x['separacao']}: n = {mil(x['n'])}, median trunk–ball carrier {pt(x['mediana_portador'], 0)}°, "
                    f"trunk–forward {pt(x['mediana_frente'], 0)}°, trunk closer to the ball in {pct(x['pct_mais_perto_da_bola'])}" for x in bf)
    L += ["## Orientation of the players of the team out of possession", "", "### orientacao_bola_vs_frente.png (Figure 16)", "",
          (f"Trunk angle relative to the ball carrier and relative to the forward direction (perpendicular to the own goal line, away from it), by "
           f"band of separation between the two directions (both matches, v3, players of the team out of possession with pose, without goalkeepers, during the "
           f"possessions, 5 observations per second). {sep}. The separation is geometry only (it does not depend on the pose)."), "",
          ("Numbers: `orientacao_defesas.bola_vs_frente` (also by distance band and by group, in `numeros_orientacao.json`); "
           f"`{NB}`, section 4.1."), ""]
    return "\n".join(L) + "\n"


def escrever(pasta: Path, n: dict) -> None:
    (pasta / "legendas_en.md").write_text(legendas(n), encoding="utf-8")


# Em linguagem corrente: escreve, em inglês e no mesmo formato de `legendas.md`, a legenda de cada uma das 16 figuras, com os números
# exatos que a figura mostra e de onde vêm.
