"""Fase 9b · três figuras novas do artigo: C8, xPass e as duas moedas por fase defensiva.

Os números vêm de resultados já calculados (`modelo_escolha.json`, `xpass_v4.json`) e das tabelas da v3.1; nada se recalcula aqui.
Mesmo tema e mesmas funções de base que `figuras_agregados`. Da raiz do repositório, através de `figuras_artigo.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import avaliacao_v31 as a31
import figuras_agregados as ag
import idioma as idi
import matplotlib.pyplot as plt
from entradas import exigir
from figuras_agregados import DPI, estilo_eixo, nota, nova_figura, pt

from footy.paths import output_dir
from footy.viz.theme import Brand

SAIDA = output_dir("cover_shadow")
MODELOS = (("M0", "sem sombra"), ("M1", "sombra virada para a bola"), ("M2", "sombra pelo tronco"), ("M3", "as duas sombras"))

# Textos das figuras 11 (C8) e 14 (xPass), por idioma. `modelos` na mesma ordem de MODELOS (M0 a M3).
TEXTOS = {
    "pt": {
        "modelos": ("sem sombra", "sombra virada para a bola", "sombra pelo tronco", "as duas sombras"),
        "bola2": "sombra virada\npara a bola", "tronco2": "sombra pelo\ntronco", "sem_sombra": "sem sombra", "ao_acaso": "ao acaso",
        "c8_titulo": "Escolha do passe: a sombra pelo tronco ajuda a prever para quem vai a bola?",
        "c8_sub": "Log-loss por passe fora da amostra (menor é melhor) e efeito de cada sombra no modelo com as duas · dois jogos",
        "c8_grupo": "treino {treino}\n→ teste {teste}",
        "c8_x": "log-loss por passe (ajuste no outro jogo)",
        "c8_coef": "Coeficientes do M3 (IC 95 %)",
        "c8_efeito": "efeito na escolha (logit; negativo = menos escolhida)",
        "c8_nota": ("{a} e {b} passes rasteiros (até 35 m) com as linhas do mesmo momento · "
                    "M0: comprimento, avanço, perigo, recetor marcado e distância ao adversário;\n"
                    "M1 a M3 acrescentam a(s) sombra(s). "
                    "Coeficientes: ajuste nos dois jogos juntos."),
        "xp_titulo": "xPass: calibração com 20 jogos e peso da sombra nos 2 jogos com pose",
        "xp_sub": "Probabilidade de um passe rasteiro chegar ao colega · xPass com 6 variáveis de geometria, sem orientação do corpo",
        "xp_x": "probabilidade prevista (média por decil)",
        "xp_y": "frequência observada",
        "xp_calibracao": "Calibração (20 jogos, um de fora de cada vez)",
        "xp_ece": "ECE {ece} · {n} passes",
        "xp_gamma": "Peso γ da sombra (IC 95 %)",
        "xp_gamma_x": "γ (logit): negativo =\na sombra reduz a chegada",
        "xp_logloss": "Log-loss, n = {n}",
        "xp_logloss_x": "fora da amostra (menor é melhor)",
        "xp_nota": ("γ e log-loss: ajustados num jogo e testados no outro, e ao contrário (2 jogos com body pose).\n"
                    "O xPass é ajustado nos restantes jogos. Sombra = 1 − índice de chegada (cone pelo tronco, parâmetros da literatura)."),
    },
    "en": {
        "modelos": ("no shadow", "ball-facing shadow", "trunk-based shadow", "both shadows"),
        "bola2": "ball-facing\nshadow", "tronco2": "trunk-based\nshadow", "sem_sombra": "no shadow", "ao_acaso": "random",
        "c8_titulo": "Pass choice: does the trunk-based shadow help predict who the ball goes to?",
        "c8_sub": "Out-of-sample log-loss per pass (lower is better) and effect of each shadow in the model with both · both matches",
        "c8_grupo": "train {treino}\n→ test {teste}",
        "c8_x": "log-loss per pass (fitted on the other match)",
        "c8_coef": "M3 coefficients (95% CI)",
        "c8_efeito": "effect on choice (logit; negative = chosen less)",
        "c8_nota": ("{a} and {b} ground passes (up to 35 m) with the lanes of the same moment · "
                    "M0: length, advance, threat, marked receiver and distance to the opponent;\n"
                    "M1 to M3 add the shadow(s). "
                    "Coefficients: fitted on both matches together."),
        "xp_titulo": "xPass: calibration on 20 matches, shadow weight on the 2 with pose",
        "xp_sub": "Probability of a ground pass reaching the teammate · xPass with 6 geometry variables, no body orientation",
        "xp_x": "predicted probability (mean per decile)",
        "xp_y": "observed frequency",
        "xp_calibracao": "Calibration (20 matches, one held out at a time)",
        "xp_ece": "ECE {ece} · {n} passes",
        "xp_gamma": "Shadow weight γ (95% CI)",
        "xp_gamma_x": "γ (logit): negative =\nthe shadow reduces arrival",
        "xp_logloss": "Log-loss, n = {n}",
        "xp_logloss_x": "out of sample (lower is better)",
        "xp_nota": ("γ and log-loss: fitted on one match and tested on the other, and vice versa (2 matches with body pose).\n"
                    "xPass is fitted on the remaining matches. Shadow = 1 − arrival index (trunk-based cone, parameters from the literature)."),
    },
}


def ler(nome: str) -> dict:
    corre = {"modelo_escolha.json": "modelo_escolha.py", "xpass_v4.json": "avaliacao_xpass.py"}.get(nome, "os passos 3 a 9 do README")
    return json.loads(exigir(SAIDA / nome, corre).read_text(encoding="utf-8"))


def ponto_com_intervalo(ax, y: float, valor: float, baixo: float, alto: float, cor: str, fundo: str, texto: str, fonte: float = 12.5) -> None:
    ax.plot([baixo, alto], [y, y], color=cor, lw=3.2, solid_capstyle="round", zorder=3)
    ax.scatter([valor], [y], s=150, color=cor, edgecolor=fundo, zorder=4)
    ax.text(alto + 0.04, y, texto, va="center", fontsize=fonte, color=cor if cor != fundo else plt.rcParams["text.color"])


# ---------------------------------------------------------------- (a) C8

def figura_c8(brand: Brand, destino: Path, titulos: dict, idioma: str = "pt") -> dict:
    """Log-loss fora da amostra de M0 a M3 nos dois sentidos e o efeito da sombra pelo tronco e pela bola no M3."""
    cor = brand.colors
    pt = idi.formatadores(idioma, menos=True)[0]
    X = TEXTOS[idioma]
    m = ler("modelo_escolha.json")
    fig = nova_figura(brand, X["c8_titulo"], X["c8_sub"])
    ax = fig.add_axes((0.30, 0.17, 0.33, 0.66))
    estilo_eixo(ax, brand, grelha="x")
    y, posicao, rotulos, grupos = [], 0.0, [], []
    for chave, d in m["fora_da_amostra"].items():
        grupos.append((posicao - 0.6, X["c8_grupo"].format(treino=titulos[d["treino"]], teste=titulos[d["teste"]])))
        for (modelo, _), rotulo in zip(MODELOS, X["modelos"]):
            v = d["modelos"][modelo]["log_loss"]
            tom = cor["primary"] if modelo == "M2" else cor["text_muted"]
            ax.scatter([v], [posicao], s=150, color=tom, edgecolor=cor["background"], zorder=4)
            ax.text(v + 0.012, posicao, f"{pt(v, 3)}", va="center", fontsize=12.5, color=cor["text"], fontweight="bold" if modelo == "M2" else "normal")
            y.append(posicao)
            rotulos.append(f"{modelo} · {rotulo}")
            posicao += 1
        ao_acaso = d["ao_acaso"]["log_loss"]
        ax.scatter([ao_acaso], [posicao], s=110, facecolor="none", edgecolor=cor["text_muted"], lw=1.8, zorder=4)
        ax.text(ao_acaso + 0.012, posicao, f"{pt(ao_acaso, 3)}", va="center", fontsize=12.5, color=cor["text_muted"])
        y.append(posicao)
        rotulos.append(X["ao_acaso"])
        posicao += 1.9
    ax.set_yticks(y, rotulos)
    ax.tick_params(axis="y", labelsize=12, colors=cor["text"])
    ax.invert_yaxis()
    ax.set_xlim(1.38, 2.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pt(v, 1)))
    for topo, texto in grupos:
        ax.text(1.382, topo - 0.15, texto, fontsize=12.5, fontweight="bold", color=cor["text"], va="bottom")
    ax.set_xlabel(X["c8_x"], fontsize=12.5, color=cor["text_muted"], labelpad=8)
    # efeito no M3
    ax2 = fig.add_axes((0.73, 0.17, 0.23, 0.66))
    estilo_eixo(ax2, brand, grelha="x")
    m3 = m["coeficientes_conjunto"]["M3"]
    for yi, (chave, rotulo) in enumerate((("sombra_bola", X["bola2"]), ("sombra_tronco", X["tronco2"]))):
        c = m3[chave]
        tom = cor["primary"] if chave == "sombra_tronco" else cor["text_muted"]
        ax2.plot([c["ic95_inf"], c["ic95_sup"]], [yi, yi], color=tom, lw=3.2, solid_capstyle="round", zorder=3)
        ax2.scatter([c["coef"]], [yi], s=150, color=tom, edgecolor=cor["background"], zorder=4)
        ax2.text(c["coef"], yi - 0.28, f"{pt(c['coef'], 2)}  [{pt(c['ic95_inf'], 2)}; {pt(c['ic95_sup'], 2)}]", ha="center", fontsize=11.5,
                 color=cor["text"])
    ax2.axvline(0, color=cor["text"], lw=1.2)
    ax2.set_yticks([0, 1], [X["bola2"], X["tronco2"]])
    ax2.tick_params(axis="y", labelsize=12, colors=cor["text"])
    ax2.set_ylim(1.7, -0.7)
    ax2.set_xlim(-2.3, 0.5)
    ax2.set_title(X["c8_coef"], loc="left", fontsize=14, fontweight="bold", color=cor["text"], pad=8)
    ax2.set_xlabel(X["c8_efeito"], fontsize=11.5, color=cor["text_muted"], labelpad=8)
    nota(fig, brand, X["c8_nota"].format(a=m["passes_por_jogo"]["1925299"], b=m["passes_por_jogo"]["1996435"]), y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"log_loss": {k: {mod: d["modelos"][mod]["log_loss"] for mod, _ in MODELOS} | {"ao_acaso": d["ao_acaso"]["log_loss"]}
                         for k, d in m["fora_da_amostra"].items()},
            "M3": {k: m3[k] for k in ("sombra_bola", "sombra_tronco")}}


# ---------------------------------------------------------------- (b) xPass

def figura_xpass(brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    """Curva de calibração do xPass (20 jogos), γ com IC 95 % e log-loss com e sem sombra nos dois jogos com pose."""
    cor = brand.colors
    pt, _, mil = idi.formatadores(idioma, menos=True)
    X = TEXTOS[idioma]
    x = ler("xpass_v4.json")
    fig = nova_figura(brand, X["xp_titulo"], X["xp_sub"])
    curva = x["xpass"]["curva_de_calibracao"]
    ax = fig.add_axes((0.07, 0.17, 0.27, 0.62))
    estilo_eixo(ax, brand, grelha="both")
    ax.plot([0, 1], [0, 1], color=cor["text_muted"], lw=1.3, ls=(0, (4, 3)), zorder=2)
    prev, obs = [c["previsto"] for c in curva], [c["observado"] for c in curva]
    ax.plot(prev, obs, color=cor["primary_ink"], lw=2.2, zorder=3)
    ax.scatter(prev, obs, s=95, color=cor["primary"], edgecolor=cor["background"], zorder=4)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel(X["xp_x"], fontsize=12, color=cor["text_muted"], labelpad=8)
    ax.set_ylabel(X["xp_y"], fontsize=12, color=cor["text_muted"], labelpad=8)
    ax.set_title(X["xp_calibracao"], loc="left", fontsize=13.5, fontweight="bold", color=cor["text"], pad=8)
    mj = x["xpass"]["metricas_juntas"]
    ax.text(0.28, 0.96, X["xp_ece"].format(ece=pt(mj["ece"], 4), n=mil(mj["n"])), fontsize=12, color=cor["text"], va="top")
    # γ
    g = x["sombra"]["gamma_dois_jogos"]
    ax2 = fig.add_axes((0.45, 0.17, 0.22, 0.62))
    estilo_eixo(ax2, brand, grelha="x")
    for yi, (chave, rotulo, tom) in enumerate((("bola", X["bola2"], cor["text_muted"]), ("tronco", X["tronco2"], cor["primary"]))):
        c = g[chave]
        ax2.plot(c["ic95"], [yi, yi], color=tom, lw=3.2, solid_capstyle="round", zorder=3)
        ax2.scatter([c["gamma"]], [yi], s=150, color=tom, edgecolor=cor["background"], zorder=4)
        ax2.text(c["gamma"], yi - 0.3, f"{pt(c['gamma'], 2)}  [{pt(c['ic95'][0], 2)}; {pt(c['ic95'][1], 2)}]", ha="center", fontsize=11.5, color=cor["text"])
    ax2.axvline(0, color=cor["text"], lw=1.2)
    ax2.set_yticks([0, 1], [X["bola2"], X["tronco2"]])
    ax2.tick_params(axis="y", labelsize=12, colors=cor["text"])
    ax2.set_ylim(1.7, -0.7)
    ax2.set_xlim(-1.2, 0.5)
    ax2.set_title(X["xp_gamma"], loc="left", fontsize=13.5, fontweight="bold", color=cor["text"], pad=8)
    ax2.set_xlabel(X["xp_gamma_x"], fontsize=11.5, color=cor["text_muted"], labelpad=8)
    # log-loss com e sem sombra
    ax3 = fig.add_axes((0.78, 0.17, 0.18, 0.62))
    estilo_eixo(ax3, brand, grelha="x")
    ll = x["sombra"]["metricas_juntas"]
    for yi, (chave, rotulo, tom) in enumerate((("sem_sombra", X["sem_sombra"], cor["text"]), ("bola", X["bola2"], cor["text_muted"]),
                                                ("tronco", X["tronco2"], cor["primary"]))):
        ax3.scatter([ll[chave]["log_loss"]], [yi], s=150, color=tom, edgecolor=cor["background"], zorder=4)
        ax3.text(ll[chave]["log_loss"] + 0.0004, yi - 0.28, pt(ll[chave]["log_loss"], 4), ha="center", fontsize=11.5, color=cor["text"])
    ax3.set_yticks([0, 1, 2], [X["sem_sombra"], X["bola2"], X["tronco2"]])
    ax3.tick_params(axis="y", labelsize=12, colors=cor["text"])
    ax3.set_ylim(2.7, -0.7)
    ax3.set_xlim(0.2655, 0.2715)
    ax3.set_title(X["xp_logloss"].format(n=mil(ll["sem_sombra"]["n"])), loc="left", fontsize=13.5, fontweight="bold", color=cor["text"], pad=8)
    ax3.set_xlabel(X["xp_logloss_x"], fontsize=11.5, color=cor["text_muted"], labelpad=8)
    nota(fig, brand, X["xp_nota"], y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"ece_20_jogos": mj["ece"], "n_20_jogos": mj["n"], "gamma": g, "log_loss": {k: v["log_loss"] for k, v in ll.items()}}


# ---------------------------------------------------------------- (c) moedas

def figura_moedas(JG: dict, brand: Brand, destino: Path) -> dict:
    """% da ameaça negada por fase defensiva na moeda xT contra a moeda «linhas quebradas» (dois jogos juntos)."""
    cor = brand.colors
    dados, sem_fase = ag.ameaca_por_contexto(JG)
    xt, li = (dados[m]["fase"].set_index("fase") for m in a31.MOEDAS)
    ordem = xt.sort_values("pct_negada").index.tolist()
    fig = nova_figura(brand, "Ameaça negada por fase defensiva: ganho de xT contra linhas quebradas",
                      "Em % da ameaça disponível nas linhas avaliadas · dois jogos juntos · definição v3.1")
    ax = fig.add_axes((0.22, 0.17, 0.72, 0.66))
    estilo_eixo(ax, brand, grelha="x")
    maximo = max(xt["pct_negada"].max(), li["pct_negada"].max())
    for i, fase in enumerate(ordem):
        a, b = xt.loc[fase, "pct_negada"], li.loc[fase, "pct_negada"]
        ax.plot([a, b], [i, i], color=cor["lines"], lw=3, zorder=2)
        ax.scatter([a], [i], s=190, color=cor["primary"], edgecolor=cor["background"], zorder=4)
        ax.scatter([b], [i], s=190, color=cor["secondary"], edgecolor=cor["background"], zorder=4)
        esquerda = a < b
        ax.text(a - maximo * 0.015 if esquerda else a + maximo * 0.015, i, f"{pt(a)} %", ha="right" if esquerda else "left", va="center",
                fontsize=12.5, color=cor["text"], fontweight="bold")
        ax.text(b + maximo * 0.015 if esquerda else b - maximo * 0.015, i, f"{pt(b)} %", ha="left" if esquerda else "right", va="center",
                fontsize=12.5, color=cor["text"], fontweight="bold")
    ax.set_yticks(range(len(ordem)), [f"{ag.ROTULOS_CONTEXTO[f]}\n{int(xt.loc[f, 'lances']):,} momentos".replace(",", " ") for f in ordem])
    ax.tick_params(axis="y", labelsize=13, colors=cor["text"])
    ax.set_xlim(0, maximo * 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    for k, moeda in enumerate(a31.MOEDAS):
        x0 = 0.22 + k * 0.30
        fig.patches.append(plt.Rectangle((x0, 0.862), 0.012, 0.022, transform=fig.transFigure, color=(cor["primary"], cor["secondary"])[k]))
        fig.text(x0 + 0.017, 0.873, ag.MOEDAS_ROTULO[moeda], fontsize=13, color=cor["text"], va="center")
    nota(fig, brand, f"Um momento = um frame alinhado (0,2 s) com pelo menos uma linha avaliada; os mesmos nas duas moedas. Fora da figura: {sem_fase} momentos sem fase atribuída.\n"
         "Índice de fecho da linha, não probabilidade calibrada.", y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"fases": {f: {"momentos": int(xt.loc[f, "lances"]), "xt": float(xt.loc[f, "pct_negada"]), "linhas": float(li.loc[f, "pct_negada"])}
                      for f in ordem}, "sem_fase": sem_fase, "ordem_xt": ordem, "ordem_linhas": li.sort_values("pct_negada").index.tolist()}


# ---------------------------------------------------------------- (d) momentos em que as moedas discordam

def figura_momento(J: dict, frame: int, rotulo: str, valores: dict, percentis: dict, brand: Brand, destino: Path) -> Path:
    """Imagem de um momento como as da revisão (cones, linhas, marcação) com o painel das duas moedas.
    `valores`: {moeda: `avaliacao_v31.negada_por_defesa_no_lance`}; `percentis`: {moeda: percentil do momento}."""
    import revisao as rev
    comprimento, largura = J["campo"]
    d = rev.dados_do_frame(J["tabela"], J["defesas"], J["match"], frame, comprimento, largura)
    fig, ax_campo, ax_texto = rev.figura_base(brand)
    rev.desenhar_lance(fig, ax_campo, ax_texto, d, J["nomes"], brand, rev.novo_campo(brand, comprimento, largura), rotulo,
                       rev.vista_em_torno(rev.jogadores_da_vista(d), comprimento, largura))
    ax_texto.clear()
    ax_texto.axis("off")
    cor = brand.colors
    nomes = rev.negada_por_defesa(d)
    texto = [("Ameaça negada por defesa", "text", True)]
    for jogador in nomes.index:
        nome = J["nomes"][jogador]
        texto.append((f"{nome}", "text_muted", False))
        texto.append((f"   xT {pt(valores['xt']['shapley'].get(nome, 0.0), 4)} · linhas {pt(valores['linhas']['shapley'].get(nome, 0.0), 2)}", "text_muted", False))
    if not len(nomes):
        texto.append(("nenhuma", "text_muted", False))
    texto += [("", "text", False), ("Momento (todas as linhas avaliadas)", "text", True)]
    for moeda, rot, casas in (("xt", "xT", 4), ("linhas", "linhas quebradas", 2)):
        v = valores[moeda]
        texto.append((f"{rot}: perigo {pt(v['perigo'], 3 if moeda == 'xt' else 0)} · negada {pt(v['negada'], casas)}", "text_muted", False))
        texto.append((f"   percentil {pt(100 * percentis[moeda], 0)} % (blocos alto e médio)", "text_muted", False))
    y = 0.98
    for linha, tom, negrito in texto:
        ax_texto.text(0, y, linha, transform=ax_texto.transAxes, va="top", fontsize=9, color=cor[tom], fontweight="bold" if negrito else "normal")
        y -= 0.045
    rev.rodape(fig, brand, ag.FONTE)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destino


def todas(JG: dict, brand: Brand, pasta: Path) -> dict:
    titulos = {j: JG[j]["titulo"] for j in JG}
    return {"c8": figura_c8(brand, pasta / "novo_c8_escolha_do_passe.png", titulos),
            "xpass": figura_xpass(brand, pasta / "novo_xpass_calibracao_e_gamma.png"),
            "moedas": figura_moedas(JG, brand, pasta / "novo_moedas_por_fase.png")}


# Em linguagem corrente: três figuras novas. A primeira mostra se saber para onde olha o defesa ajuda a prever para quem vai o passe
# (fora da amostra, nos dois jogos). A segunda mostra que a probabilidade «base» de um passe chegar está bem calibrada com 20 jogos e
# qual o peso que a sombra ganha por cima dela. A terceira compara, por fase defensiva, a percentagem da ameaça que se nega
# contada em xT e contada em adversários ultrapassados.
