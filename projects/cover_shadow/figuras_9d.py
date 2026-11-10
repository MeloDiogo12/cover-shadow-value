"""Fase 9d · figuras do artigo na moeda oficial «risco» (D44), em `outputs/cover_shadow/artigo/`.

Lê os números de `artigo_v31.json` (secção `risco`, de `numeros_9d.py`), por isso cada figura mostra exatamente o que está no ficheiro.
As figuras em xT ficam guardadas em `outputs/cover_shadow/artigo_v31_xt/` (copiadas antes de regenerar). Da raiz do repositório:
    uv run python projects/cover_shadow/figuras_9d.py
Regenera: C06 (passo a passo e corpo vs bola), C03, fase e terço, C4, C5, moedas por fase (três moedas) e o caso novo de pressão alta;
a figura das versões mantém o xT (histórico) com uma nota. As legendas (`legendas.md`) são as de `figuras_artigo.legendas`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import figuras_agregados as ag
import figuras_artigo as fa
import figuras_lance as fl
import gerar_candidatos as gc
import idioma as idi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from entradas import exigir
from figuras_agregados import estilo_eixo, nota, nova_figura
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
from orientacao_defesas import acrescentar_a_legendas

from footy.paths import PROJECTS, output_dir
from footy.viz.theme import Brand, apply_theme, load_brand

SAIDA = output_dir("cover_shadow")
PASTA = SAIDA / "artigo"
C06, C03 = fa.C06, fa.C03
PRESSAO_ALTA = {"jogo": 1996435, "frame": 22256, "defesa": "Goodwin"}
NOTA_RISCO = "Valor da linha = ganho de xT + xT do adversário no ponto onde a linha seria cortada (risco).\nÍndice de fecho da linha, não probabilidade calibrada."
LIMITE_MOMENTOS_VISIVEL = 50

# Textos das figuras 6, 12 e 13, por idioma (os das figuras 8 e 10 estão em `figuras_agregados.TEXTOS`).
TEXTOS = {
    "pt": {
        "nota_risco": NOTA_RISCO,
        "ft_titulo": "Ameaça negada por fase defensiva e por terço da posse, em risco",
        "ft_sub": "Em % da ameaça disponível nas linhas avaliadas e em % da ameaça negada total · dois jogos juntos · definição v3.1",
        "ft_rotulos": ("% da ameaça disponível que se nega", "parte da ameaça negada total"),
        "ft_nota": ("Um momento = um frame com pelo menos uma linha avaliada. Barra esmaecida: menos de {minimo} momentos. Terço = onde a posse começou "
                    "(etiqueta SkillCorner).\nA parte do total conta também os {sem_fase} momentos sem fase atribuída, que não aparecem.\n{nota_risco}"),
        "mo_moedas": {"risco": "risco (oficial)", "xt": "xT", "linhas": "linhas quebradas"},
        "mo_pares": ("risco e xT", "risco e linhas", "xT e linhas"),
        "mo_titulo": "Três moedas para o valor de uma linha, por fase defensiva",
        "mo_sub": "Esquerda: parte da ameaça negada total que cada fase reúne · direita: acordo entre moedas na ordem dos jogadores dentro de cada bloco",
        "mo_esq": "Parte da ameaça negada total",
        "mo_dir": "ρ de Spearman entre moedas",
        "mo_nota": ("Parte do total: dois jogos juntos, fases com etiqueta. ρ: por jogador dentro da posição, média ponderada, só os momentos do bloco.\n"
                    "Risco = ganho de xT + xT do adversário no ponto de corte."),
        "pa_titulo": "Pressão alta · {nome} fecha a saída do guarda-redes",
        "pa_sub": "{jogo} · {parte} · minuto {minuto} · {equipa} com a bola",
        "pa_quanto": "Quanto vale fechar esta linha",
        "pa_xt": ("Medido só pelo avanço no campo (xT), fechar a saída do guarda-redes vale {valor}: "
                  "o passe seria curto e para o lado."),
        "pa_risco": "risco",
        "pa_explica": "Perder a bola ali dá ao adversário um perigo\nenorme junto à baliza: é esse risco que o avançado\nacrescenta ao fechar a linha.",
        "pa_percentil": "{nome} no bloco alto: à frente de {percentil}% dos jogadores",
        "pa_vezes": "{vezes} vezes a média da fase (índice {indice})",
    },
    "en": {
        "nota_risco": ("Lane value = xT gain + the opponent's xT at the point where the lane would be cut (risk).\n"
                       "Lane-closure index, not a calibrated probability."),
        "ft_titulo": "Denied threat by defensive phase and by third of the possession, in risk",
        "ft_sub": ("As % of the available threat on the evaluated lanes and as % of the total denied threat · both matches together · "
                   "v3.1 definition"),
        "ft_rotulos": ("% of the available threat that is denied", "share of the total denied threat"),
        "ft_nota": ("A moment = a frame with at least one evaluated lane. Faded bar: fewer than {minimo} moments. Third = where the possession started "
                    "(SkillCorner label).\nThe share of the total also counts the {sem_fase} moments with no phase assigned, which do not appear.\n{nota_risco}"),
        "mo_moedas": {"risco": "risk (official)", "xt": "xT", "linhas": "lines broken"},
        "mo_pares": ("risk and xT", "risk and lines", "xT and lines"),
        "mo_titulo": "Three currencies for the value of a lane, by defensive phase",
        "mo_sub": ("Left: share of the total denied threat that each phase accounts for · right: agreement between currencies "
                   "on the player ranking within each block"),
        "mo_esq": "Share of the total denied threat",
        "mo_dir": "Spearman ρ between currencies",
        "mo_nota": ("Share of the total: both matches together, phases with a label. ρ: per player within position, weighted average, only the moments of the block.\n"
                    "Risk = xT gain + the opponent's xT at the cut point."),
        "pa_titulo": "High press · {nome} closes the goalkeeper's outlet",
        "pa_sub": "{jogo} · {parte} · minute {minuto} · {equipa} in possession",
        "pa_quanto": "What closing this lane is worth",
        "pa_xt": ("Measured only by advance on the pitch (xT),\nclosing the goalkeeper's outlet is worth {valor}:\n"
                  "the pass would be short and sideways."),
        "pa_risco": "risk",
        "pa_explica": "Losing the ball there gives the opponent an\nenormous threat close to goal: that is the risk\nthe forward adds by closing the lane.",
        "pa_percentil": "{nome} in the high block: ahead of {percentil}% of players",
        "pa_vezes": "{vezes} times the phase average (index {indice})",
    },
}


def ler_numeros() -> dict:
    return json.loads(exigir(SAIDA / "artigo_v31.json", "numeros_v31.py, numeros_9c.py e numeros_9d.py").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- fase e terço (parte do total e % negada)

def figura_fase_e_terco_risco(numeros: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    """Por fase defensiva e por terço da posse, em risco: % negada da ameaça disponível (g + c) e parte da ameaça negada total."""
    cor = brand.colors
    _, pct, mil = ag.formatadores(idioma)
    X, A = TEXTOS[idioma], ag.TEXTOS[idioma]
    dados = numeros["risco"]["fase_e_terco"]
    sem_fase = int(sum(r["lances"] for r in dados["fase"] if r["fase"] is None))
    fig = nova_figura(brand, X["ft_titulo"], X["ft_sub"])
    eixos = [fig.add_axes((0.20, 0.17, 0.30, 0.64)), fig.add_axes((0.71, 0.17, 0.25, 0.64))]
    tons = (cor["primary"], cor["secondary"])
    rotulos = X["ft_rotulos"]
    chaves = ("pct_negada", "pct_da_negada_total")
    maximo = max(r[c] for por in ("fase", "terco") for r in dados[por] if r[por] is not None for c in chaves)
    for ax, (por, titulo) in zip(eixos, [("fase", A["ft_fase"]), ("terco", A["ft_terco"])]):
        g = pd.DataFrame([r for r in dados[por] if r[por] is not None]).sort_values("pct_negada").reset_index(drop=True)
        estilo_eixo(ax, brand)
        for k, chave in enumerate(chaves):
            y = np.arange(len(g)) + (k - 0.5) * 0.38
            ax.barh(y, g[chave], color=[to_rgba(tons[k], 1.0 if n >= ag.MIN_LANCES else 0.4) for n in g["lances"]], height=0.36)
            for yi, v in zip(y, g[chave]):
                ax.text(v + maximo * 0.012, yi, pct(v), va="center", fontsize=11, color=cor["text"], fontweight="bold")
        ax.set_yticks(range(len(g)), [f"{A['ctx'][r[por]]}\n{A['momentos'].format(n=mil(r['lances']))}" for _, r in g.iterrows()])
        ax.set_xlim(0, maximo * 1.2)
        ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
        ax.tick_params(axis="y", labelsize=12.5, colors=cor["text"])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    for k, rotulo in enumerate(rotulos):
        x = 0.20 + k * 0.30
        fig.patches.append(plt.Rectangle((x, 0.862), 0.012, 0.022, transform=fig.transFigure, color=tons[k]))
        fig.text(x + 0.017, 0.873, rotulo, fontsize=13, color=cor["text"], va="center")
    nota(fig, brand, X["ft_nota"].format(minimo=ag.MIN_LANCES, sem_fase=sem_fase, nota_risco=X["nota_risco"]), y=0.045)
    fig.savefig(destino, dpi=ag.DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"moeda": "risco", "sem_fase": sem_fase, **{por: [r for r in dados[por]] for por in ("fase", "terco")}}


# ---------------------------------------------------------------- C4 e C5 a partir dos números

def figura_c4_risco(JG: dict, numeros: dict, brand: Brand, destino: Path, idioma: str = "pt") -> list[dict]:
    t = pd.DataFrame(numeros["risco"]["c4"])
    ag.figura_c4(JG, brand, destino, t=t, moeda="risco", idioma=idioma)
    return t.to_dict("records")


def figura_c5_risco(JG: dict, numeros: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    c = numeros["risco"]["c5"]
    n = {"rho": {"rho_media": c["rho_media"], "rho_min": c["rho_minimo"]}, "jogadores": pd.DataFrame(c["jogadores_tabela"]), "pct_mudam": c["pct_mudam"],
         "linhas": c["linhas"], "pct_negada": {int(j): tuple(v) for j, v in c["pct_negada_tronco_bola"].items()}}
    r = ag.figura_c5(JG, brand, destino, n=n, moeda="risco", idioma=idioma)
    r["moeda"] = "risco"
    return r


# ---------------------------------------------------------------- moedas por fase (três moedas)

COR_MOEDA = {"risco": "primary", "xt": "secondary", "linhas": "text_muted"}
ROTULO_MOEDA = {"risco": "risco (oficial)", "xt": "xT", "linhas": "linhas quebradas"}
PARES = (("xt_risco", "risco e xT"), ("linhas_risco", "risco e linhas"), ("xt_linhas", "xT e linhas"))
BLOCOS = ("high_block", "medium_block", "low_block")


def figura_moedas_3(numeros: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    """Esquerda: parte da ameaça negada total que cada fase defensiva reúne, em cada moeda. Direita: ρ de Spearman por jogador (dentro da posição)
    dentro de cada bloco, entre cada par de moedas."""
    cor = brand.colors
    pt, pct, _ = ag.formatadores(idioma)
    X, ctx = TEXTOS[idioma], ag.TEXTOS[idioma]["ctx"]
    por_fase = numeros["fase_9c"]["numeros_por_fase"]
    partes = {m: pd.DataFrame(por_fase["parte_da_negada_total_por_fase"][m]).dropna(subset=["fase"]).set_index("fase") for m in COR_MOEDA}
    rho = por_fase["rho_por_jogador_dentro_da_posicao"]
    ordem = partes["risco"].sort_values("pct_da_negada_total").index.tolist()
    fig = nova_figura(brand, X["mo_titulo"], X["mo_sub"])
    ax = fig.add_axes((0.20, 0.17, 0.36, 0.64))
    estilo_eixo(ax, brand)
    maximo = max(partes[m]["pct_da_negada_total"].max() for m in partes)
    for k, moeda in enumerate(COR_MOEDA):
        y = np.arange(len(ordem)) + (k - 1) * 0.27
        v = partes[moeda].loc[ordem, "pct_da_negada_total"]
        ax.barh(y, v, color=cor[COR_MOEDA[moeda]], height=0.25)
        for yi, vi in zip(y, v):
            ax.text(vi + maximo * 0.012, yi, pct(vi), va="center", fontsize=10.5, color=cor["text"], fontweight="bold" if moeda == "risco" else "normal")
    ax.set_yticks(range(len(ordem)), [ctx[f] for f in ordem])
    ax.tick_params(axis="y", labelsize=12.5, colors=cor["text"])
    ax.set_xlim(0, maximo * 1.18)
    ax.set_title(X["mo_esq"], loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    ax2 = fig.add_axes((0.67, 0.17, 0.29, 0.64))
    estilo_eixo(ax2, brand, grelha="x")
    tons_par = (cor["primary"], cor["secondary"], cor["text_muted"])
    for i, bloco in enumerate(BLOCOS):
        for k, (par, _) in enumerate(PARES):
            y = i + (k - 1) * 0.24
            v = rho[bloco][par]["media"]
            ax2.scatter([v], [y], s=130, color=tons_par[k], edgecolor=cor["background"], zorder=4)
            ax2.text(v + 0.02, y, pt(v, 2), va="center", fontsize=11, color=cor["text"])
    ax2.set_yticks(range(len(BLOCOS)), [ctx[b] for b in BLOCOS])
    ax2.tick_params(axis="y", labelsize=12.5, colors=cor["text"])
    ax2.invert_yaxis()
    ax2.set_xlim(0.2, 1.08)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 1)))
    ax2.set_title(X["mo_dir"], loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
    for k, moeda in enumerate(COR_MOEDA):
        x0 = 0.20 + k * 0.12
        fig.patches.append(plt.Rectangle((x0, 0.862), 0.012, 0.022, transform=fig.transFigure, color=cor[COR_MOEDA[moeda]]))
        fig.text(x0 + 0.017, 0.873, X["mo_moedas"][moeda], fontsize=13, color=cor["text"], va="center")
    for k, rotulo in enumerate(X["mo_pares"]):
        x0 = 0.67 + k * 0.10
        fig.patches.append(plt.Rectangle((x0, 0.862), 0.012, 0.022, transform=fig.transFigure, color=tons_par[k]))
        fig.text(x0 + 0.017, 0.873, rotulo, fontsize=11.5, color=cor["text"], va="center")
    nota(fig, brand, X["mo_nota"], y=0.05)
    fig.savefig(destino, dpi=ag.DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"fases": {f: {m: float(partes[m].loc[f, "pct_da_negada_total"]) for m in COR_MOEDA} for f in ordem},
            "rho": {b: {par: rho[b][par]["media"] for par, _ in PARES} for b in BLOCOS}, "ordem_risco": ordem}


# ---------------------------------------------------------------- caso novo: pressão alta sobre o guarda-redes

def figura_pressao_alta(J: dict, numeros: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    """O caso 1996435 frame 22256: o avançado Goodwin cobre a linha do guarda-redes para o central. Em xT a linha vale zero; em risco vale muito
    porque perder a bola ali dá ao adversário um perigo enorme. Versão para o artigo: sem texto técnico."""
    cor = brand.colors
    pt = ag.formatadores(idioma)[0]
    X = TEXTOS[idioma]
    frame = PRESSAO_ALTA["frame"]
    minuto = gc.minuto_do_frame(J["fases"], frame)
    rotulo = f"Pressão alta · {J['titulo']} · 1.ª parte · minuto {minuto}"
    lance = fl.Lance(J, frame, rotulo, brand, moeda="risco", idioma=idioma)
    lance_xt = fl.Lance(J, frame, rotulo, brand, moeda="xt", idioma=idioma)
    quem = fl.jogador_por_nome(lance.nomes, PRESSAO_ALTA["defesa"])
    caso = numeros["risco"]["caso_pressao_alta"]
    g = caso["por_defesa"][lance.nomes[quem]]
    d = lance.d
    r = lance.recetores()
    linhas_de_quem = d["atribuicoes"][(d["atribuicoes"]["defesa"] == quem)].merge(r[["recetor", "x", "y", "valor_xt", "ameaca_negada"]], on="recetor")
    alvo = linhas_de_quem.sort_values("shapley").iloc[-1]
    equipa_defesa = lance.equipa_a_defender
    fig = fl.preparar_figura(brand, X["pa_titulo"].format(nome=lance.nomes[quem].split(". ")[-1]),
                             X["pa_sub"].format(jogo=lance.titulo_jogo, parte=idi.PARTES[idioma]["1.ª parte"], minuto=minuto,
                                                equipa=lance.equipa_com_bola))
    fl.legenda_equipas(fig, lance, 0.02, 0.045)
    area = (0.02, 0.09, 0.60, 0.80)
    vista = ((-53.0, -5.0), (-12.0, 24.0))  # a área do Sydney FC, onde a ação acontece
    ax = fl.eixo_do_campo(fig, lance, area, vista)
    ox, oy = d["origem"]
    fl.desenhar_cones(ax, lance, {quem}, alpha_relevante=0.30, setas=True)
    for _, linha in r.iterrows():
        sombra = linha["recetor"] == alvo["recetor"]
        ax.plot([ox, linha["x"]], [oy, linha["y"]], color=cor["primary_ink"] if sombra else cor["text_muted"], lw=5.0 if sombra else 1.2,
                alpha=1.0 if sombra else 0.55, zorder=4 if sombra else 3, solid_capstyle="round")
    fl.desenhar_jogadores(ax, lance)
    x_q, y_q = lance.xy(quem)
    ax.scatter([x_q], [y_q], s=720, facecolor="none", edgecolor=cor["primary_ink"], lw=2.6, ls="--", zorder=6)
    et = fl.Etiquetas(ax, list(zip(d["jogadores"]["x"], d["jogadores"]["y"])))
    fl.etiquetar_jogadores(ax, lance, {lance.portador, quem, int(alvo["recetor"])}, {}, et)
    # painel: o mesmo lance medido em xT e em risco
    px = 0.65
    fig.text(px, 0.84, X["pa_quanto"], fontsize=17, fontweight="bold", color=cor["text"], va="top")
    fig.text(px, 0.77, X["pa_xt"].format(valor=pt(lance_xt.shapley.get(quem, 0.0), 3)), fontsize=12.5, color=cor["text_muted"], va="top", wrap=True)
    fig.text(px, 0.62, "xT", fontsize=14, color=cor["text_muted"], va="top")
    fig.text(px, 0.595, pt(float(lance_xt.shapley.get(quem, 0.0)), 3), fontsize=44, fontweight="bold", color=cor["secondary_ink"], va="top")
    fig.text(px, 0.47, X["pa_risco"], fontsize=14, color=cor["text_muted"], va="top")
    fig.text(px, 0.445, pt(g["negada"], 3), fontsize=44, fontweight="bold", color=cor["primary_ink"], va="top")
    fig.text(px, 0.30, X["pa_explica"], fontsize=12.5, color=cor["text_muted"], va="top")
    leitura = fl.leitura_do_indice(g["indice_na_fase"], g["percentil_na_fase"], idioma)
    fig.text(px, 0.19, X["pa_percentil"].format(nome=lance.nomes[quem].split(". ")[-1], percentil=leitura["percentil"]), fontsize=12.5,
             fontweight="bold", color=cor["text"], va="top")
    fig.text(px, 0.155, X["pa_vezes"].format(vezes=leitura["vezes"], indice=leitura["indice"]), fontsize=12, color=cor["text_muted"], va="top")
    fig.savefig(destino, dpi=fl.DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"ficheiro": destino.name, "jogo": PRESSAO_ALTA["jogo"], "frame": frame, "minuto": minuto, "defesa": lance.nomes[quem], "equipa_defesa": equipa_defesa,
            "xt": float(lance_xt.shapley.get(quem, 0.0)), "risco": g["negada"], "indice_na_fase": g["indice_na_fase"], "percentil_na_fase": g["percentil_na_fase"],
            "momentos_na_fase": g["momentos_na_fase"], "fase": caso["fase"], "linha": {"valor_risco": float(alvo["valor_xt"]), "negada_risco": float(alvo["ameaca_negada"])}}


# ---------------------------------------------------------------- tudo

def main() -> None:
    brand = load_brand()
    apply_theme(brand)
    numeros = ler_numeros()
    assert numeros.get("moeda_oficial") == "risco", "corre primeiro numeros_9d.py"
    PASTA.mkdir(parents=True, exist_ok=True)
    antigas = SAIDA / "artigo_v31_xt"
    assert antigas.exists(), ("as figuras em xT têm de estar guardadas em artigo_v31_xt/ antes de regenerar: "
                              "corre figuras_artigo.py e copia artigo/ para artigo_v31_xt/")
    n = json.loads(exigir(PASTA / "numeros_das_figuras.json", "figuras_artigo.py").read_text(encoding="utf-8"))
    JG = ag.carregar_jogos()
    ag.JG_TITULOS.update({j: JG[j]["titulo"] for j in JG})
    J6, J3, JP = gc.carregar(C06["jogo"]), gc.carregar(C03["jogo"]), gc.carregar(PRESSAO_ALTA["jogo"])
    n["c06_passos"] = {**fl.c06_passo_a_passo(J6, C06["frame"], C06["rotulo"], brand, PASTA, moeda="risco")}
    n["c06_corpo_bola"] = fl.c06_corpo_vs_bola(J6, C06["frame"], C06["rotulo"], brand, PASTA, moeda="risco")
    n["c03"] = fl.c03_lance(J3, C03["frame"], C03["rotulo"], brand, PASTA, "Klimala", "Isaias", moeda="risco")
    n["fase_terco"] = figura_fase_e_terco_risco(numeros, brand, PASTA / "agregado_fase_e_terco.png")
    n["c4"] = figura_c4_risco(JG, numeros, brand, PASTA / "agregado_c4_estabilidade.png")
    n["c5"] = figura_c5_risco(JG, numeros, brand, PASTA / "agregado_c5_tronco_vs_bola.png")
    n["versoes"] = ag.figura_versoes(brand, PASTA / "agregado_versoes.png",
                                     "\nA v3.1 oficial usa a moeda risco; esta figura mantém o xT, a moeda das versões históricas.")
    n["novas"]["moedas"] = figura_moedas_3(numeros, brand, PASTA / "novo_moedas_por_fase.png")
    n["pressao_alta"] = figura_pressao_alta(JP, numeros, brand, PASTA / "novo_pressao_alta_goodwin.png")
    n["moeda"] = "risco"
    (PASTA / "numeros_das_figuras.json").write_text(json.dumps(n, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (PASTA / "legendas.md").write_text(fa.legendas({**n, "c06_passos": {int(k): v for k, v in n["c06_passos"].items()}}), encoding="utf-8")
    if (PASTA / "numeros_orientacao.json").exists():
        acrescentar_a_legendas(PASTA)
    copia = PROJECTS / "cover_shadow" / "artigo"
    copia.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PASTA / "legendas.md", copia / "legendas.md")
    print(f"Figuras, numeros_das_figuras.json e legendas.md em {PASTA}; copia das legendas em {copia}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script refaz as figuras do artigo com a moeda oficial (risco): o lance do C06 passo a passo, o corpo contra a bola,
# o C03, a ameaça negada por fase e por terço, a estabilidade face aos parâmetros, o cone pelo tronco contra o cone para a bola e a comparação das
# três moedas. Junta uma figura nova, a de um avançado que fecha a saída do guarda-redes, onde o xT diz zero e o risco diz muito. Todos os números
# saem do ficheiro de números, para a figura e o texto não poderem discordar.
