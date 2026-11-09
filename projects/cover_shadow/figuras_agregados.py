"""Fase 6 e 9b · figuras agregadas do artigo (dois jogos, definição v3.1), a partir dos resultados guardados.

Cada figura sai de funções do notebook 05 (`avaliacao`, `metrica.agregacao`), com a mesma lógica; `NOTEBOOK`
diz a secção onde o número aparece. Forma de cada gráfico (regra de dataviz do projeto): comparações de
grandeza = barras horizontais ordenadas, um só tom; proporções de respostas = barras 100% empilhadas com a
contagem no segmento; estimativas com intervalo = pontos com IC sobre uma linha de referência; relação entre
duas medidas do mesmo jogador = pontos com a diagonal. Sem eixos duplos nem 3D, rótulos diretos.
"""

from __future__ import annotations

import math
import re
import textwrap
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import avaliacao as av
import avaliacao_v31 as a31
import idioma as idi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figuras_lance import apelido, equipa_para_mostrar
from matplotlib.colors import to_rgba
from matplotlib.ticker import FuncFormatter
from metrica import agregacao
from metrica.parametros import PADRAO
from metrica.pipeline import carregar_tabela, preparar_defesas

from footy.ingest import skillcorner as sc
from footy.paths import PROJECTS
from footy.viz.theme import Brand

DPI = 300
FIGSIZE = (16, 9)
FONTE = "SkillCorner Open Data (MIT)"
JOGOS = sc.JOGOS_COM_POSE
VERSOES = ("v1", "v2", "v3", "v3_1")
ATUAL = "v3_1"  # a definição do artigo (D43)
ROTULO_VERSAO = {"v1": "v1", "v2": "v2", "v3": "v3", "v3_1": "v3.1"}
DESTAQUE_FASES = ("high_block", "medium_block")
MIN_LANCES = 50  # grupos com menos lances ficam a cinzento: pouca amostra

ROTULOS_CONTEXTO = {
    "attacking_third": "terço ofensivo", "middle_third": "terço médio", "defensive_third": "terço defensivo",
    "high_block": "bloco alto", "medium_block": "bloco médio", "low_block": "bloco baixo", "chaotic": "caótica",
    "defending_direct": "defesa de jogo direto", "defending_set_play": "bola parada",
    "defending_quick_break": "contra-ataque", "defending_transition": "transição defensiva"}
NOMES_PARAMETROS = {
    "angulo_cone": "Ângulo do cone", "comprimento_cone": "Comprimento do cone", "largura_junto_ao_corpo": "Largura junto ao corpo",
    "tempo_reacao": "Tempo de reação", "velocidade_bola": "Velocidade da bola", "incerteza_tempo": "Incerteza do tempo",
    "distancia_marcacao": "Marcação: distância de entrada", "tempo_marcacao": "Marcação: tempo mínimo",
    "raio_bloqueio_corpo": "Bloqueio pelo corpo: raio fixo", "comprimento_maximo_linha": "Comprimento máximo da linha"}
UNIDADES = {"angulo_cone": "°", "comprimento_cone": " m", "largura_junto_ao_corpo": " m", "tempo_reacao": " s",
            "velocidade_bola": " m/s", "incerteza_tempo": " s", "distancia_marcacao": " m", "tempo_marcacao": " s",
            "raio_bloqueio_corpo": " m", "comprimento_maximo_linha": " m"}


def pt(valor: float, casas: int = 1, idioma: str = "pt") -> str:
    return idi.numero(valor, casas, idioma, menos=True)


def formatadores(idioma: str = "pt"):
    """(pt, pct, mil) do idioma, com o sinal de menos tipográfico das figuras agregadas."""
    return idi.formatadores(idioma, menos=True)


# Textos das figuras agregadas, por idioma (os números formatam-se à parte). Os nomes das chaves ficam em português.
TEXTOS = {
    "pt": {
        "ctx": idi.ROTULOS_CONTEXTO["pt"],
        "momentos": "{n} momentos",
        "moeda_xt": "moeda xT (ganho de xT)",
        "moeda_linhas": "moeda linhas quebradas (adversários ultrapassados)",
        # fase e terço (xT / linhas)
        "ft_titulo": "Ameaça negada por fase defensiva e por terço da posse, em duas moedas",
        "ft_sub": "Em % da ameaça disponível nas linhas avaliadas · dois jogos juntos · definição v3.1",
        "ft_fase": "Fase defensiva",
        "ft_terco": "Terço da posse",
        "ft_nota": ("Um momento = um frame com pelo menos uma linha avaliada. Barra esmaecida: menos de {minimo} momentos. "
                    "Terço = onde a posse começou (etiqueta SkillCorner).\nÍndice de fecho da linha, não probabilidade calibrada. "
                    "Fora da figura: {sem_fase} momentos sem fase atribuída."),
        # revisões
        "rv_titulo": "Revisão de lances por um treinador, em três rondas",
        "rv_sub": ("Respostas a duas perguntas sobre cada lance (15, 15 e 8 lances); cada ronda com a definição "
                   "da métrica desse momento"),
        "rv_sombras": "As sombras fazem sentido?",
        "rv_marcacao": "A marcação está certa?",
        "rv_rondas": ("1.ª ronda · v1", "2.ª ronda · v2", "3.ª ronda · v3"),
        "rv_respostas": ("sim", "em parte", "não"),
        "rv_nota": "Número de lances em cada resposta; barras a 100 %. As rondas usam lances diferentes (sem repetir posses).",
        # C3
        "c3_titulo": "Sombra da linha seguida face às alternativas",
        "c3_sub": ("Sombra da linha seguida menos sombra das outras linhas do mesmo momento, com intervalo de confiança a 95 % · "
                   "dois jogos juntos"),
        "c3_controlada": "Alternativas comparáveis\n(mesmo comprimento e perigo)",
        "c3_sem_controlo": "Todas as alternativas\ndo mesmo momento",
        "c3_eixo": "sombra da linha seguida − sombra das alternativas (0 = igual; negativo = menos sombra)",
        "c3_nota": ("v1: primeira avaliação · v2: depois da 1.ª revisão · v3: definição anterior · v3.1: definição do artigo (linhas até 35 m). "
                    "Sombra = 1 − índice de chegada. Passes com pelo menos uma linha comparável."),
        # C4
        "c4_titulo": "Estabilidade da ordem dos jogadores face aos parâmetros",
        "c4_sub": ("Correlação de Spearman (dentro de cada posição) entre a ameaça negada por jogador da definição v3.1 e a de cada "
                   "variante{risco} · dois jogos"),
        "c4_risco": " · moeda risco",
        "c4_parametros": {
            "angulo_cone": "Ângulo do cone", "comprimento_cone": "Comprimento do cone", "largura_junto_ao_corpo": "Largura junto ao corpo",
            "tempo_reacao": "Tempo de reação", "velocidade_bola": "Velocidade da bola", "incerteza_tempo": "Incerteza do tempo",
            "distancia_marcacao": "Marcação: distância de entrada", "tempo_marcacao": "Marcação: tempo mínimo",
            "raio_bloqueio_corpo": "Bloqueio pelo corpo: raio fixo", "comprimento_maximo_linha": "Comprimento máximo da linha"},
        "c4_v31": "v3.1 (ρ = 1)",
        "c4_eixo": "ρ de Spearman face à v3.1, por jogador (média ponderada pelas posições)",
        "c4_nota": ("Cada ponto = uma variante, mudando um parâmetro de cada vez e mantendo o resto da v3.1. "
                    "Eixo começa em {inferior} (o valor mais baixo é {minimo})."),
        # C5
        "c5_titulo": "Ameaça negada por jogador: cone pelo tronco vs cone virado para a bola",
        "c5_sub": ("Ameaça negada por jogador e jogo com o cone pelo tronco e com o cone virado para a bola · definição v3.1 · moeda {unidade}"),
        "c5_unidade_risco": "risco",
        "c5_eixo_x": "cone pelo tronco: índice de ameaça negada (soma do valor das linhas, {unidade})",
        "c5_eixo_y": "cone virado para a bola: índice de ameaça negada\n(soma do valor das linhas, {unidade})",
        "c5_diagonal": "acima da diagonal:\nmais com o cone para a bola",
        "c5_um_ponto": "Um ponto = um jogador num jogo",
        "c5_rho": "ρ de Spearman por jogador: {rho}",
        "c5_rho_min": "(mínimo entre posições: {minimo})",
        "c5_muda": "Linhas em que muda quem faz sombra: {pct}",
        "c5_linhas": "({n} linhas avaliadas)",
        "c5_pct_negada": "Ameaça negada em % da disponível",
        "c5_por_jogo": "{jogo}: {tronco} tronco · {bola} bola",
        "c5_nota": ("Índice, não golos: a soma por jogador junta o valor de linhas alternativas do mesmo momento. "
                    "Os defesas sem pose ficam sem orientação nas duas versões: as linhas avaliadas são as mesmas."),
        # versões
        "vs_titulo": "Ameaça negada e linhas excluídas por versão",
        "vs_sub": "Ameaça negada (em % da disponível) e linhas excluídas (em % das linhas) em cada versão, nos dois jogos",
        "vs_paineis": ("Ameaça negada (% da disponível)", "Linhas excluídas (% das linhas)"),
        "vs_x": ("v1\nprimeira\navaliação", "v2\ndepois da\n1.ª revisão", "v3\ndefinição\nanterior", "v3.1\ndefinição\ndo artigo"),
        "vs_nota": ("v2: guarda-redes fora, fora de jogo, cruzamentos, pressão, continuidade e bloqueio pelo corpo · "
                    "v3: marcação a 4 m com tolerância e bloqueio gradual · v3.1: v3 sem as linhas com mais de 35 m (contam como excluídas)."),
        "vs_nota_final": "\nA v3.1 oficial usa a moeda risco; esta figura mantém o xT, a moeda das versões históricas.",
    },
    "en": {
        "ctx": idi.ROTULOS_CONTEXTO["en"],
        "momentos": "{n} moments",
        "moeda_xt": "xT currency (xT gain)",
        "moeda_linhas": "lines-broken currency (opponents bypassed)",
        "ft_titulo": "Denied threat by defensive phase and by third of the possession, in two currencies",
        "ft_sub": "As % of the available threat on the evaluated lanes · both matches together · v3.1 definition",
        "ft_fase": "Defensive phase",
        "ft_terco": "Third of the possession",
        "ft_nota": ("A moment = a frame with at least one evaluated lane. Faded bar: fewer than {minimo} moments. "
                    "Third = where the possession started (SkillCorner label).\nLane-closure index, not a calibrated probability. "
                    "Outside the figure: {sem_fase} moments with no phase assigned."),
        "rv_titulo": "Review of plays by a coach, in three rounds",
        "rv_sub": ("Answers to two questions about each play (15, 15 and 8 plays); each round with the metric definition "
                   "of that moment"),
        "rv_sombras": "Do the shadows make sense?",
        "rv_marcacao": "Is the marking right?",
        "rv_rondas": ("1st round · v1", "2nd round · v2", "3rd round · v3"),
        "rv_respostas": ("yes", "partly", "no"),
        "rv_nota": "Number of plays in each answer; bars at 100%. The rounds use different plays (no repeated possessions).",
        "c3_titulo": "Shadow on the lane followed vs the alternatives",
        "c3_sub": ("Shadow on the lane followed minus shadow on the other lanes of the same moment, with a 95% confidence interval · "
                   "both matches together"),
        "c3_controlada": "Comparable alternatives\n(same length and threat)",
        "c3_sem_controlo": "All alternatives\nof the same moment",
        "c3_eixo": "shadow on the lane followed − shadow on the alternatives (0 = equal; negative = less shadow)",
        "c3_nota": ("v1: first evaluation · v2: after the 1st review · v3: previous definition · v3.1: article definition (lanes up to 35 m). "
                    "Shadow = 1 − arrival index. Passes with at least one comparable lane."),
        "c4_titulo": "Stability of the player ranking across parameters",
        "c4_sub": ("Spearman correlation (within each position) between the denied threat per player in v3.1 and in each "
                   "variant{risco} · both matches"),
        "c4_risco": " · risk currency",
        "c4_parametros": {
            "angulo_cone": "Cone angle", "comprimento_cone": "Cone length", "largura_junto_ao_corpo": "Width at the body",
            "tempo_reacao": "Reaction time", "velocidade_bola": "Ball speed", "incerteza_tempo": "Time uncertainty",
            "distancia_marcacao": "Marking: entry distance", "tempo_marcacao": "Marking: minimum time",
            "raio_bloqueio_corpo": "Body block: fixed radius", "comprimento_maximo_linha": "Maximum lane length"},
        "c4_v31": "v3.1 (ρ = 1)",
        "c4_eixo": "Spearman ρ vs v3.1, per player (average weighted by position)",
        "c4_nota": ("Each point = one variant, changing one parameter at a time and keeping the rest of v3.1. "
                    "Axis starts at {inferior} (the lowest value is {minimo})."),
        "c5_titulo": "Denied threat per player: trunk-based cone vs ball-facing cone",
        "c5_sub": ("Denied threat per player and match with the trunk-based cone and with the ball-facing cone · v3.1 definition · {unidade} currency"),
        "c5_unidade_risco": "risk",
        "c5_eixo_x": "trunk-based cone: denied-threat index (sum of lane values, {unidade})",
        "c5_eixo_y": "ball-facing cone: denied-threat index\n(sum of lane values, {unidade})",
        "c5_diagonal": "above the diagonal:\nmore with the ball-facing cone",
        "c5_um_ponto": "One point = one player in one match",
        "c5_rho": "Spearman ρ per player: {rho}",
        "c5_rho_min": "(minimum across positions: {minimo})",
        "c5_muda": "Lanes where the shadow caster changes: {pct}",
        "c5_linhas": "({n} lanes evaluated)",
        "c5_pct_negada": "Denied threat as % of the available",
        "c5_por_jogo": "{jogo}: {tronco} trunk · {bola} ball",
        "c5_nota": ("Index, not goals: the per-player sum adds up alternative lanes of the same moment. "
                    "Defenders without pose have no orientation in either version: same evaluated lanes."),
        "vs_titulo": "Denied threat and excluded lanes by version",
        "vs_sub": "Denied threat (as % of the available) and excluded lanes (as % of lanes) in each version, in both matches",
        "vs_paineis": ("Denied threat (% of the available)", "Excluded lanes (% of lanes)"),
        "vs_x": ("v1\nfirst\nevaluation", "v2\nafter the\n1st review", "v3\nprevious\ndefinition", "v3.1\narticle\ndefinition"),
        "vs_nota": ("v2: goalkeeper out, offside, crosses, pressure, continuity and body block · "
                    "v3: marking at 4 m with tolerance and gradual block · v3.1: v3 without lanes longer than 35 m (counted as excluded)."),
        "vs_nota_final": "\nThe official v3.1 uses the risk currency; this figure keeps xT, the currency of the historical versions.",
    },
}


def texto(idioma: str, chave: str, **valores):
    """Texto (ou tuplo de textos) de uma figura; só se formata o que é texto."""
    t = TEXTOS[idioma][chave]
    return t.format(**valores) if isinstance(t, str) else t


# ---------------------------------------------------------------- dados

def carregar_jogos() -> dict:
    """Tudo o que as figuras precisam de cada jogo (v3.1): tabela, defesas, jogadores, eventos e fases."""
    jogos = {}
    for jogo in JOGOS:
        match = sc.ler_match(jogo)
        tabela = carregar_tabela(jogo)
        jogadores = pd.DataFrame([{"jogador": p["id"], "nome": p["short_name"], "equipa": p["team_id"],
                                   "posicao": p["player_role"]["acronym"]} for p in match["players"]])
        jogos[jogo] = {"jogo": jogo, "match": match, "tabela": tabela, "defesas": preparar_defesas(tabela, match, PADRAO),
                       "jogadores": jogadores, "eventos": sc.ler_eventos(jogo), "fases": sc.ler_fases(jogo),
                       "titulo": equipa_para_mostrar(f"{match['home_team']['short_name']} – {match['away_team']['short_name']}")}
    return jogos


def ler(jogo: int, variante: str, versao: str):
    return av.ler_variante(jogo, variante, av.PASTAS[versao])


def lances_da_variante(variante: str, versao: str) -> pd.DataFrame:
    return pd.concat([agregacao.por_lance(ler(j, variante, versao)[0]).assign(jogo=j) for j in JOGOS], ignore_index=True)


def jogadores_da_variante(JG: dict, variante: str, versao: str) -> pd.DataFrame:
    partes = []
    for j in JOGOS:
        linhas, atribuicoes = ler(j, variante, versao)
        presenca = JG[j]["defesas"][["frame", "posse_id", "jogador"]]
        partes.append(av.negada_por_jogador(linhas, atribuicoes, presenca, JG[j]["jogadores"]).assign(jogo=j))
    return pd.concat(partes, ignore_index=True)


def ler_folha_xlsx(caminho: Path) -> pd.DataFrame:
    """Folha preenchida da revisão (lance, sombras, marcação), lida do .xlsx sem dependências extra."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(caminho)
    partilhadas = ["".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
                   for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    linhas = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        celulas = {}
        for c in r.findall("m:c", ns):
            v = c.find("m:v", ns)
            valor = "" if v is None else (partilhadas[int(v.text)] if c.get("t") == "s" else v.text)
            celulas[re.match(r"[A-Z]+", c.get("r")).group()] = valor
        linhas.append(celulas)
    tabela = pd.DataFrame([{"lance": r.get("A", ""), "sombras": r.get("D", "").strip().lower(),
                            "marcacao": r.get("E", "").strip().lower()} for r in linhas])
    return tabela[tabela["lance"].str.match(r"^[LVW]\d\d$")].reset_index(drop=True)


# ---------------------------------------------------------------- base das figuras

def nova_figura(brand: Brand, titulo: str, subtitulo: str):
    cor = brand.colors
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=cor["background"])
    fig.text(0.04, 0.965, titulo, fontsize=24, fontweight="bold", color=cor["text"], va="top")
    fig.text(0.04, 0.913, subtitulo, fontsize=13.5, color=cor["text_muted"], va="top")
    fig.text(0.5, 0.015, brand.credit(FONTE), ha="center", va="bottom", color=cor["text_muted"], fontsize=11)
    return fig


def estilo_eixo(ax, brand: Brand, grelha: str = "x") -> None:
    cor = brand.colors
    ax.set_facecolor(cor["background"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(cor["lines"])
    ax.tick_params(colors=cor["text_muted"], labelsize=12, length=0)
    if grelha:
        ax.grid(axis=grelha, color=cor["lines"], alpha=0.45, lw=0.8)
        ax.set_axisbelow(True)


def nota(fig, brand: Brand, texto: str, y: float = 0.055, largura: int = 150) -> None:
    """Nota de rodapé; cada parágrafo parte-se em linhas de ~`largura` caracteres para não sair pela margem direita."""
    quebrado = "\n".join("\n".join(textwrap.wrap(p, largura)) for p in texto.split("\n"))
    fig.text(0.04, y, quebrado, fontsize=11, color=brand.colors["text_muted"], va="bottom")


# ---------------------------------------------------------------- 1. por fase defensiva e por terço

MOEDAS_ROTULO = {"xt": "moeda xT (ganho de xT)", "linhas": "moeda linhas quebradas (adversários ultrapassados)"}


def tabelas_da_moeda(moeda: str, versao: str = ATUAL) -> dict:
    """{jogo: (linhas, atribuições)} da base da versão, na moeda pedida (`xt` ou `linhas`, ver `avaliacao_v31`)."""
    saida = {}
    for j in JOGOS:
        linhas, atribuicoes = ler(j, "base", versao)
        saida[j] = (linhas, atribuicoes) if moeda == "xt" else a31.moeda_linhas(linhas, atribuicoes, a31.ultrapassados_guardados(j))
    return saida


def ameaca_por_contexto(JG: dict) -> tuple[dict, int]:
    """Ameaça negada (v3.1, dois jogos juntos) por fase defensiva e por terço da posse, nas duas moedas: lances, disponível, negada, %."""
    contexto = pd.concat([agregacao.contexto_dos_frames(JG[j]["tabela"], JG[j]["fases"]).assign(jogo=j) for j in JOGOS],
                         ignore_index=True)
    saida, sem_fase = {}, 0
    for moeda in a31.MOEDAS:
        tabelas = tabelas_da_moeda(moeda)
        lances = pd.concat([agregacao.por_lance(tabelas[j][0]).assign(jogo=j) for j in JOGOS], ignore_index=True)
        saida[moeda] = {}
        for por in ("fase", "terco"):
            g = a31.pct_por_contexto(lances, contexto, por)
            g["rotulo"] = g[por].map(ROTULOS_CONTEXTO)
            if por == "fase":  # os lances sem fase atribuída ficam fora da figura
                sem_fase = int(g.loc[g["rotulo"].isna(), "lances"].sum())
                g = g[g["rotulo"].notna()]
            saida[moeda][por] = g
    return saida, sem_fase


def figura_fase_e_terco(JG: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    cor = brand.colors
    _pt, pct, mil = formatadores(idioma)
    T = lambda chave, **v: texto(idioma, chave, **v)
    dados, sem_fase = ameaca_por_contexto(JG)
    fig = nova_figura(brand, T("ft_titulo"), T("ft_sub"))
    eixos = [fig.add_axes((0.20, 0.17, 0.30, 0.64)), fig.add_axes((0.71, 0.17, 0.25, 0.64))]
    tons = {"xt": cor["primary"], "linhas": cor["secondary"]}
    maximo = max(g["pct_negada"].max() for m in dados.values() for g in m.values())
    for ax, (por, titulo) in zip(eixos, [("fase", T("ft_fase")), ("terco", T("ft_terco"))]):
        ordem = dados["xt"][por].sort_values("pct_negada", ascending=True).reset_index(drop=True)
        estilo_eixo(ax, brand)
        for k, moeda in enumerate(a31.MOEDAS):
            g = dados[moeda][por].set_index(por).loc[ordem[por]].reset_index()
            y = np.arange(len(g)) + (k - 0.5) * 0.38
            alphas = [1.0 if n >= MIN_LANCES else 0.4 for n in g["lances"]]
            ax.barh(y, g["pct_negada"], color=[to_rgba(tons[moeda], a) for a in alphas], height=0.36)
            for yi, v in zip(y, g["pct_negada"]):
                ax.text(v + maximo * 0.012, yi, pct(v), va="center", fontsize=11, color=cor["text"], fontweight="bold")
        ax.set_yticks(range(len(ordem)), [f"{TEXTOS[idioma]['ctx'][r[por]]}\n{T('momentos', n=mil(r.lances))}" for _, r in ordem.iterrows()])
        ax.set_xlim(0, maximo * 1.2)
        ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
        ax.tick_params(axis="y", labelsize=12.5, colors=cor["text"])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
    for k, moeda in enumerate(a31.MOEDAS):
        x = 0.20 + k * 0.30
        fig.patches.append(plt.Rectangle((x, 0.862), 0.012, 0.022, transform=fig.transFigure, color=tons[moeda]))
        fig.text(x + 0.017, 0.873, T("moeda_" + moeda), fontsize=13, color=cor["text"], va="center")
    nota(fig, brand, T("ft_nota", minimo=MIN_LANCES, sem_fase=sem_fase), y=0.045)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"sem_fase": sem_fase, **{m: {por: g.drop(columns=["rotulo"]).to_dict("records") for por, g in d.items()} for m, d in dados.items()}}


# ---------------------------------------------------------------- 2. as três rondas de revisão

RONDAS = [("1.ª ronda · v1", "revisao_lances_preenchida.xlsx"), ("2.ª ronda · v2", "revisao_lances_preenchidas_v2.xlsx"),
          ("3.ª ronda · v3", "revisao_lances_preenchida_v3.xlsx")]
RESPOSTAS = ("sim", "em parte", "não")


def figura_revisoes(brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    cor = brand.colors
    T = lambda chave, **v: texto(idioma, chave, **v)
    cores = {"sim": cor["secondary"], "em parte": cor["text_muted"], "não": cor["text"]}
    tinta = {"sim": cor["background"], "em parte": cor["background"], "não": cor["background"]}
    folhas = {nome: ler_folha_xlsx(PROJECTS / "cover_shadow" / ficheiro) for nome, ficheiro in RONDAS}
    fig = nova_figura(brand, T("rv_titulo"), T("rv_sub"))
    eixos = [fig.add_axes((0.20, 0.20, 0.35, 0.62)), fig.add_axes((0.62, 0.20, 0.35, 0.62))]
    numeros = {}
    for ax, (coluna, titulo) in zip(eixos, [("sombras", T("rv_sombras")), ("marcacao", T("rv_marcacao"))]):
        estilo_eixo(ax, brand, grelha="")
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks([])
        for i, (nome, folha) in enumerate(reversed(list(folhas.items()))):
            n = len(folha)
            esquerda = 0.0
            contagem = {r: int((folha[coluna] == r).sum()) for r in RESPOSTAS}
            assert sum(contagem.values()) == n, (nome, coluna, contagem)
            numeros.setdefault(nome, {})[coluna] = {**contagem, "n": n}
            for r in RESPOSTAS:
                if contagem[r]:
                    ax.barh(i, contagem[r] / n * 100, left=esquerda, height=0.62, color=cores[r], edgecolor=cor["background"],
                            linewidth=2.5)
                    ax.text(esquerda + contagem[r] / n * 50, i, f"{contagem[r]}", ha="center", va="center", fontsize=15,
                            fontweight="bold", color=tinta[r])
                esquerda += contagem[r] / n * 100
        ax.set_yticks(range(len(folhas)), [T("rv_rondas")[list(folhas).index(nome)] for nome in reversed(list(folhas))])
        ax.tick_params(axis="y", labelsize=13, colors=cor["text"])
        ax.set_xlim(0, 100)
        ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
        if ax is eixos[1]:
            ax.set_yticklabels([])
    # legenda direta, por cima do primeiro painel
    x = 0.20
    for r in RESPOSTAS:
        fig.patches.append(plt.Rectangle((x, 0.115), 0.012, 0.022, transform=fig.transFigure, color=cores[r]))
        fig.text(x + 0.017, 0.126, T("rv_respostas")[RESPOSTAS.index(r)], fontsize=13, color=cor["text"], va="center")
        x += 0.09
    nota(fig, brand, T("rv_nota"), y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return numeros


# ---------------------------------------------------------------- 3. C3

def passes_da_versao(JG: dict, versao: str) -> pd.DataFrame:
    """Passes reais com a sombra da linha seguida e as linhas comparáveis (como em `passes_da_versao` do notebook 05)."""
    partes = []
    for j in JOGOS:
        linhas = av.adicionar_comprimento(ler(j, "base", versao)[0], JG[j]["tabela"])
        com_linha = av.passes_com_linha(av.passes_do_jogo(JG[j]["eventos"]), linhas)
        partes.append(av.com_comparaveis(com_linha, linhas).assign(jogo=j))
    return pd.concat(partes, ignore_index=True)


def numeros_c3(JG: dict) -> dict:
    saida = {}
    for v in VERSOES:
        passes = passes_da_versao(JG, v)
        saida[v] = av.comparar_com_comparaveis(passes)
        baixa = passes[passes["p_chegar"] < 0.5]
        taxa, _, _ = av.intervalo_wilson(int(baixa["completo"].sum()), len(baixa))
        saida[v].update({"n_baixa": len(baixa), "completos_baixa": int(baixa["completo"].sum()), "taxa_baixa": taxa})
    return saida


def figura_c3(JG: dict, brand: Brand, destino: Path, idioma: str = "pt") -> dict:
    cor = brand.colors
    pt = formatadores(idioma)[0]
    T = lambda chave, **v: texto(idioma, chave, **v)
    n = numeros_c3(JG)
    fig = nova_figura(brand, T("c3_titulo"), T("c3_sub"))
    ax = fig.add_axes((0.24, 0.20, 0.70, 0.60))
    estilo_eixo(ax, brand, grelha="x")
    linhas = []  # (rotulo, chave, versao)
    for medida, titulo in (("dif_controlada", T("c3_controlada")), ("dif_sem_controlo", T("c3_sem_controlo"))):
        for v in (ATUAL, "v3", "v2", "v1"):
            linhas.append((titulo if v == ATUAL else "", medida, v))
    y = []
    posicao = 0.0
    for i, (rotulo, medida, v) in enumerate(linhas):
        if i == len(VERSOES):
            posicao += 0.8
        y.append(posicao)
        posicao += 1
    for (rotulo, medida, v), yi in zip(linhas, y):
        r = n[v]
        valor, baixo, alto = r[medida], r[medida + "_inf"], r[medida + "_sup"]
        destaque = v == ATUAL
        ax.plot([baixo, alto], [yi, yi], color=cor["primary_ink"] if destaque else cor["text_muted"], lw=3, solid_capstyle="round", zorder=3)
        ax.scatter([valor], [yi], s=170, color=cor["primary"] if destaque else cor["text_muted"], edgecolor=cor["background"],
                   zorder=4)
        ax.text(alto + 0.004, yi, f"{pt(valor, 3)}  [{pt(baixo, 3)}; {pt(alto, 3)}]", va="center", fontsize=12.5,
                color=cor["text"] if destaque else cor["text_muted"], fontweight="bold" if destaque else "normal",
                bbox={"facecolor": cor["background"], "edgecolor": "none", "pad": 1.5}, zorder=5)
    ax.axvline(0, color=cor["text"], lw=1.2, zorder=2)
    ax.set_yticks(y, [f"{ROTULO_VERSAO[v]} · n = {n[v]['n']}" for _, _, v in linhas])
    ax.tick_params(axis="y", labelsize=13, colors=cor["text"])
    ax.invert_yaxis()
    ax.set_xlim(-0.20, 0.06)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 2)))
    ax.set_xlabel(T("c3_eixo"), fontsize=12.5, color=cor["text_muted"], labelpad=10)
    for i, (rotulo, _, _) in enumerate(linhas):
        if rotulo:
            ax.text(-0.205, y[i] - 0.62, rotulo.replace("\n", " "), fontsize=13.5, fontweight="bold", color=cor["text"], va="center",
                    ha="left", transform=ax.transData)
    nota(fig, brand, T("c3_nota"))
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return n


# ---------------------------------------------------------------- 4. C4

def numeros_c4(JG: dict) -> pd.DataFrame:
    """ρ de Spearman por jogador (dentro da posição) de cada variante face à v3, e % negada por jogo."""
    base = jogadores_da_variante(JG, "base", ATUAL)
    linhas = []
    for parametro, valores in {**av.VALORES_ANTIGOS_V3, "raio_bloqueio_corpo": av.RAIOS_BLOQUEIO_V3,
                               "comprimento_maximo_linha": av.LIMITES_COMPRIMENTO_V31}.items():
        for valor in valores:
            nome = av.nome_da_variante(parametro, valor)
            rho = av.spearman_por_posicao(base, jogadores_da_variante(JG, nome, ATUAL))["rho_media"]
            linhas.append({"parametro": parametro, "valor": valor, "variante": nome, "rho_jogador": rho,
                           **{f"pct_{j}": av.pct_negada(ler(j, nome, ATUAL)[0]) for j in JOGOS}})
    return pd.DataFrame(linhas)


def limite_inferior_c4(rho_minimo: float, folga: float = 0.01) -> float:
    """Início do eixo da C4: o múltiplo de 0,01 imediatamente abaixo do mínimo menos uma folga (para o ponto e o número não tocarem o eixo)."""
    return math.floor(round((rho_minimo - folga) * 100, 6)) / 100


def figura_c4(JG: dict, brand: Brand, destino: Path, t: pd.DataFrame | None = None, moeda: str = "xt", idioma: str = "pt") -> pd.DataFrame:
    """`t`: tabela de `numeros_c4` (por omissão calcula-se em xT); `moeda`: só muda o subtítulo («risco» na D44)."""
    cor = brand.colors
    pt = formatadores(idioma)[0]
    T = lambda chave, **v: texto(idioma, chave, **v)
    nomes_parametros = TEXTOS[idioma]["c4_parametros"]
    t = numeros_c4(JG) if t is None else t
    fig = nova_figura(brand, T("c4_titulo"), T("c4_sub", risco=T("c4_risco") if moeda == "risco" else ""))
    ax = fig.add_axes((0.30, 0.15, 0.64, 0.70))
    estilo_eixo(ax, brand, grelha="x")
    ordem = list(NOMES_PARAMETROS)
    y, rotulos, posicao, bandas = [], [], 0.0, []
    for k, parametro in enumerate(ordem):
        g = t[t["parametro"] == parametro].sort_values("valor")
        inicio = posicao
        for r in g.itertuples():
            y.append(posicao)
            rotulos.append(f"{nomes_parametros[parametro]} · {pt(r.valor, 0 if r.valor == int(r.valor) else 1)}{UNIDADES[parametro]}")
            posicao += 1
        bandas.append((inicio - 0.5, posicao - 0.5, k % 2 == 0))
        posicao += 0.4
    for inicio, fim, escura in bandas:
        if escura:
            ax.axhspan(inicio, fim, color=cor["lines"], alpha=0.14, lw=0)
    ordenado = pd.concat([t[t["parametro"] == p].sort_values("valor") for p in ordem])
    ax.scatter(ordenado["rho_jogador"], y, s=130, color=cor["primary"], edgecolor=cor["background"], zorder=4)
    for rho, yi in zip(ordenado["rho_jogador"], y):
        ax.text(rho + 0.002, yi, f"  {pt(rho, 3)}", va="center", fontsize=11.5, color=cor["text"],
                bbox={"facecolor": cor["background"], "edgecolor": "none", "pad": 1.0}, zorder=5)
    ax.axvline(1.0, color=cor["text"], lw=1.2, ls=(0, (4, 3)))
    ax.text(1.0, -1.0, T("c4_v31"), ha="center", fontsize=11.5, color=cor["text"],
            bbox={"facecolor": cor["background"], "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(y, rotulos)
    ax.tick_params(axis="y", labelsize=11.5, colors=cor["text"])
    ax.invert_yaxis()
    minimo = float(t["rho_jogador"].min())
    inferior = limite_inferior_c4(minimo)
    ax.set_xlim(inferior, 1.012)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: pt(v, 2)))
    ax.set_xlabel(T("c4_eixo"), fontsize=12.5, color=cor["text_muted"], labelpad=10)
    nota(fig, brand, T("c4_nota", inferior=pt(inferior, 2), minimo=pt(minimo, 3)), y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return t


# ---------------------------------------------------------------- 5. C5

def numeros_c5(JG: dict, versao: str = ATUAL) -> dict:
    tronco, bola = jogadores_da_variante(JG, "base", versao), jogadores_da_variante(JG, "bola", versao)
    rho = av.spearman_por_posicao(tronco, bola)
    juntos = tronco.merge(bola, on=["jogo", "jogador", "posicao", "nome"], suffixes=("_tronco", "_bola"))
    mudancas = pd.concat([av.linhas_que_mudam(*ler(j, "base", versao)[:1], ler(j, "bola", versao)[0], ler(j, "base", versao)[1],
                                              ler(j, "bola", versao)[1]).assign(jogo=j) for j in JOGOS], ignore_index=True)
    return {"rho": rho, "jogadores": juntos, "pct_mudam": 100 * mudancas["muda"].mean(), "linhas": len(mudancas),
            "pct_negada": {j: (av.pct_negada(ler(j, "base", versao)[0]), av.pct_negada(ler(j, "bola", versao)[0])) for j in JOGOS}}


def figura_c5(JG: dict, brand: Brand, destino: Path, n: dict | None = None, moeda: str = "xt", idioma: str = "pt") -> dict:
    """`n`: saída de `numeros_c5` (por omissão calcula-se em xT); `moeda`: «xt» ou «risco» (só muda os textos)."""
    cor = brand.colors
    pt, pct, mil = formatadores(idioma)
    T = lambda chave, **v: texto(idioma, chave, **v)
    n = numeros_c5(JG) if n is None else n
    unidade = "xT" if moeda == "xt" else T("c5_unidade_risco")
    j = n["jogadores"].copy()
    nomes = {jg: JG[jg]["jogadores"].set_index("jogador")["nome"].to_dict() for jg in JG}
    fig = nova_figura(brand, T("c5_titulo"), T("c5_sub", unidade=unidade))
    ax = fig.add_axes((0.07, 0.14, 0.50, 0.71))
    estilo_eixo(ax, brand, grelha="both")
    maximo = float(j[["negada_tronco", "negada_bola"]].max().max()) * 1.06
    ax.plot([0, maximo], [0, maximo], color=cor["text_muted"], lw=1.3, ls=(0, (4, 3)), zorder=2)
    ax.scatter(j["negada_tronco"], j["negada_bola"], s=95, color=cor["primary"], alpha=0.85, edgecolor=cor["background"], zorder=3)
    j["dif"] = (j["negada_bola"] - j["negada_tronco"]).abs()
    for r in j.nlargest(4, "dif").itertuples():
        ax.annotate(apelido(nomes[r.jogo], r.jogador), (r.negada_tronco, r.negada_bola), xytext=(8, 8), textcoords="offset points",
                    fontsize=11.5, color=cor["text"])
    ax.set_xlim(0, maximo)
    ax.set_ylim(0, maximo)
    ax.set_aspect("equal")
    ax.set_xlabel(T("c5_eixo_x", unidade=unidade), fontsize=13, color=cor["text_muted"], labelpad=8)
    ax.set_ylabel(T("c5_eixo_y", unidade=unidade), fontsize=13, color=cor["text_muted"], labelpad=8)
    # canto superior esquerdo: zona acima da diagonal onde não há pontos nem a própria diagonal
    ax.text(maximo * 0.03, maximo * 0.97, T("c5_diagonal"), ha="left", va="top",
            fontsize=11.5, color=cor["text_muted"])
    # painel de números à direita
    px = 0.63
    linhas = [(T("c5_um_ponto"), None),
              (T("c5_rho", rho=pt(n["rho"]["rho_media"], 3)), True),
              (T("c5_rho_min", minimo=pt(n["rho"]["rho_min"], 3)), False),
              ("", False),
              (T("c5_muda", pct=pct(n["pct_mudam"])), True),
              (T("c5_linhas", n=mil(n["linhas"])), False),
              ("", False), (T("c5_pct_negada"), True)]
    for jogo in JOGOS:
        tronco, bola = n["pct_negada"][jogo]
        linhas.append((T("c5_por_jogo", jogo=JG[jogo]["titulo"], tronco=pct(tronco), bola=pct(bola)), False))
    y = 0.80
    for linha_texto, forte in linhas:
        if forte is None:
            fig.text(px, y, linha_texto, fontsize=13, color=cor["primary_ink"], fontweight="bold", va="top")
        else:
            fig.text(px, y, linha_texto, fontsize=13.5 if forte else 12.5, color=cor["text"] if forte else cor["text_muted"],
                     fontweight="bold" if forte else "normal", va="top")
        y -= 0.05
    nota(fig, brand, T("c5_nota"), y=0.05, largura=200)  # uma linha só: com 2 sobrepunha-se ao título do eixo
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"rho": n["rho"]["rho_media"], "rho_min": n["rho"]["rho_min"], "pct_mudam": n["pct_mudam"], "linhas": n["linhas"],
            "pct_negada": n["pct_negada"], "jogadores": len(j)}


# ---------------------------------------------------------------- 6. as três versões da métrica

def figura_versoes(brand: Brand, destino: Path, nota_final: str = "", idioma: str = "pt") -> dict:
    """% da ameaça negada e % de linhas excluídas por versão e jogo (v1, v2, v3), da base de cada versão."""
    cor = brand.colors
    _pt, pct, _mil = formatadores(idioma)
    T = lambda chave, **v: texto(idioma, chave, **v)
    dados = {}
    for v in VERSOES:
        for j in JOGOS:
            linhas = ler(j, "base", v)[0]
            dados[(v, j)] = (av.pct_negada(linhas), 100 * linhas["excluida"].mean())
    fig = nova_figura(brand, T("vs_titulo"), T("vs_sub"))
    eixos = [fig.add_axes((0.12, 0.26, 0.36, 0.57)), fig.add_axes((0.60, 0.26, 0.36, 0.57))]
    tons = {JOGOS[0]: cor["primary"], JOGOS[1]: cor["secondary"]}
    for k, (ax, titulo) in enumerate(zip(eixos, T("vs_paineis"))):
        estilo_eixo(ax, brand, grelha="y")
        largura = 0.36
        for m, j in enumerate(JOGOS):
            x = np.arange(len(VERSOES)) + (m - 0.5) * (largura + 0.04)
            valores = [dados[(v, j)][k] for v in VERSOES]
            ax.bar(x, valores, width=largura, color=tons[j], edgecolor=cor["background"], linewidth=2)
            for xi, val in zip(x, valores):
                ax.text(xi, val + 1, pct(val), ha="center", fontsize=12.5, color=cor["text"], fontweight="bold")
        ax.set_xticks(range(len(VERSOES)), T("vs_x"))
        ax.tick_params(axis="x", labelsize=12.5, colors=cor["text"])
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: pct(v, 0)))
        ax.set_title(titulo, loc="left", fontsize=16, fontweight="bold", color=cor["text"], pad=10)
    for m, j in enumerate(JOGOS):
        fig.patches.append(plt.Rectangle((0.12 + m * 0.30, 0.150), 0.012, 0.022, transform=fig.transFigure, color=tons[j]))
        fig.text(0.137 + m * 0.30, 0.161, JG_TITULOS[j], fontsize=13, color=cor["text"], va="center")
    nota(fig, brand, T("vs_nota") + nota_final, y=0.05)
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {f"{v}_{j}": {"pct_negada": a, "pct_excluidas": b} for (v, j), (a, b) in dados.items()}


JG_TITULOS: dict = {}


# Em linguagem corrente: este módulo desenha os gráficos do artigo que resumem os dois jogos, sempre a partir
# das tabelas já calculadas pela avaliação (as mesmas do notebook 05). Mostra onde se nega mais ameaça, o que
# disseram as três rondas de revisão, se os passes feitos fogem às sombras, se a ordem dos jogadores aguenta
# mudanças nos parâmetros e o que muda quando o cone aponta para a bola em vez de seguir o tronco.
