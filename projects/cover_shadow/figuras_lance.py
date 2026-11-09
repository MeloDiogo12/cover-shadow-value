"""Fase 6 · figuras finais dos lances de exemplo (artigo): C06 passo a passo, C06 corpo vs bola e C03.

PNG a 300 dpi, 16:9, tema da marca (`config/brand.yaml`). O cálculo é o da métrica (`revisao.dados_do_frame`, mesmas funções
que a avaliação): aqui só se desenha. Todos os passos do C06 usam o mesmo enquadramento. As etiquetas dos
jogadores e das linhas não se sobrepõem (`Etiquetas`): primeiro as das linhas, depois as dos jogadores
relevantes, por fim as dos restantes (discretas).
"""

from __future__ import annotations

import textwrap
import unicodedata
from pathlib import Path

import idioma as idi
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from metrica.linhas import pode_afetar
from metrica.parametros import PADRAO
from metrica.pipeline import preparar_defesas
from metrica.sombra import contorno_do_cone
from revisao import (
    MARCACAO_TRACO,
    dados_do_frame,
    guarda_redes_defensor,
    jogadores_da_vista,
    novo_campo,
    vista_em_torno,
)

from footy.metrics.orientacao import vetor
from footy.viz.theme import Brand

DPI = 300
FIGSIZE = (16, 9)
FONTE = "SkillCorner Open Data (MIT)"
AREA_CAMPO = (0.02, 0.09, 0.69, 0.80)  # x, y, largura, altura da figura
AREA_PAINEL = (0.73, 0.09, 0.25, 0.80)
LIMITE_ETIQUETA_P = 0.95  # abaixo disto a linha tem sombra (laranja, com etiqueta); acima, é livre (cinzento)
LIMITE_ETIQUETA_VALOR = 0.005  # só as linhas com pelo menos este valor têm etiqueta no passo 1 (em xT ou em risco)
NOTA_SOMA = "Os totais somam as linhas do momento (alternativas entre si): não são golos."  # em pt; `TEXTOS[idioma]["nota_soma"]` para os dois idiomas
PASSOS_C06 = ("Linhas de passe", "Quem marca", "Quem pode afetar cada linha", "Cones pelo tronco",
              "Índice de chegada", "Ameaça negada por linha", "Ameaça negada por defesa")  # pt; os nomes dos ficheiros saem daqui
TRONCO, BOLA = "Cone pelo tronco", "Cone virado para a bola"  # chaves dos números de `c06_corpo_vs_bola` (não mudam com o idioma)
# O ficheiro do passo 5 mantém o nome antigo (o artigo já o referencia), embora o título passe a ser «Índice de chegada» (D43).
FICHEIRO_DO_PASSO = {5: "probabilidade_de_o_passe_chegar"}


# Tudo o que se lê nas figuras dos lances, por idioma (os números formatam-se à parte, em `idioma.py`).
TEXTOS = {
    "pt": {
        "unidade_risco": "risco",
        "nota_soma": NOTA_SOMA,
        "passos": PASSOS_C06,
        "resumo": "Resumo do lance",
        "contexto": "{jogo} · {parte} · {equipa} com bola",
        "leg_ataque": "azul = {equipa} (com bola)",
        "leg_defesa": "{cor_defesa} = {equipa} (a defender)",
        "leg_gr": "GR (excluído)",
        "leg_contorno": "contorno = sem body pose",
        "passo_n": "PASSO {n} DE {total}",
        "eixo_negada": "ameaça negada ({unidade})",
        "linhas_sombra_1": "{n} linha com sombra de {avaliadas} avaliadas",
        "linhas_sombra_n": "{n} linhas com sombra de {avaliadas} avaliadas",
        "por_fora": "por fora",
        "por_dentro": "por dentro",
        "sem_sombras": "nenhuma linha tem sombra",
        "sombra_de": "a sombra de {defesa} tapa a linha {lado}, para {recetor}",
        "p1_inicio": "Do portador ({portador}, círculo laranja) saem linhas para todos os colegas. ",
        "p1_risco": ("Cada linha vale o perigo que cria e o risco de a perder: o ganho de xT se a bola chegar ao colega, mais o xT que o "
                     "adversário teria no ponto onde o passe fosse cortado. Quanto mais grossa e mais laranja, mais vale."),
        "p1_xt": "Cada linha vale o perigo do passe: o ganho de xT se a bola chegar ao colega. Quanto mais grossa e mais laranja, mais perigo.",
        "p1_n_linhas": "{n} linhas de passe",
        "p1_perigo": "perigo total: {valor} {unidade}",
        "p1_etiqueta": "Etiqueta: valor da linha ({unidade}), nas linhas com pelo menos {minimo}.",
        "p2": ("Só o defesa que acompanha um atacante não faz sombra: fica de fora. Quem pressiona o portador não está a "
               "marcar e conta para a sombra. Círculo tracejado = defesa a marcar, ligado ao atacante que acompanha."),
        "p2_marcam": "A marcar: {quem}",
        "p2_ninguem": "ninguém",
        "p2_de": "{n} de {total} defesas (sem GR)",
        "p3": ("Só pode fechar uma linha o defesa que não marca e está a menos de 11 m dela. Círculo azul = pode afetar "
               "alguma linha; o número na linha diz quantos a podem afetar. Sem dados do corpo de um deles, a linha sai da conta."),
        "p3_afetam": "{n} defesas podem afetar alguma linha",
        "p3_excluidas": "{n} de {total} linhas excluídas neste lance",
        "p4": ("Por trás do tronco de cada defesa abre-se um cone de {angulo}° e {comprimento} m, "
               "com a largura do corpo à saída. A seta laranja é a frente do tronco; o cone fica nas costas."),
        "p4_cones": "Cones: {n} defesas",
        "p5": ("Se a linha atravessa o cone de um defesa, calcula-se a hipótese de ele chegar à bola a tempo "
               "(reação {reacao} s, {velocidade} m/s; bola a {bola} m/s) ou de a bloquear com o corpo. O índice de chegada da linha é o de nenhum lá chegar. "
               "Laranja = com sombra (índice abaixo de {limite}); cinzento = sem sombra. Lê-se como índice, não como probabilidade calibrada."),
        "p5_etiqueta": "Etiqueta: índice de chegada.",
        "p6": ("Ameaça negada da linha = perigo × (1 − índice de chegada). Cinzento: perigo da linha; "
               "laranja: parte negada. Etiqueta: negada de perigo (xT)."),
        "p6_etiqueta": "{negada} de {valor}",
        "p6_perigo": "perigo das linhas com sombra: {valor} {unidade}",
        "p6_negada": "ameaça negada: {valor} {unidade}",
        "p7": ("Quando mais de um defesa tapa a mesma linha, a ameaça negada reparte-se pelo valor de Shapley: "
               "cada um recebe o que acrescenta, em média, a todas as ordens possíveis. As partes somam o total."),
        "p7_total": "total: {valor} {unidade}",
        "p8_titulo": "Quem fechou o quê",
        "p8": ("{equipa} constrói com {portador}; {frase}. "
               "Cones azuis, setas laranja = frente do tronco, tracejado = marcação, linhas laranja = linhas com sombra "
               "(etiqueta: índice de chegada, só abaixo de {limite})."),
        "p8_perigo": "perigo disponível: {valor} {unidade}",
        "p8_negada": "ameaça negada: {valor} {unidade} ({pct})",
        "corpo_titulo": "Corpo vs bola",
        "tronco": "Cone pelo tronco",
        "bola": "Cone virado para a bola",
        "corpo_negada": "Ameaça negada: {valor} {unidade}",
        "corpo_nenhuma": "nenhuma",
        "corpo_nota": ("Mesmas linhas nas duas versões. Setas laranja: frente do tronco (esq.) ou direção do portador (dir.). "
                       "Etiqueta: índice de chegada (< {limite}). Totais somam as linhas do momento, não são golos."),
        "c03_titulo": "Linha para {alvo} dentro do cone de {quem}",
        "c03_painel": "A linha dentro do cone",
        "c03_texto": ("O cone por trás do tronco de {quem} cobre a linha do portador ({portador}) "
                      "para {alvo}. O índice de chegada desce para {p}."),
        "c03_perigo": "perigo da linha: {valor} {unidade}",
        "c03_negada_linha": "ameaça negada na linha: {valor} {unidade}",
        "c03_negada_lance": "ameaça negada no lance: {valor} {unidade}",
    },
    "en": {
        "unidade_risco": "risk",
        "nota_soma": "Totals add up the lanes of the moment (alternatives to each other): they are not goals.",
        "passos": ("Passing lanes", "Who is marking", "Who can affect each lane", "Trunk-based cones",
                   "Arrival index", "Denied threat by lane", "Denied threat by defender"),
        "resumo": "Play summary",
        "contexto": "{jogo} · {parte} · {equipa} in possession",
        "leg_ataque": "blue = {equipa} (in possession)",
        "leg_defesa": "{cor_defesa} = {equipa} (defending)",
        "leg_gr": "GK (excluded)",
        "leg_contorno": "outline = no direct data",
        "passo_n": "STEP {n} OF {total}",
        "eixo_negada": "denied threat ({unidade})",
        "linhas_sombra_1": "{n} lane with a shadow out of {avaliadas} evaluated",
        "linhas_sombra_n": "{n} lanes with a shadow out of {avaliadas} evaluated",
        "por_fora": "on the outside",
        "por_dentro": "on the inside",
        "sem_sombras": "no lane has a shadow",
        "sombra_de": "{defesa}'s shadow covers the lane to {recetor}, {lado}",
        "p1_inicio": "From the ball carrier ({portador}, orange circle) lanes run to every teammate. ",
        "p1_risco": ("Each lane is worth the threat it creates and the risk of losing the ball: the xT gain if the ball reaches the teammate, "
                     "plus the xT the opponent would have at the point where the pass was cut. The thicker and more orange, the more it is worth."),
        "p1_xt": "Each lane is worth the threat of the pass: the xT gain if the ball reaches the teammate. The thicker and more orange, the more threat.",
        "p1_n_linhas": "{n} passing lanes",
        "p1_perigo": "available threat: {valor} {unidade}",
        "p1_etiqueta": "Label: lane value ({unidade}), on lanes worth at least {minimo}.",
        "p2": ("Only a defender tracking an attacker casts no shadow: that defender is left out. Whoever presses the ball carrier is not "
               "marking and counts towards the shadow. Dashed circle = defender marking, linked to the attacker being tracked."),
        "p2_marcam": "Marking: {quem}",
        "p2_ninguem": "nobody",
        "p2_de": "{n} of {total} defenders (without GK)",
        "p3": ("Only a defender who is not marking and is within 11 m of a lane can close it. Blue circle = can affect "
               "some lane; the number on the lane says how many can affect it. Without body data for one of them, the lane is left out of the count."),
        "p3_afetam": "{n} defenders can affect some lane",
        "p3_excluidas": "{n} of {total} lanes excluded in this play",
        "p4": ("Behind the trunk of each defender a {angulo}° cone opens, {comprimento} m long, "
               "as wide as the body at its start. The orange arrow is the front of the trunk; the cone sits at the back."),
        "p4_cones": "Cones: {n} defenders",
        "p5": ("If the lane crosses a defender's cone, we compute the chance of that defender reaching the ball in time "
               "(reaction {reacao} s, {velocidade} m/s; ball at {bola} m/s) or blocking it with the body. The arrival index of the lane is the chance that none of them gets there. "
               "Orange = shadow (index below {limite}); grey = no shadow. It is an index, not a calibrated probability."),
        "p5_etiqueta": "Label: arrival index.",
        "p6": ("Denied threat of the lane = threat × (1 − arrival index). Grey: threat of the lane; "
               "orange: denied part. Label: denied of threat (xT)."),
        "p6_etiqueta": "{negada} of {valor}",
        "p6_perigo": "threat on lanes with a shadow: {valor} {unidade}",
        "p6_negada": "denied threat: {valor} {unidade}",
        "p7": ("When more than one defender covers the same lane, the denied threat is split by Shapley value: "
               "each gets what they add, on average, over all possible orders. The parts add up to the total."),
        "p7_total": "total: {valor} {unidade}",
        "p8_titulo": "Who closed what",
        "p8": ("{equipa} build with {portador}; {frase}. "
               "Blue cones, orange arrows = front of the trunk, dashed = marking, orange lines = lanes with a shadow "
               "(label: arrival index, only below {limite})."),
        "p8_perigo": "available threat: {valor} {unidade}",
        "p8_negada": "denied threat: {valor} {unidade} ({pct})",
        "corpo_titulo": "Body vs ball",
        "tronco": "Trunk-based cone",
        "bola": "Ball-facing cone",
        "corpo_negada": "Denied threat: {valor} {unidade}",
        "corpo_nenhuma": "none",
        "corpo_nota": ("Same lanes in both versions. Orange arrows: front of the trunk (left) or direction of the ball carrier (right). "
                       "Label: arrival index (< {limite}). Totals add up the lanes of the moment, they are not goals."),
        "c03_titulo": "Lane to {alvo} inside {quem}'s cone",
        "c03_painel": "The lane inside the cone",
        "c03_texto": ("The cone behind {quem}'s trunk covers the ball carrier's ({portador}) lane "
                      "to {alvo}. The arrival index drops to {p}."),
        "c03_perigo": "lane threat: {valor} {unidade}",
        "c03_negada_linha": "denied threat on the lane: {valor} {unidade}",
        "c03_negada_lance": "denied threat in the play: {valor} {unidade}",
    },
}


def texto(idioma: str, chave: str, variante: str = "escura", **valores) -> str:
    """Texto de uma figura; falha com a chave em falta em vez de deixar um buraco."""
    valores.setdefault("cor_defesa", idi.cor_defesa(idioma, variante))
    return TEXTOS[idioma][chave].format(**valores)


# ---------------------------------------------------------------- etiquetas sem sobreposições

class Etiquetas:
    """Coloca texto junto de pontos do campo sem que os rótulos se sobreponham entre si, aos
    jogadores nem às etiquetas das linhas. Trabalha em pontos (1/72 polegada) sobre o ecrã final."""

    DESLOCAMENTOS = ((0, 10, "center", "bottom"), (0, -10, "center", "top"), (10, 0, "left", "center"),
                     (-10, 0, "right", "center"), (8, 8, "left", "bottom"), (-8, 8, "right", "bottom"),
                     (8, -8, "left", "top"), (-8, -8, "right", "top"), (0, 22, "center", "bottom"),
                     (0, -22, "center", "top"), (22, 0, "left", "center"), (-22, 0, "right", "center"))

    # Só com `procurar_mais_longe` (vídeo, onde o campo está mais cheio de texto): mais posições e, se nenhuma
    # estiver livre, a que bate em menos etiquetas em vez da primeira.
    DESLOCAMENTOS_LONGE = ((0, 34, "center", "bottom"), (0, -34, "center", "top"), (34, 0, "left", "center"),
                           (-34, 0, "right", "center"), (26, 26, "left", "bottom"), (-26, 26, "right", "bottom"),
                           (26, -26, "left", "top"), (-26, -26, "right", "top"))

    def __init__(self, ax, jogadores_xy: list[tuple[float, float]], procurar_mais_longe: bool = False):
        self.procurar_mais_longe = procurar_mais_longe
        self.ax, self.fig = ax, ax.figure
        self.fundo = ax.get_facecolor()
        ax.apply_aspect()
        self.ocupados: list[tuple[float, float, float, float]] = []
        self.jogadores = [self._pt(x, y) for x, y in jogadores_xy]
        for px, py in self.jogadores:  # o marcador de cada jogador (raio ~9 pt)
            self.ocupados.append((px - 9, py - 9, px + 9, py + 9))

    def _pt(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.ax.transData.transform((x, y))
        return px * 72 / self.fig.dpi, py * 72 / self.fig.dpi

    @staticmethod
    def _caixa(px, py, dx, dy, ha, va, largura, altura):
        x0 = px + dx - (0 if ha == "left" else largura if ha == "right" else largura / 2)
        y0 = py + dy - (0 if va == "bottom" else altura if va == "top" else altura / 2)
        return (x0, y0, x0 + largura, y0 + altura)

    @staticmethod
    def _sobrepoe(a, b, folga=1.5) -> bool:
        return not (a[2] + folga < b[0] or b[2] + folga < a[0] or a[3] + folga < b[1] or b[3] + folga < a[1])

    def _dentro(self, caixa) -> bool:
        x0, y0, x1, y1 = self.ax.get_window_extent().extents * 72 / self.fig.dpi
        return caixa[0] >= x0 and caixa[2] <= x1 and caixa[1] >= y0 and caixa[3] <= y1

    @staticmethod
    def _tamanho(texto: str, tamanho: float) -> tuple[float, float]:
        linhas = texto.split("\n")
        return 0.6 * tamanho * max(len(linha) for linha in linhas) + 2, 1.3 * tamanho * len(linhas)

    def ponto_da_linha(self, origem, destino, texto: str, tamanho: float, cor: str, fundo: str,
                       fracoes=(0.55, 0.65, 0.45, 0.75, 0.35, 0.85, 0.25)) -> None:
        """Etiqueta no meio de uma linha (com fundo), na primeira fração livre."""
        largura, altura = self._tamanho(texto, tamanho)
        escolhida = None
        for f in fracoes:
            x, y = origem[0] + f * (destino[0] - origem[0]), origem[1] + f * (destino[1] - origem[1])
            px, py = self._pt(x, y)
            caixa = (px - largura / 2, py - altura / 2, px + largura / 2, py + altura / 2)
            if self._dentro(caixa) and not any(self._sobrepoe(caixa, o) for o in self.ocupados):
                escolhida = (x, y, caixa)
                break
        if escolhida is None:  # sem espaço livre: a fração do meio
            x, y = origem[0] + 0.5 * (destino[0] - origem[0]), origem[1] + 0.5 * (destino[1] - origem[1])
            px, py = self._pt(x, y)
            escolhida = (x, y, (px - largura / 2, py - altura / 2, px + largura / 2, py + altura / 2))
        x, y, caixa = escolhida
        self.ocupados.append(caixa)
        self.ax.text(x, y, texto, fontsize=tamanho, color=cor, ha="center", va="center", zorder=9,
                     bbox={"facecolor": fundo, "edgecolor": "none", "pad": 1.2})

    def junto_ao_jogador(self, x: float, y: float, texto: str, tamanho: float, cor: str, peso: str = "normal") -> None:
        px, py = self._pt(x, y)
        largura, altura = self._tamanho(texto, tamanho)
        melhor, menos = None, None
        candidatos = self.DESLOCAMENTOS + (self.DESLOCAMENTOS_LONGE if self.procurar_mais_longe else ())
        for dx, dy, ha, va in candidatos:
            caixa = self._caixa(px, py, dx, dy, ha, va, largura, altura)
            proprio = (px - 9, py - 9, px + 9, py + 9)
            batidas = sum(self._sobrepoe(caixa, o) for o in self.ocupados if o != proprio)
            livre = self._dentro(caixa) and batidas == 0
            if livre:
                melhor = (dx, dy, ha, va, caixa)
                break
            if melhor is None:
                melhor = (dx, dy, ha, va, caixa)
            if self.procurar_mais_longe and self._dentro(caixa) and (menos is None or batidas < menos[0]):
                menos = (batidas, (dx, dy, ha, va, caixa))
        else:
            if menos is not None:  # sem lugar livre: o que bate em menos coisas (só com `procurar_mais_longe`)
                melhor = menos[1]
        dx, dy, ha, va, caixa = melhor
        self.ocupados.append(caixa)
        # etiqueta afastada (só com `procurar_mais_longe`): uma linha fina liga-a ao jogador
        ligacao = {"arrowstyle": "-", "color": cor, "lw": 0.9, "shrinkA": 2, "shrinkB": 8} if (
            self.procurar_mais_longe and np.hypot(dx, dy) > 24) else None
        self.ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=tamanho,
                         color=cor, fontweight=peso, zorder=9, linespacing=1.05, arrowprops=ligacao,
                         path_effects=[pe.withStroke(linewidth=3, foreground=self.fundo)])


# ---------------------------------------------------------------- dados de um lance

def sem_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn").lower()


def jogador_por_nome(nomes: dict, parte: str) -> int:
    """Identificador do jogador cujo nome contém `parte` (sem acentos nem maiúsculas); erro se não for único."""
    achados = [j for j, n in nomes.items() if sem_acentos(parte) in sem_acentos(n)]
    if len(achados) != 1:
        raise ValueError(f"'{parte}' encontra {len(achados)} jogadores: {achados}")
    return achados[0]


# Correções só de apresentação (os dados ficam como vêm da fonte).
CORRECOES_NOMES = {"Isaias": "Isaías", "Jovanovic": "Jovanović"}


# Nome da equipa como se mostra nas figuras (a fonte chama-lhe "Brisbane FC"; o artigo usa o nome completo).
EQUIPAS_PARA_MOSTRAR = {"Brisbane FC": "Brisbane Roar"}


def equipa_para_mostrar(texto: str) -> str:
    for fonte, mostrar in EQUIPAS_PARA_MOSTRAR.items():
        texto = texto.replace(fonte, mostrar)
    return texto


def leitura_do_indice(indice: float, percentil: float, idioma: str = "pt") -> dict[str, str]:
    """Textos do índice por fase, para o percentil se ler primeiro e o multiplicador da média depois: um «157» ou «293» sozinho
    parece exagerado a quem lê. Usada pela Figura 13 e pela animação do C06 (têm de dizer o mesmo); não muda números, só os formata."""
    num = idi.formatadores(idioma)[0]
    return {"percentil": num(percentil, 0), "vezes": num(indice / 100, 1), "indice": num(indice, 0)}


def apelido(nomes: dict, jogador: int) -> str:
    """Apelido para mostrar; se se repetir no jogo (ex.: dois Warland), com a inicial ("B. Warland")."""
    ap = nomes[jogador].split(". ")[-1]
    if sum(1 for n in nomes.values() if n.split(". ")[-1] == ap) > 1:
        return nomes[jogador]
    return CORRECOES_NOMES.get(ap, ap)


class Lance:
    """Tudo o que se desenha de um frame: dados da métrica, enquadramento e jogadores relevantes."""

    def __init__(self, J: dict, frame: int, rotulo: str, brand: Brand, orientacao: str = "tronco", moeda: str = "xt", idioma: str = "pt"):
        self.J, self.frame, self.rotulo, self.brand = J, frame, rotulo, brand
        self.idioma = idi.validar(idioma)
        self.pt, self.pct, self.mil = idi.formatadores(idioma)
        self.moeda, self.unidade = moeda, "xT" if moeda == "xt" else TEXTOS[idioma]["unidade_risco"]
        self.cor = brand.colors
        self.nomes = J["nomes"]
        comprimento, largura = J["campo"]
        defesas = J["defesas"] if orientacao == "tronco" else preparar_defesas(J["tabela"], J["match"], PADRAO, "bola")
        self.d = dados_do_frame(J["tabela"], defesas, J["match"], frame, comprimento, largura)
        if moeda == "risco":  # D44: o valor da linha passa a ser g + c; as sombras e as probabilidades são as mesmas
            import avaliacao_risco as ar  # importa-se aqui porque arrasta o cálculo todo da v4
            self.d["linhas"], self.d["atribuicoes"] = ar.moeda_risco_do_lance(
                self.d["linhas"], self.d["atribuicoes"], ar.custos_guardados(J["jogo"]), frame)
        self.portador = self.d["portador"]
        a = self.d["atribuicoes"]
        self.shapley = a.groupby("defesa")["shapley"].sum().sort_values(ascending=False) if len(a) else pd.Series(dtype=float)
        self.marcam = self.d["defesas"][self.d["defesas"]["a_marcar"]]
        linhas = self.d["linhas"]
        # linha com sombra = atravessa um cone e a probabilidade de o passe chegar é < LIMITE_ETIQUETA_P
        self.com_sombra = set(linhas.loc[~linhas["excluida"] & (linhas["n_sombras"] > 0)
                                         & (linhas["p_chegar"] < LIMITE_ETIQUETA_P), "recetor"])
        equipas = {J["match"]["home_team"]["id"]: equipa_para_mostrar(J["match"]["home_team"]["short_name"]),
                   J["match"]["away_team"]["id"]: equipa_para_mostrar(J["match"]["away_team"]["short_name"])}
        self.equipa_com_bola = equipas[self.d["equipa_com_bola"]]
        self.equipa_a_defender = next(n for i, n in equipas.items() if i != self.d["equipa_com_bola"])
        self.titulo_jogo = equipa_para_mostrar(J["titulo"])
        # o enquadramento inclui sempre os recetores das linhas com sombra (podem estar longe do portador)
        sombra = self.d["jogadores"][self.d["jogadores"]["jogador"].isin(self.com_sombra)]
        self.jogadores_vista = pd.concat([jogadores_da_vista(self.d), sombra]).drop_duplicates("jogador")

    def xy(self, jogador: int) -> tuple[float, float]:
        j = self.d["jogadores"]
        r = j[j["jogador"] == jogador].iloc[0]
        return float(r["x"]), float(r["y"])

    def recetores(self) -> pd.DataFrame:
        """Linhas do portador, com a posição do recetor (para desenhar)."""
        j = self.d["jogadores"].set_index("jogador")[["x", "y"]]
        return self.d["linhas"].join(j, on="recetor")

    def contexto(self) -> str:
        """Linha de contexto: jogo, parte, minuto e quem tem a bola."""
        return self.texto("contexto", jogo=self.titulo_jogo, parte=idi.rotulo_do_lance(self.rotulo, self.idioma).split(" · ", 1)[1],
                          equipa=self.equipa_com_bola)

    def texto(self, chave: str, **valores) -> str:
        return texto(self.idioma, chave, self.brand.variante, **valores)


def gr_ids(lance: Lance) -> set:
    return set(guarda_redes_defensor(lance.d)["jogador"])


# ---------------------------------------------------------------- primitivas de desenho

def preparar_figura(brand: Brand, titulo: str, subtitulo: str):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=brand.colors["background"])
    fig.text(0.02, 0.965, titulo, fontsize=24, fontweight="bold", color=brand.colors["text"], va="top")
    fig.text(0.02, 0.915, subtitulo, fontsize=13.5, color=brand.colors["text_muted"], va="top")
    fig.text(0.5, 0.015, brand.credit(FONTE), ha="center", va="bottom", color=brand.colors["text_muted"], fontsize=11)
    return fig


def legenda_equipas(fig, lance: Lance, x: float, y: float, loc: str = "lower left") -> None:
    """Legenda das cores: quem tem a bola, quem defende, guarda-redes."""
    cor = lance.cor
    def marca(forma, cor_marca, texto, vazio=False):
        return Line2D([], [], marker=forma, ls="", markersize=10, markerfacecolor="none" if vazio else cor_marca,
                      markeredgecolor=cor["text_muted"] if vazio else cor["background"], markeredgewidth=1.6 if vazio else 0.8,
                      label=texto)
    itens = [marca("o", cor["secondary"], lance.texto("leg_ataque", equipa=lance.equipa_com_bola)),
             marca("o", cor["text"], lance.texto("leg_defesa", equipa=lance.equipa_a_defender)),
             marca("s", cor["neutral"], lance.texto("leg_gr"))]
    if lance.d["defesas"]["orientacao"].isna().any():
        itens.append(marca("o", cor["text_muted"], lance.texto("leg_contorno"), vazio=True))
    fig.legend(handles=itens, loc=loc, bbox_to_anchor=(x, y), ncol=len(itens), frameon=False, fontsize=11,
               labelcolor=cor["text_muted"], handletextpad=0.3, columnspacing=1.4)


def eixo_do_campo(fig, lance: Lance, area, vista) -> plt.Axes:
    ax = fig.add_axes(area)
    campo = novo_campo(lance.brand, *lance.J["campo"])
    campo.draw(ax=ax)
    ax.set_xlim(vista[0])
    ax.set_ylim(vista[1])
    return ax


def proporcao_da_area(area) -> float:
    return (area[2] * FIGSIZE[0]) / (area[3] * FIGSIZE[1])


def vista_do_lance(lance: Lance, area=AREA_CAMPO, margem: float = 6.0, outros: tuple = ()) -> tuple:
    """Enquadramento do lance; `outros`: lances do mesmo frame cujos jogadores também têm de caber."""
    jogadores = pd.concat([lance.jogadores_vista, *(o.jogadores_vista for o in outros)])
    return vista_em_torno(jogadores, *lance.J["campo"], margem=margem, proporcao=proporcao_da_area(area))


def desenhar_cones(ax, lance: Lance, quais: set | None = None, alpha_relevante=0.22, alpha_outros=0.07,
                   setas: bool = True) -> None:
    """Cone por trás de cada defesa que não marca e tem pose; `quais` limita a estes defesas."""
    cor = lance.cor
    defesas = lance.d["defesas"]
    for _, j in defesas[~defesas["a_marcar"] & defesas["orientacao"].notna()].iterrows():
        if quais is not None and j["jogador"] not in quais:
            continue
        relevante = j["jogador"] in lance.shapley.index
        ax.add_patch(Polygon(contorno_do_cone(j["x"], j["y"], j["orientacao"], PADRAO), closed=True,
                             facecolor=cor["secondary"], alpha=alpha_relevante if relevante else alpha_outros,
                             edgecolor=cor["secondary"], lw=1.0 if relevante else 0.5, zorder=1))
        if setas:
            fx, fy = vetor(j["orientacao"])
            ax.annotate("", xy=(j["x"] + 2.4 * fx, j["y"] + 2.4 * fy), xytext=(j["x"], j["y"]), zorder=7,
                        arrowprops={"arrowstyle": "-|>", "color": cor["primary_ink"], "lw": 1.8, "mutation_scale": 14})


def desenhar_jogadores(ax, lance: Lance) -> None:
    cor, d = lance.cor, lance.d
    j, defesas = d["jogadores"], d["defesas"]
    ataque = j[j["equipa"] == d["equipa_com_bola"]]
    ax.scatter(ataque["x"], ataque["y"], s=190, color=cor["secondary"], edgecolor=cor["background"], zorder=5)
    com_pose, sem_pose = defesas[defesas["orientacao"].notna()], defesas[defesas["orientacao"].isna()]
    ax.scatter(com_pose["x"], com_pose["y"], s=190, color=cor["text"], edgecolor=cor["background"], zorder=5)
    ax.scatter(sem_pose["x"], sem_pose["y"], s=190, facecolor="none", edgecolor=cor["text_muted"], lw=1.6, zorder=5)
    gr = guarda_redes_defensor(d)
    ax.scatter(gr["x"], gr["y"], s=190, marker="s", facecolor=cor["neutral"], edgecolor=cor["text_muted"], lw=1.6,
               zorder=5)
    ox, oy = d["origem"]
    ax.scatter([ox], [oy], s=620, facecolor="none", edgecolor=cor["primary_ink"], lw=2.8, zorder=6)


def etiquetar_jogadores(ax, lance: Lance, relevantes: set, extra: dict | None = None,
                        ocupados_antes: Etiquetas | None = None, escala: float = 1.0, outros: bool = True) -> Etiquetas:
    """Nomes: os relevantes em destaque, os restantes discretos. `extra`: texto a acrescentar por jogador.
    `escala` aumenta os tamanhos (vídeo, lido num ecrã pequeno); `outros=False` deixa os restantes sem nome."""
    cor, extra = lance.cor, extra or {}
    j = lance.d["jogadores"]
    gr = gr_ids(lance)
    et = ocupados_antes or Etiquetas(ax, list(zip(j["x"], j["y"])))
    ordem = sorted(j["jogador"], key=lambda i: (i not in relevantes, i))
    for i in ordem:
        x, y = lance.xy(i)
        nome = lance.texto("leg_gr") if i in gr else apelido(lance.nomes, i)
        texto = nome + (f"\n{extra[i]}" if i in extra else "")
        if i in relevantes:
            et.junto_ao_jogador(x, y, texto, 12 * escala, cor["text"], "bold")
        elif outros:
            et.junto_ao_jogador(x, y, texto, 8.5 * escala, cor["text_muted"])
    return et


def painel(fig, lance: Lance, passo: int | None, titulo: str, texto: str, numeros: list[str], area=AREA_PAINEL) -> plt.Axes:
    """Coluna da direita: o que se vê, em frases curtas, e os números do passo."""
    cor = lance.cor
    ax = fig.add_axes(area)
    ax.axis("off")
    y = 0.99
    if passo is not None:
        ax.text(0, y, lance.texto("passo_n", n=passo, total=len(PASSOS_C06)), fontsize=11, color=cor["primary_ink"], fontweight="bold", va="top",
                transform=ax.transAxes)
        y -= 0.05
    for linha in textwrap.wrap(titulo, 26):
        ax.text(0, y, linha, fontsize=17, color=cor["text"], fontweight="bold", va="top", transform=ax.transAxes)
        y -= 0.058
    y -= 0.02
    for linha in textwrap.wrap(texto, 40):
        ax.text(0, y, linha, fontsize=12, color=cor["text_muted"], va="top", transform=ax.transAxes)
        y -= 0.04
    y -= 0.03
    for item in numeros:
        for linha in textwrap.wrap(item, 42, subsequent_indent="  ") or [""]:
            ax.text(0, y, linha, fontsize=12.5, color=cor["text"], va="top", transform=ax.transAxes)
            y -= 0.043
    return ax


def barras_por_defesa(fig, lance: Lance, area=(0.745, 0.11, 0.225, 0.30)) -> None:
    """Ameaça negada por defesa (Shapley), horizontal, um só tom, com o valor ao lado."""
    cor = lance.cor
    ax = fig.add_axes(area, facecolor=cor["background"])
    s = lance.shapley.sort_values()
    ax.barh([apelido(lance.nomes, j) for j in s.index], s.values, color=cor["primary"], height=0.55)
    for i, v in enumerate(s.values):
        ax.text(v + s.max() * 0.03, i, lance.pt(v, 4), va="center", fontsize=11, color=cor["text"])
    ax.set_xlim(0, s.max() * 1.45)
    ax.tick_params(axis="y", labelsize=12, colors=cor["text"], length=0)
    ax.tick_params(axis="x", labelsize=9, colors=cor["text_muted"])
    ax.set_xlabel(lance.texto("eixo_negada", unidade=lance.unidade), fontsize=10, color=cor["text_muted"])
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(cor["lines"])


def linhas_com_sombra(n_sombra: int, n_avaliadas: int, idioma: str = "pt") -> str:
    """«1 linha com sombra de 7 avaliadas», «2 linhas com sombra de 7 avaliadas»."""
    return texto(idioma, "linhas_sombra_1" if n_sombra == 1 else "linhas_sombra_n", n=n_sombra, avaliadas=n_avaliadas)


def lado_da_linha(lance: Lance, recetor: int) -> str:
    """«por fora» se o recetor está mais perto da linha lateral do que o portador, «por dentro» se mais perto do centro."""
    return lance.texto("por_fora" if abs(lance.xy(recetor)[1]) > abs(lance.d["origem"][1]) else "por_dentro")


def frase_das_sombras(lance: Lance) -> str:
    """Que sombras tapam que linhas, a partir das linhas com sombra do lance (e não escrito à mão para um lance só)."""
    a = lance.d["atribuicoes"]
    partes = []
    for recetor in sorted(lance.com_sombra):
        quem = a[a["recetor"] == recetor].sort_values("shapley", ascending=False)
        if quem.empty:
            continue
        partes.append((apelido(lance.nomes, int(quem.iloc[0]["defesa"])), lado_da_linha(lance, recetor), apelido(lance.nomes, recetor)))
    if not partes:
        return lance.texto("sem_sombras")
    return "; ".join(lance.texto("sombra_de", defesa=d, lado=lado, recetor=r) for d, lado, r in partes)


def etiqueta_xt(valor: float, casas: int = 3, idioma: str = "pt") -> str:
    return idi.numero(valor, casas, idioma)


# ---------------------------------------------------------------- um passo do C06

def relevantes_do_passo(lance: Lance, passo: int) -> set:
    """Quem tem o nome em destaque em cada passo."""
    base = {lance.portador}
    if passo == 1:
        return base | set(lance.recetores().query("valor_xt >= @LIMITE_ETIQUETA_VALOR")["recetor"])
    if passo == 2:
        return base | set(lance.marcam["jogador"]) | set(lance.marcam["atacante"])
    if passo == 3:
        return base | podem_afetar_alguma(lance)
    return base | set(lance.shapley.index) | lance.com_sombra | set(lance.marcam["jogador"])


def podem_afetar_alguma(lance: Lance) -> set:
    """Defesas que podem afetar pelo menos uma linha (D18)."""
    d = lance.d
    defesas = d["defesas"]
    r = lance.recetores()
    afeta = np.zeros(len(defesas), dtype=bool)
    for _, linha in r.iterrows():
        afeta |= pode_afetar(defesas, d["origem"], linha, PADRAO)
    return set(defesas.loc[afeta, "jogador"])


def desenhar_passo_c06(lance: Lance, passo: int, vista, caminho: Path) -> dict:
    """Uma figura do passo a passo (1 a 7) ou, com `passo = 8`, o resumo do lance. Devolve os números mostrados."""
    cor, d = lance.cor, lance.d
    resumo = passo == 8
    pt = lance.pt
    titulo = lance.texto("resumo") if resumo else TEXTOS[lance.idioma]["passos"][passo - 1]
    fig = preparar_figura(lance.brand, f"{lance.rotulo.split(' · ')[0]} · {titulo}", lance.contexto())
    legenda_equipas(fig, lance, 0.02, 0.045)
    ax = eixo_do_campo(fig, lance, AREA_CAMPO, vista)
    r = lance.recetores()
    origem = d["origem"]
    ox, oy = origem
    maxima_xt = max(r["valor_xt"].max(), 1e-9)
    avaliadas = r[~r["excluida"]]
    maxima_negada = max(avaliadas["ameaca_negada"].max(), 1e-9)
    rotulos_de_linha: list[tuple] = []  # (destino, texto)
    numeros: list[str] = []

    if passo in (4, 5, 8):
        desenhar_cones(ax, lance, setas=passo in (4, 8) or True,
                       alpha_relevante=0.26 if passo in (4, 8) else 0.14,
                       alpha_outros=0.09 if passo in (4, 8) else 0.0)

    for _, linha in r.iterrows():
        destino = (linha["x"], linha["y"])
        excluida = linha["excluida"]
        estilo = {"zorder": 3, "solid_capstyle": "round"}
        if passo == 1:
            largura = 1.2 + 5.5 * linha["valor_xt"] / maxima_xt
            ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["primary_ink"] if linha["valor_xt"] >= LIMITE_ETIQUETA_VALOR else cor["text_muted"],
                    lw=largura, alpha=0.95 if linha["valor_xt"] >= LIMITE_ETIQUETA_VALOR else 0.5, **estilo)
            if linha["valor_xt"] >= LIMITE_ETIQUETA_VALOR:
                rotulos_de_linha.append((destino, etiqueta_xt(linha["valor_xt"], idioma=lance.idioma)))
        elif passo in (2, 3, 4):
            ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["text_muted"], lw=1.2, alpha=0.65,
                    ls=(0, (4, 3)) if excluida else "-", **estilo)
            if passo == 3 and linha["n_podem_afetar"] > 0:
                rotulos_de_linha.append((destino, f"{int(linha['n_podem_afetar'])}"))
        elif passo == 5:
            if linha["recetor"] not in lance.com_sombra:
                ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["text_muted"], lw=1.0, alpha=0.5, **estilo)
            else:
                ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["primary_ink"], lw=4.2, **estilo)
                if linha["p_chegar"] < LIMITE_ETIQUETA_P:
                    rotulos_de_linha.append((destino, pt(linha["p_chegar"], 2)))
        elif passo == 6:
            ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["neutral"], lw=1.2 + 5.5 * linha["valor_xt"] / maxima_xt,
                    alpha=0.9, **estilo)
            if linha["recetor"] in lance.com_sombra:
                ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["primary_ink"],
                        lw=1.2 + 5.5 * linha["ameaca_negada"] / maxima_xt, **estilo | {"zorder": 4})
                rotulos_de_linha.append((destino, lance.texto("p6_etiqueta", negada=pt(linha["ameaca_negada"], 4), valor=pt(linha["valor_xt"], 4))))
        elif passo == 7:
            sombra = linha["recetor"] in lance.com_sombra
            ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["primary_ink"] if sombra else cor["text_muted"],
                    lw=2.2 if sombra else 0.9, alpha=0.9 if sombra else 0.4, **estilo)
        elif resumo:
            sombra = linha["recetor"] in lance.com_sombra
            if sombra:
                ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["primary_ink"],
                        lw=1.5 + 5 * linha["ameaca_negada"] / maxima_negada, **estilo)
                if linha["p_chegar"] < LIMITE_ETIQUETA_P:
                    rotulos_de_linha.append((destino, pt(linha["p_chegar"], 2)))
            else:
                ax.plot([ox, destino[0]], [oy, destino[1]], color=cor["text_muted"], lw=0.9, alpha=0.45,
                        ls=(0, (4, 3)) if excluida else "-", **estilo)

    # marcação
    if passo in (2, 3, 4, 5, 6, 7, 8):
        for _, j in lance.marcam.iterrows():
            ax.plot([j["x"], j["atacante_x"]], [j["y"], j["atacante_y"]], color=cor["save"], ls=MARCACAO_TRACO, lw=2.0, zorder=4)
            ax.scatter([j["x"]], [j["y"]], s=520, facecolor="none", edgecolor=cor["save"], ls=MARCACAO_TRACO, lw=2.4, zorder=6)
    if passo == 3:
        afetam = podem_afetar_alguma(lance)
        defesas = d["defesas"]
        sel = defesas[defesas["jogador"].isin(afetam)]
        ax.scatter(sel["x"], sel["y"], s=520, facecolor="none", edgecolor=cor["secondary_ink"], lw=2.6, zorder=6)
    desenhar_jogadores(ax, lance)
    if passo == 7:
        for j, v in lance.shapley.items():
            x, y = lance.xy(j)
            ax.scatter([x], [y], s=300 + 9000 * v / lance.shapley.max() * 0.1, facecolor="none", edgecolor=cor["primary_ink"], lw=2.4,
                       zorder=6)

    et = Etiquetas(ax, list(zip(d["jogadores"]["x"], d["jogadores"]["y"])))
    for destino, texto in rotulos_de_linha:
        et.ponto_da_linha(origem, destino, texto, 10.5, cor["text"], cor["background"])
    extra = {}
    if passo == 7:
        extra = {j: pt(v, 4) for j, v in lance.shapley.items()}
    rel = relevantes_do_passo(lance, 7 if passo == 8 else passo)
    etiquetar_jogadores(ax, lance, rel, extra, et)

    numeros = painel_do_passo(fig, lance, passo, avaliadas)
    fig.savefig(caminho, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return numeros


def painel_do_passo(fig, lance: Lance, passo: int, avaliadas: pd.DataFrame) -> dict:
    """Texto e números do painel de cada passo; devolve os números (para a legenda)."""
    d, nomes = lance.d, lance.nomes
    pt, T, passos = lance.pt, lance.texto, TEXTOS[lance.idioma]["passos"]
    nota_soma = T("nota_soma")
    linhas = d["linhas"]
    portador = apelido(nomes, lance.portador)
    n_linhas, n_aval = len(linhas), len(avaliadas)
    perigo_total, negada_total = avaliadas["valor_xt"].sum(), avaliadas["ameaca_negada"].sum()
    nomes_marcam = ", ".join(apelido(nomes, j) for j in lance.marcam["jogador"]) or T("p2_ninguem")
    nums: dict = {"linhas": n_linhas, "avaliadas": n_aval, "perigo_total_xt": perigo_total, "negada_total_xt": negada_total}
    if passo == 1:
        painel(fig, lance, 1, passos[0],
               T("p1_inicio", portador=portador) + T("p1_risco" if lance.moeda == "risco" else "p1_xt"),
               [T("p1_n_linhas", n=n_linhas), T("p1_perigo", valor=pt(perigo_total, 3), unidade=lance.unidade),
                T("p1_etiqueta", unidade=lance.unidade, minimo=pt(LIMITE_ETIQUETA_VALOR, 3)), nota_soma])
    elif passo == 2:
        painel(fig, lance, 2, passos[1], T("p2"),
               [T("p2_marcam", quem=nomes_marcam), T("p2_de", n=len(lance.marcam), total=len(d["defesas"]))])
        nums["marcam"] = [apelido(nomes, j) for j in lance.marcam["jogador"]]
    elif passo == 3:
        afetam = podem_afetar_alguma(lance)
        painel(fig, lance, 3, passos[2], T("p3"),
               [T("p3_afetam", n=len(afetam)), T("p3_excluidas", n=int(linhas["excluida"].sum()), total=n_linhas)])
        nums["podem_afetar"] = len(afetam)
        nums["excluidas"] = int(linhas["excluida"].sum())
    elif passo == 4:
        painel(fig, lance, 4, passos[3],
               T("p4", angulo=f"{PADRAO.angulo_cone:.0f}", comprimento=f"{PADRAO.comprimento_cone:.0f}"),
               [T("p4_cones", n=sum(1 for _ in lance.d["defesas"].query("~a_marcar and orientacao == orientacao").itertuples()))])
    elif passo == 5:
        painel(fig, lance, 5, passos[4],
               T("p5", reacao=pt(PADRAO.tempo_reacao), velocidade=pt(PADRAO.velocidade_defesa, 0), bola=pt(PADRAO.velocidade_bola, 0),
                 limite=pt(LIMITE_ETIQUETA_P, 2)),
               [linhas_com_sombra(len(lance.com_sombra), n_aval, lance.idioma), T("p5_etiqueta")])
        nums["p_com_sombra"] = sorted(float(v) for v in avaliadas.loc[avaliadas["recetor"].isin(lance.com_sombra), "p_chegar"])
        nums["cruzam_cone"] = int((avaliadas["n_sombras"] > 0).sum())
    elif passo == 6:
        painel(fig, lance, 6, passos[5], T("p6"),
               [T("p6_perigo", valor=pt(avaliadas.loc[avaliadas["recetor"].isin(lance.com_sombra), "valor_xt"].sum(), 3), unidade=lance.unidade),
                T("p6_negada", valor=pt(negada_total, 4), unidade=lance.unidade), nota_soma])
    elif passo == 7:
        painel(fig, lance, 7, passos[6], T("p7"),
               [T("p7_total", valor=pt(negada_total, 4), unidade=lance.unidade), nota_soma])
        barras_por_defesa(fig, lance)
    else:
        painel(fig, lance, None, T("p8_titulo"),
               T("p8", equipa=lance.equipa_com_bola, portador=portador, frase=frase_das_sombras(lance), limite=pt(LIMITE_ETIQUETA_P, 2)),
               [T("p8_perigo", valor=pt(perigo_total, 3), unidade=lance.unidade),
                T("p8_negada", valor=pt(negada_total, 4), unidade=lance.unidade, pct=lance.pct(100 * negada_total / perigo_total, 1)), nota_soma])
        barras_por_defesa(fig, lance)
    nums["shapley"] = {apelido(nomes, j): float(v) for j, v in lance.shapley.items()}
    return nums


def c06_passo_a_passo(J: dict, frame: int, rotulo: str, brand: Brand, pasta: Path, moeda: str = "xt", passos: tuple | None = None,
                      idioma: str = "pt") -> dict:
    """Sete figuras de passo e o resumo, todas com o mesmo enquadramento (ou só os `passos` pedidos). Devolve os números por figura."""
    lance = Lance(J, frame, rotulo, brand, moeda=moeda, idioma=idioma)
    vista = vista_do_lance(lance)
    numeros = {}
    for passo in range(1, 9) if passos is None else passos:
        nome = f"c06_passo{passo}_{'resumo' if passo == 8 else FICHEIRO_DO_PASSO.get(passo, PASSOS_C06[passo - 1].lower().replace(' ', '_'))}"
        nome = unicodedata.normalize("NFD", nome).encode("ascii", "ignore").decode()
        numeros[passo] = desenhar_passo_c06(lance, passo, vista, pasta / f"{nome}.png")
        numeros[passo]["ficheiro"] = f"{nome}.png"
    return numeros


# ---------------------------------------------------------------- C06: corpo vs bola

def c06_corpo_vs_bola(J: dict, frame: int, rotulo: str, brand: Brand, pasta: Path, moeda: str = "xt", idioma: str = "pt") -> dict:
    """Dois painéis, mesmo enquadramento: cone pelo tronco (a métrica) e cone virado para a bola."""
    area_esq, area_dir = (0.02, 0.19, 0.47, 0.63), (0.51, 0.19, 0.47, 0.63)
    versoes = [(TRONCO, Lance(J, frame, rotulo, brand, moeda=moeda, idioma=idioma), area_esq),
               (BOLA, Lance(J, frame, rotulo, brand, "bola", moeda, idioma), area_dir)]
    pt, T = versoes[0][1].pt, versoes[0][1].texto
    vista = vista_do_lance(versoes[0][1], area_esq, margem=6.5, outros=(versoes[1][1],))
    cor = brand.colors
    fig = preparar_figura(brand, f"{rotulo.split(' · ')[0]} · {T('corpo_titulo')}", versoes[0][1].contexto())
    numeros = {}
    legenda_equipas(fig, versoes[0][1], 0.98, 0.915, "lower right")
    for titulo, lance, area in versoes:
        ax = eixo_do_campo(fig, lance, area, vista)
        d = lance.d
        ox, oy = d["origem"]
        desenhar_cones(ax, lance, alpha_relevante=0.26, alpha_outros=0.06)
        maxima = max(d["linhas"]["valor_xt"].max(), 1e-9)
        for _, linha in lance.recetores().iterrows():
            sombra = linha["recetor"] in lance.com_sombra
            ax.plot([ox, linha["x"]], [oy, linha["y"]], color=cor["primary_ink"] if sombra else cor["text_muted"],
                    lw=1.5 + 4 * linha["ameaca_negada"] / maxima if sombra else 0.8, alpha=0.95 if sombra else 0.4, zorder=3)
        for _, j in lance.marcam.iterrows():
            ax.plot([j["x"], j["atacante_x"]], [j["y"], j["atacante_y"]], color=cor["save"], ls=MARCACAO_TRACO, lw=1.8, zorder=4)
            ax.scatter([j["x"]], [j["y"]], s=420, facecolor="none", edgecolor=cor["save"], ls=MARCACAO_TRACO, lw=2.2, zorder=6)
        desenhar_jogadores(ax, lance)
        # procurar_mais_longe: com dois jogadores a menos de ~2 m (O'Shea e Bugarija) os nomes não podem ficar um em cima do outro
        et = Etiquetas(ax, list(zip(d["jogadores"]["x"], d["jogadores"]["y"])), procurar_mais_longe=True)
        for _, linha in lance.recetores().iterrows():
            if linha["recetor"] in lance.com_sombra and linha["p_chegar"] < LIMITE_ETIQUETA_P:
                # painel da direita: a etiqueta a meio da linha ficava por cima do nome do Waddingham; prefere-se a
                # parte da linha mais afastada do portador
                fracoes = (0.7, 0.8, 0.6, 0.45, 0.85, 0.35, 0.25) if titulo == BOLA else (0.55, 0.65, 0.45, 0.75, 0.35, 0.85, 0.25)
                et.ponto_da_linha((ox, oy), (linha["x"], linha["y"]), pt(linha["p_chegar"], 2), 9.5, cor["text"],
                                  cor["background"], fracoes)
        extra = {j: pt(v, 4) for j, v in lance.shapley.items()}
        etiquetar_jogadores(ax, lance, {lance.portador} | set(lance.shapley.index) | set(lance.marcam["jogador"]),
                            extra, et)
        total = float(d["linhas"]["ameaca_negada"].sum())
        ax.set_title(lance.texto("tronco" if titulo == TRONCO else "bola"), loc="left", fontsize=17, fontweight="bold", color=cor["text"], pad=8)
        partes = " · ".join(f"{apelido(lance.nomes, j)} {pt(v, 4)}" for j, v in lance.shapley.items())
        fig.text(area[0], 0.145, T("corpo_negada", valor=pt(total, 4), unidade=lance.unidade), fontsize=15, fontweight="bold", color=cor["primary_ink"])
        fig.text(area[0], 0.115, "\n".join(textwrap.wrap(partes or T("corpo_nenhuma"), 78)), fontsize=11.5, color=cor["text"], va="top")
        numeros[titulo] = {"total_xt": total, "por_defesa": {apelido(lance.nomes, j): float(v) for j, v in lance.shapley.items()}}
    fig.text(0.02, 0.043, T("corpo_nota", limite=pt(LIMITE_ETIQUETA_P, 2)), fontsize=10, color=cor["text_muted"])
    destino = pasta / "c06_corpo_vs_bola.png"
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    numeros["ficheiro"] = destino.name
    return numeros


# ---------------------------------------------------------------- C03

def c03_lance(J: dict, frame: int, rotulo: str, brand: Brand, pasta: Path, atacante_bloqueador: str,
              recetor_em_destaque: str, moeda: str = "xt", idioma: str = "pt") -> dict:
    """O lance C03: o defesa `atacante_bloqueador` (por nome) e a linha para `recetor_em_destaque` em destaque."""
    lance = Lance(J, frame, rotulo, brand, moeda=moeda, idioma=idioma)
    pt, T = lance.pt, lance.texto
    cor, d = lance.cor, lance.d
    quem = jogador_por_nome(lance.nomes, atacante_bloqueador)
    alvo = jogador_por_nome(lance.nomes, recetor_em_destaque)
    vista = vista_do_lance(lance)
    fig = preparar_figura(brand, f"{rotulo.split(' · ')[0]} · " + T("c03_titulo", alvo=apelido(lance.nomes, alvo), quem=apelido(lance.nomes, quem)),
                          lance.contexto())
    legenda_equipas(fig, lance, 0.02, 0.045)
    ax = eixo_do_campo(fig, lance, AREA_CAMPO, vista)
    ox, oy = d["origem"]
    desenhar_cones(ax, lance, {quem}, alpha_relevante=0.30, setas=True)
    desenhar_cones(ax, lance, set(lance.d["defesas"]["jogador"]) - {quem}, alpha_relevante=0.10, alpha_outros=0.05, setas=False)
    r = lance.recetores()
    linha_alvo = r[r["recetor"] == alvo].iloc[0]
    for _, linha in r.iterrows():
        if linha["recetor"] != alvo:
            ax.plot([ox, linha["x"]], [oy, linha["y"]], color=cor["text_muted"], lw=1.0, alpha=0.5, zorder=3)
    ax.plot([ox, linha_alvo["x"]], [oy, linha_alvo["y"]], color=cor["primary_ink"], lw=5.0, zorder=4, solid_capstyle="round")
    for _, j in lance.marcam.iterrows():
        ax.plot([j["x"], j["atacante_x"]], [j["y"], j["atacante_y"]], color=cor["save"], ls=MARCACAO_TRACO, lw=1.8, zorder=4)
        ax.scatter([j["x"]], [j["y"]], s=520, facecolor="none", edgecolor=cor["save"], ls=MARCACAO_TRACO, lw=2.4, zorder=6)
    desenhar_jogadores(ax, lance)
    x_q, y_q = lance.xy(quem)
    ax.scatter([x_q], [y_q], s=720, facecolor="none", edgecolor=cor["primary_ink"], lw=2.6, ls="--", zorder=6)
    et = Etiquetas(ax, list(zip(d["jogadores"]["x"], d["jogadores"]["y"])))
    p_alvo = pt(float(linha_alvo["p_chegar"]), 2) if not linha_alvo["excluida"] else "–"
    et.ponto_da_linha((ox, oy), (linha_alvo["x"], linha_alvo["y"]), p_alvo, 10.5, cor["text"], cor["background"],
                       fracoes=(0.78, 0.84, 0.72, 0.9))
    etiquetar_jogadores(ax, lance, {lance.portador, quem, alvo} | set(lance.shapley.index) | set(lance.marcam["jogador"]), {}, et)

    a = d["atribuicoes"]
    partes = a[a["recetor"] == alvo].set_index("defesa")["shapley"]
    texto_painel = T("c03_texto", quem=apelido(lance.nomes, quem), portador=apelido(lance.nomes, lance.portador),
                     alvo=apelido(lance.nomes, alvo), p=p_alvo)
    numeros_txt = [T("c03_perigo", valor=pt(float(linha_alvo["valor_xt"]), 4), unidade=lance.unidade),
                   T("c03_negada_linha", valor=pt(float(linha_alvo["ameaca_negada"]), 4), unidade=lance.unidade)]
    numeros_txt += [f"  {apelido(lance.nomes, j)}: {pt(v, 4)} {lance.unidade}" for j, v in partes.items()]
    numeros_txt += ["", T("c03_negada_lance", valor=pt(float(d["linhas"]["ameaca_negada"].sum()), 4), unidade=lance.unidade), T("nota_soma")]
    painel(fig, lance, None, T("c03_painel"), texto_painel, numeros_txt)
    destino = pasta / "c03_lance.png"
    fig.savefig(destino, dpi=DPI, facecolor=cor["background"])
    plt.close(fig)
    return {"ficheiro": destino.name, "p_chegar": float(linha_alvo["p_chegar"]), "valor_xt": float(linha_alvo["valor_xt"]),
            "negada_linha": float(linha_alvo["ameaca_negada"]), "partes": {apelido(lance.nomes, j): float(v) for j, v in partes.items()},
            "negada_lance": float(d["linhas"]["ameaca_negada"].sum())}


# Em linguagem corrente: este módulo desenha o lance do artigo como um treinador o quereria ler, um passo de
# cada vez. O primeiro mostra as linhas de passe; os seguintes acrescentam quem marca, quem pode fechar cada
# linha, os cones, a probabilidade de o passe chegar, a ameaça negada por linha e por defesa. Todos têm o
# mesmo enquadramento para se poder alternar entre eles. Os nomes dos jogadores que contam aparecem em
# destaque e os outros discretos, e uma pequena rotina mantém as etiquetas afastadas umas das outras.
