"""Fase 7a · animação de um lance (protótipo do C06, 16:9, PT) que explica a Cover Shadow Value.

Mesmos números das figuras do artigo (Figuras 2 a 4 e 9): o cálculo é o da métrica (`figuras_lance.Lance`),
aqui só se anima. Três coisas vivem em sítios únicos, para mudar sem mexer no desenho:
- `DURACOES` / `ORDEM`: quanto dura cada passo e por que ordem;
- `TEXTOS`: tudo o que se lê no ecrã, por idioma (o EN vem depois);
- `Configuracao`: a vista (zoom) de cada passo e a forma de desenhar os jogadores (`ESTILOS_JOGADOR`).

Regras de ritmo do projeto: o laranja nunca aparece com transparência (acende de uma vez ou engrossa o
traço); os textos saem por mistura de cor com o fundo nos últimos 0,3 s de cada passo.
"""

from __future__ import annotations

import dataclasses
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import idioma as idi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figuras_lance import (
    FIGSIZE,
    FONTE,
    Etiquetas,
    Lance,
    apelido,
    etiquetar_jogadores,
    gr_ids,
    jogador_por_nome,
    leitura_do_indice,
    proporcao_da_area,
    relevantes_do_passo,
    vista_do_lance,
)
from matplotlib import animation
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon, Rectangle
from metrica.parametros import PADRAO
from metrica.sombra import contorno_do_cone
from revisao import MARCACAO_TRACO, novo_campo, vista_em_torno

from footy.ingest import skillcorner as sc
from footy.metrics.orientacao import vetor
from footy.video.compose import save_animation
from footy.viz.goal import mistura
from footy.viz.theme import Brand

# ---------------------------------------------------------------- o que se pode mudar

FPS = 25
DPI = 120  # 16 x 9 polegadas a 120 dpi = 1920 x 1080
JANELA_JOGO_S = 6.0  # segundos de jogo, antes do frame do lance, que passam no primeiro passo (em tempo real)
SAIDA_S = 0.3  # os textos saem por mistura de cor com o fundo nos últimos 0,3 s de cada passo
ENTRADA_S = 0.4  # transições curtas (linhas a passar a cinzento, setas a encolher...)
CAMARA_S = 1.5  # a câmara aproxima-se do campo inteiro para o enquadramento do lance nos primeiros 1,5 s do congelamento
MARGEM_CAMPO = 2.0  # m à volta do relvado na vista do campo inteiro
LARGURA_MAX_LEGENDA = 0.92  # fração da largura do vídeo que uma legenda de baixo pode ocupar
MARGEM_EXPLICACAO = 8.0  # m à volta dos jogadores que entram na explicação

ORDEM = ("abertura", "contexto", "pausa", "linhas", "marcacao", "corpo", "cones", "probabilidade", "valor",
         "tronco_vs_bola", "cartao")
PAUSA_FINAL_S = 1.5  # o valor final fica parado no ecrã mais este tempo, depois de a animação do passo acabar (10f)
PAUSAS = {"valor": PAUSA_FINAL_S, "tronco_vs_bola": PAUSA_FINAL_S}  # os passos com pausa no fim
DURACOES = {  # segundos de cada passo (a pausa final já conta)
    "abertura": 4.0, "contexto": 6.0, "pausa": 2.0, "linhas": 5.0, "marcacao": 4.0, "corpo": 6.0, "cones": 5.0,
    "probabilidade": 6.0, "valor": 6.0 + PAUSA_FINAL_S, "tronco_vs_bola": 6.0 + PAUSA_FINAL_S, "cartao": 4.0,
}
VELOCIDADE_CONTEXTO = JANELA_JOGO_S / DURACOES["contexto"]  # 1 = tempo real; < 1 seria câmara lenta

AREA_CAMPO = (0.02, 0.215, 0.68, 0.665)  # x, y, largura, altura da figura
AREA_PAINEL = (0.725, 0.215, 0.255, 0.665)
AREA_INSET = (0.735, 0.30, 0.235, 0.42)
ESCALA_TEXTO = 1.4  # nomes e etiquetas no campo: o vídeo vê-se num ecrã pequeno
ESCALA_TRACO = 1.3
ESCALA_MARCADOR = 1.7
LIMITE_XT_ETIQUETA = 0.005  # só as linhas com pelo menos este perigo têm etiqueta (como na Figura do passo 1)
COMPRIMENTO_SETA = 3.0  # m
COMPRIMENTO_SETA_REALCE = 6.0  # m, a do Jelacic no passo da orientação
APELIDO_REALCE = "Jelacic"
MOEDA = "risco"  # as linhas, a ameaça negada e o Shapley do vídeo estão em risco (D44); a versão em xT está em `animacoes/v31_xt/`
# cones na variante escura (10f): o azul sobre o relvado escuro perdia-se num ecrã de telemóvel. Mais opacidade e contorno mais grosso,
# mas o cone fica por baixo das linhas (zorder 3) e dos jogadores, por isso continua a não tapar nada. Índice: defesa relevante?
OPACIDADE_CONE = {True: 0.55, False: 0.26}  # só o preenchimento; o contorno é opaco
LARGURA_CONTORNO_CONE = {True: 2.6, False: 1.6}
FRACOES_ETIQUETA = (0.55, 0.65, 0.45, 0.75, 0.35, 0.85, 0.25, 0.9, 0.18, 0.5)  # onde tentar pôr a etiqueta de cada linha

TEXTOS = {
    "pt": {
        "titulo": "{jogo} · {parte} · {minuto}'",
        "subtitulo": "{equipa} com a bola",
        "parte_1": "1.ª parte",
        "parte_2": "2.ª parte",
        "legenda_abertura": "",
        "legenda_contexto": "A defender: {equipa_defesa} ({cor_defesa})",
        "legenda_pausa": "{portador} tem a bola. Para quem passa?",
        "legenda_linhas": "Cada linha vale o perigo que cria e o risco de a perder",
        "legenda_marcacao": "Quem marca um adversário não conta para a sombra",
        "legenda_corpo": "A sombra nasce atrás do tronco",
        "legenda_cones": "Um cone de {angulo}° e {comprimento} m atrás de quem não marca",
        "legenda_probabilidade": "Índice de chegada: quanto a sombra fecha a linha",
        "legenda_valor": "Ameaça negada: {pct}% do perigo do lance",
        "legenda_tronco_vs_bola": "E se todos olhassem para a bola? O corpo muda o valor",
        "legenda_cartao": "",
        "leg_ataque": "azul = {equipa} (com bola)",
        "leg_defesa": "{cor_defesa} = {equipa} (a defender)",
        "leg_gr": "GR (excluído)",
        "leg_contorno": "contorno = sem dado direto",
        "painel_valor": "Ameaça negada (risco)",
        "painel_perigo": "{pct}% do perigo",
        "painel_chegada_titulo": "Índice de chegada",
        "painel_chegada_para": "para {recetor}",
        "painel_chegada_nota1": "perto de 0 = linha fechada · perto de 1 = linha aberta",
        "painel_chegada_nota2": "é um índice, não uma probabilidade calibrada",
        "painel_perigo_total": "perigo do lance: {perigo} (risco)",
        "painel_indice": "{nome}: à frente de {percentil}%",
        "painel_indice_nota": "dos jogadores em {fase} · {vezes}× a média (índice {indice})",
        "painel_tronco": "Cone pelo tronco",
        "painel_bola": "Cone virado para a bola",
        "inset_titulo": "{nome}, visto de cima",
        "inset_ombros": "ombros",
        "inset_ancas": "ancas",
        "inset_tronco": "frente do tronco",
        "abertura_pergunta": "Quanto vale fechar uma linha de passe sem tocar na bola?",
        "abertura_defesa": "A defender: {equipa_defesa} ({cor_defesa})",
        "cartao_titulo": "Cover Shadow Value",
        "cartao_subtitulo": "A ameaça que a defesa nega sem tocar na bola",
        "cartao_credito": "Dados: {fonte} · {handle}",
    },
    "en": {
        "titulo": "{jogo} · {parte} · {minuto}'",
        "subtitulo": "{equipa} in possession",
        "parte_1": "first half",
        "parte_2": "second half",
        "legenda_abertura": "",
        "legenda_contexto": "Defending: {equipa_defesa} ({cor_defesa})",
        "legenda_pausa": "{portador} is the ball carrier. Who gets the pass?",
        "legenda_linhas": "Each lane is worth the threat it creates and the risk of losing the ball",
        "legenda_marcacao": "A defender marking an opponent adds no cover shadow",
        "legenda_corpo": "The cover shadow starts behind the trunk",
        "legenda_cones": "Trunk-based cone: {angulo}° and {comprimento} m behind defenders not marking",
        "legenda_probabilidade": "Arrival index: how much the shadow closes the lane",
        "legenda_valor": "Denied threat: {pct}% of the play's threat",
        "legenda_tronco_vs_bola": "What if everyone faced the ball? The body changes the value",
        "legenda_cartao": "",
        "leg_ataque": "blue = {equipa} (in possession)",
        "leg_defesa": "{cor_defesa} = {equipa} (defending)",
        "leg_gr": "GK (excluded)",
        "leg_contorno": "outline = no direct data",
        "painel_valor": "Denied threat (risk)",
        "painel_perigo": "{pct}% of the threat",
        "painel_chegada_titulo": "Arrival index",
        "painel_chegada_para": "to {recetor}",
        "painel_chegada_nota1": "near 0 = lane closed · near 1 = lane open",
        "painel_chegada_nota2": "it is an index, not a calibrated probability",
        "painel_perigo_total": "threat of the play: {perigo} (risk)",
        "painel_indice": "{nome}: ahead of {percentil}%",
        "painel_indice_nota": "of players in {fase} · {vezes}× the average (phase index {indice})",
        "painel_tronco": "Trunk-based cone",
        "painel_bola": "Ball-facing cone",
        "inset_titulo": "{nome}, seen from above",
        "inset_ombros": "shoulders",
        "inset_ancas": "hips",
        "inset_tronco": "front of the trunk",
        "abertura_pergunta": "How much is it worth to close a passing lane without touching the ball?",
        "abertura_defesa": "Defending: {equipa_defesa} ({cor_defesa})",
        "cartao_titulo": "Cover Shadow Value",
        "cartao_subtitulo": "The threat the defence denies without touching the ball",
        "cartao_credito": "Data: {fonte} · {handle}",
    },
}


def texto(idioma: str, chave: str, variante: str = "escura", **valores) -> str:
    """Texto do ecrã; falha com a chave em falta em vez de mostrar um buraco."""
    valores.setdefault("cor_defesa", idi.cor_defesa(idioma, variante))
    return TEXTOS[idioma][chave].format(**valores)


# ---------------------------------------------------------------- tempo

def cronograma(duracoes: dict = DURACOES, ordem: tuple = ORDEM) -> list[tuple[str, float, float]]:
    """[(passo, início, fim)] em segundos."""
    t, fora = 0.0, []
    for nome in ordem:
        fora.append((nome, t, t + duracoes[nome]))
        t += duracoes[nome]
    return fora


def estado_em(t: float, plano: list[tuple[str, float, float]]) -> tuple[int, str, float, float]:
    """(índice do passo, nome, segundos desde o início do passo, progresso 0–1) no instante t.

    O progresso acaba onde começa a pausa final do passo (`PAUSAS`): a animação não abranda, o valor só fica mais tempo parado."""
    for i, (nome, ini, fim) in enumerate(plano):
        if t < fim or i == len(plano) - 1:
            tp = float(np.clip(t - ini, 0, fim - ini))
            return i, nome, tp, min(tp / (fim - ini - PAUSAS.get(nome, 0.0)), 1.0)
    raise ValueError("cronograma vazio")


def suave(x: float) -> float:
    """Entra e sai devagar: 0 a 1 em S."""
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def janela(p: float, ini: float, fim: float) -> float:
    """Progresso 0–1 de `p` entre `ini` e `fim` (parado fora do intervalo)."""
    return float(np.clip((p - ini) / (fim - ini), 0, 1))


# ---------------------------------------------------------------- tracking a 10 fps -> 25 fps

@dataclass
class Percurso:
    """Posições (tracking bruto, 10 fps) de um grupo de jogadores e da bola numa janela de frames."""
    frames: np.ndarray  # (n,) números de frame de tracking
    jogadores: list[int]
    x: np.ndarray  # (n, m)
    y: np.ndarray
    detetado: np.ndarray  # (n, m) bool; False = posição estimada (extrapolada), sem dado direto da câmara
    bola: np.ndarray  # (n, 2)


def ler_percurso(ficheiro: Path, frame_inicio: int, frame_fim: int, jogadores: list[int]) -> Percurso:
    """Lê do tracking bruto os frames [frame_inicio, frame_fim] (um frame por linha: a linha i é o frame i)."""
    frames, x, y, det, bola = [], [], [], [], []
    posicao = {j: k for k, j in enumerate(jogadores)}
    with open(ficheiro, encoding="utf-8") as f:
        for i, linha in enumerate(f):
            if i < frame_inicio:
                continue
            if i > frame_fim:
                break
            dados = json.loads(linha)
            assert dados["frame"] == i, f"linha {i} traz o frame {dados['frame']}"
            fx, fy, fd = (np.full(len(jogadores), np.nan) for _ in range(3))
            for p in dados["player_data"]:
                k = posicao.get(p["player_id"])
                if k is not None and p["x"] is not None:
                    fx[k], fy[k], fd[k] = p["x"], p["y"], bool(p["is_detected"])
            b = dados["ball_data"]
            frames.append(i)
            x.append(fx)
            y.append(fy)
            det.append(fd == 1)
            bola.append((np.nan if b["x"] is None else b["x"], np.nan if b["y"] is None else b["y"]))
    return Percurso(np.array(frames), list(jogadores), np.array(x), np.array(y), np.array(det), np.array(bola))


def posicoes_em(percurso: Percurso, frame: float) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Posições interpoladas linearmente entre os dois frames de tracking vizinhos de `frame` (fracionário).
    Um jogador só conta como detetado se o for nos dois frames: entre dois dados diretos a posição é estimada."""
    pos = float(np.clip(frame - percurso.frames[0], 0, len(percurso.frames) - 1))
    i0 = int(np.floor(pos))
    i1 = min(i0 + 1, len(percurso.frames) - 1)
    w = pos - i0

    def lin(a):
        return (1 - w) * a[i0] + w * a[i1]

    det = percurso.detetado[i0] & (percurso.detetado[i1] if w > 0 else True)
    tabela = pd.DataFrame({"jogador": percurso.jogadores, "x": lin(percurso.x), "y": lin(percurso.y), "detetado": det})
    bola = lin(percurso.bola)
    return tabela.dropna(subset=["x", "y"]).reset_index(drop=True), (float(bola[0]), float(bola[1]))


# ---------------------------------------------------------------- vistas (zoom) e estilos de jogador

@dataclass(frozen=True)
class Vista:
    xlim: tuple[float, float]
    ylim: tuple[float, float]


def vista_tatica(lance: Lance, outros: tuple = (), area=AREA_CAMPO) -> Vista:
    """O enquadramento das figuras do artigo."""
    xlim, ylim = vista_do_lance(lance, area, outros=outros)
    return Vista(tuple(xlim), tuple(ylim))


def jogadores_da_explicacao(lances: tuple) -> set:
    """Quem entra na explicação, em qualquer das versões do cone: portador, recetores das linhas com sombra,
    defesas com valor e os que marcam."""
    quais: set = set()
    for lance in lances:
        quais |= {lance.portador} | set(lance.com_sombra) | set(lance.shapley.index) | set(lance.marcam["jogador"])
    return quais


def vista_da_explicacao(lances: tuple, area=AREA_CAMPO, margem: float = MARGEM_EXPLICACAO) -> Vista:
    """Enquadra só os jogadores da explicação, com `margem` e a proporção da área do campo."""
    quais = jogadores_da_explicacao(lances)
    j = lances[0].d["jogadores"]
    xlim, ylim = vista_em_torno(j[j["jogador"].isin(quais)], *lances[0].J["campo"], margem=margem, proporcao=proporcao_da_area(area))
    return Vista(tuple(xlim), tuple(ylim))


def vista_campo(comprimento: float, largura: float, area=AREA_CAMPO, margem: float = MARGEM_CAMPO) -> Vista:
    """O relvado inteiro (com `margem`), centrado, na proporção da área."""
    meia_altura = max(largura / 2 + margem, (comprimento / 2 + margem) / proporcao_da_area(area))
    meia_largura = meia_altura * proporcao_da_area(area)
    return Vista((-meia_largura, meia_largura), (-meia_altura, meia_altura))


def vista_zoom(centro: tuple[float, float], raio_m: float, area=AREA_CAMPO) -> Vista:
    """Vista aproximada: `raio_m` para cada lado na vertical (a horizontal segue a proporção da área)."""
    prop = proporcao_da_area(area)
    return Vista((centro[0] - raio_m * prop, centro[0] + raio_m * prop), (centro[1] - raio_m, centro[1] + raio_m))


def interpolar_vista(a: Vista, b: Vista, p: float) -> Vista:
    """Câmara a mover-se de `a` para `b` (p de 0 a 1)."""
    p = suave(p)
    lim = lambda u, v: (u[0] + p * (v[0] - u[0]), u[1] + p * (v[1] - u[1]))
    return Vista(lim(a.xlim, b.xlim), lim(a.ylim, b.ylim))


def desenhar_pontos(ax, jogadores: pd.DataFrame, cor: dict, escala: float = ESCALA_MARCADOR) -> None:
    """Jogadores como pontos (as colunas: x, y, papel = ataque | defesa | gr, contorno = sem dado direto)."""
    s = 190 * escala
    for papel, cor_papel in (("ataque", cor["secondary"]), ("defesa", cor["text"])):
        sel = jogadores[(jogadores["papel"] == papel)]
        cheios, vazios = sel[~sel["contorno"]], sel[sel["contorno"]]
        ax.scatter(cheios["x"], cheios["y"], s=s, color=cor_papel, edgecolor=cor["background"], zorder=5)
        ax.scatter(vazios["x"], vazios["y"], s=s, facecolor="none", edgecolor=cor["text_muted"], lw=1.8, zorder=5)
    gr = jogadores[jogadores["papel"] == "gr"]
    ax.scatter(gr["x"], gr["y"], s=s, marker="s", facecolor=cor["neutral"], edgecolor=cor["text_muted"], lw=1.8, zorder=5)


ESTILOS_JOGADOR: dict[str, Callable] = {"ponto": desenhar_pontos}  # o "esqueleto" entra aqui na versão 2


@dataclass
class Configuracao:
    idioma: str = "pt"
    estilo_jogadores: str = "ponto"
    # vista de cada passo: (quadro, passo, segundos no passo, progresso) -> Vista. Por omissão: campo inteiro no
    # contexto e, a seguir ao congelamento, o enquadramento das figuras do artigo.
    vista: Callable | None = None
    mostrar_bola: bool = True
    incluir: tuple = ORDEM  # passos a incluir (o 9.º é opcional no protótipo)
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- dados do lance, por passo

def papeis_do_lance(lance: Lance) -> dict[int, str]:
    """ataque | defesa | gr (só o guarda-redes que defende é excluído, como nas figuras)."""
    j = lance.d["jogadores"]
    gr = gr_ids(lance)
    return {int(r.jogador): ("gr" if r.jogador in gr else "ataque" if r.equipa == lance.d["equipa_com_bola"] else "defesa")
            for r in j.itertuples()}


def jogadores_congelados(lance: Lance, papeis: dict[int, str]) -> pd.DataFrame:
    j = lance.d["jogadores"]
    return pd.DataFrame({"jogador": j["jogador"], "x": j["x"], "y": j["y"],
                         "papel": j["jogador"].map(papeis), "contorno": ~j["detetado"].astype(bool)}).reset_index(drop=True)


def numeros_do_lance(lance: Lance) -> dict:
    """Os números que o vídeo mostra, tirados do lance (para os confrontar com as figuras do artigo)."""
    avaliadas = lance.recetores()
    avaliadas = avaliadas[~avaliadas["excluida"]]
    perigo, negada = float(avaliadas["valor_xt"].sum()), float(avaliadas["ameaca_negada"].sum())
    com_sombra = avaliadas[avaliadas["recetor"].isin(lance.com_sombra)]
    return {"perigo_total_xt": perigo, "negada_total_xt": negada, "pct_negada": 100 * negada / perigo,
            "shapley": {apelido(lance.nomes, j): float(v) for j, v in lance.shapley.items()},
            "p_com_sombra": sorted(float(v) for v in com_sombra["p_chegar"]),
            "chegada_minima": (None if com_sombra.empty else
                               (float(com_sombra["p_chegar"].min()),
                                apelido(lance.nomes, int(com_sombra.loc[com_sombra["p_chegar"].idxmin(), "recetor"])))),
            "linhas_com_etiqueta_xt": sorted(float(v) for v in avaliadas["valor_xt"] if v >= LIMITE_XT_ETIQUETA)}


def cone_ate(x: float, y: float, orientacao: float, fracao: float) -> np.ndarray:
    """Cone da métrica com o comprimento reduzido a `fracao` (para o ver crescer a partir do jogador)."""
    p = dataclasses.replace(PADRAO, comprimento_cone=max(PADRAO.comprimento_cone * fracao, 0.05))
    return contorno_do_cone(x, y, orientacao, p)


def com_estagio(k: int, n: int, p: float, atraso: float = 0.45) -> float:
    """Progresso do k-ésimo de n elementos que arrancam escalonados ao longo de `atraso` do tempo."""
    inicio = atraso * k / max(n - 1, 1)
    return suave(janela(p, inicio, inicio + (1 - atraso)))


# ---------------------------------------------------------------- o quadro (figura reutilizada em cada frame)

class Quadro:
    """Uma figura criada uma vez; cada frame apaga o que é dinâmico e volta a desenhar."""

    def __init__(self, J: dict, frame: int, rotulo: str, brand: Brand, cfg: Configuracao | None = None):
        self.cfg = cfg or Configuracao()
        self.brand, self.cor, self.idioma = brand, brand.colors, self.cfg.idioma
        self.J, self.frame = J, frame
        self.L = Lance(J, frame, rotulo, brand, moeda=MOEDA, idioma=self.idioma)  # D44: a moeda oficial
        self.Lb = Lance(J, frame, rotulo, brand, "bola", MOEDA, idioma=self.idioma)
        self.papeis = papeis_do_lance(self.L)
        self.plano = cronograma(DURACOES, tuple(p for p in ORDEM if p in self.cfg.incluir))
        self.duracao = self.plano[-1][2]
        self.n_frames = round(self.duracao * FPS)
        self.numeros = {"tronco": numeros_do_lance(self.L), "bola": numeros_do_lance(self.Lb)}
        self.realce = jogador_por_nome(self.L.nomes, APELIDO_REALCE)
        self.vista_base = vista_da_explicacao((self.L, self.Lb))
        self.vista_inteira = vista_campo(*J["campo"])
        n = round(JANELA_JOGO_S * 10)
        self.percurso = ler_percurso(sc.caminho(J["jogo"], "tracking_extrapolated.jsonl"), frame - n, frame, list(self.papeis))
        self.pt = idi.formatadores(self.idioma)[0]  # «0,0109» em pt, «0.0109» em en
        self.parte = texto(self.idioma, f"parte_{self.L.d['periodo']}")
        self._montar_figura()

    # -- figura e elementos fixos
    def _montar_figura(self) -> None:
        cor, L = self.cor, self.L
        self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=cor["background"])
        minuto = L.rotulo.rsplit(" ", 1)[-1]
        self.t_titulo = self.fig.text(0.02, 0.965, texto(self.idioma, "titulo", jogo=L.titulo_jogo, parte=self.parte,
                                                         minuto=minuto), fontsize=30, fontweight="bold", color=cor["text"], va="top")
        self.t_sub = self.fig.text(0.02, 0.905, texto(self.idioma, "subtitulo", equipa=L.equipa_com_bola), fontsize=19,
                                   color=cor["text_muted"], va="top")
        self.legendas_reduzidas: dict[str, int] = {}  # chave da legenda -> tamanho a que teve de descer
        self.t_legenda = self.fig.text(0.5, 0.118, "", fontsize=34, fontweight="bold", color=cor["text"], ha="center", va="center",
                                       linespacing=1.15)
        self.t_credito = self.fig.text(0.5, 0.015, idi.marca(self.brand, self.idioma).credit(FONTE), ha="center", va="bottom", color=cor["text_muted"], fontsize=13)
        self._legenda_de_cores()
        self.ax = self.fig.add_axes(AREA_CAMPO)
        novo_campo(self.brand, *self.J["campo"]).draw(ax=self.ax)
        self.base = set(self.ax.get_children())
        self.ax_painel = self.fig.add_axes(AREA_PAINEL)
        self.ax_inset = self.fig.add_axes(AREA_INSET)
        self.t_cartao = [self.fig.text(0.5, 0.56, texto(self.idioma, "cartao_titulo"), fontsize=64, fontweight="bold",
                                       color=cor["text"], ha="center", va="center"),
                         self.fig.text(0.5, 0.44, texto(self.idioma, "cartao_subtitulo"), fontsize=28, color=cor["primary"],
                                       ha="center", va="center"),
                         self.fig.text(0.5, 0.30, texto(self.idioma, "cartao_credito", fonte=FONTE, handle=self.brand.handle),
                                       fontsize=24, color=cor["text_muted"], ha="center", va="center")]
        for t in self.t_cartao:
            t.set_visible(False)
        self.t_abertura = [self.fig.text(0.5, 0.56, textwrap.fill(texto(self.idioma, "abertura_pergunta"), 32), fontsize=50, fontweight="bold",
                                         color=cor["text"], ha="center", va="center", linespacing=1.2),
                           self.fig.text(0.5, 0.30, texto(self.idioma, "abertura_defesa", self.brand.variante, equipa_defesa=self.L.equipa_a_defender),
                                         fontsize=32, color=cor["text"], ha="center", va="center")]
        for t in self.t_abertura:
            t.set_visible(False)

    def _legenda_de_cores(self) -> None:
        cor, L, i = self.cor, self.L, self.idioma
        itens = [("o", cor["secondary"], False, texto(i, "leg_ataque", equipa=L.equipa_com_bola)),
                 ("o", cor["text"], False, texto(i, "leg_defesa", self.brand.variante, equipa=L.equipa_a_defender)),
                 ("s", cor["neutral"], False, texto(i, "leg_gr")),
                 ("o", cor["text_muted"], True, texto(i, "leg_contorno"))]
        from matplotlib.lines import Line2D
        marcas = [Line2D([], [], marker=m, ls="", markersize=13, markerfacecolor="none" if vazio else c,
                         markeredgecolor=cor["text_muted"] if vazio or m == "s" else cor["background"],
                         markeredgewidth=1.8 if vazio else 0.8, label=txt) for m, c, vazio, txt in itens]
        self.legenda_cores = self.fig.legend(handles=marcas, loc="lower left", bbox_to_anchor=(0.02, 0.17), ncol=4, frameon=False,
                                             fontsize=14, labelcolor=cor["text_muted"], handletextpad=0.3, columnspacing=1.6)

    # -- limpeza entre frames
    def _limpar(self) -> None:
        for a in list(self.ax.get_children()):
            if a not in self.base:
                a.remove()
        for ax in (self.ax_painel, self.ax_inset):
            ax.clear()
            ax.axis("off")
        self.ax_painel.set_xlim(0, 1)
        self.ax_painel.set_ylim(0, 1)

    # -- um frame
    def desenhar(self, t: float) -> None:
        _, nome, tp, p = estado_em(t, self.plano)
        self._limpar()
        carta = nome in ("abertura", "cartao")
        for objeto in (self.ax, self.t_titulo, self.t_sub, self.legenda_cores):
            objeto.set_visible(not carta)
        for tc in self.t_cartao:
            tc.set_visible(nome == "cartao")
        saida = janela(tp, DURACOES[nome] - SAIDA_S, DURACOES[nome])
        for ta, cor_texto in zip(self.t_abertura, (self.cor["text"], self.cor["text"])):
            ta.set_visible(nome == "abertura")
            ta.set_color(mistura(cor_texto, self.cor["background"], saida))  # sai por mistura de cor, nunca por transparência
        self.t_credito.set_visible(nome != "cartao")
        self._legenda_do_passo(nome, tp, DURACOES[nome])
        if carta:
            return
        v = self.cfg.vista(self, nome, tp, p) if self.cfg.vista else self._vista_por_omissao(nome, tp)
        self.ax.set_xlim(v.xlim)
        self.ax.set_ylim(v.ylim)
        ativos = {n: k for k, (n, _, _) in enumerate(self.plano)}
        passa = lambda n: n in ativos and ativos[n] <= ativos[nome]
        if nome == "contexto":
            self._contexto(tp)
        else:
            self._lance(nome, tp, p, passa)

    def _vista_por_omissao(self, nome: str, tp: float) -> Vista:
        if nome == "contexto":
            return self.vista_inteira
        if nome == "pausa":
            return interpolar_vista(self.vista_inteira, self.vista_base, janela(tp, 0, CAMARA_S))
        return self.vista_base

    # -- textos de baixo
    def _legenda_do_passo(self, nome: str, tp: float, dur: float) -> None:
        n = self.numeros["tronco"]
        chave = f"legenda_{nome}"
        s = texto(self.idioma, chave, portador=apelido(self.L.nomes, self.L.portador), pct=self.pt(n["pct_negada"], 1),
                  angulo=self.pt(PADRAO.angulo_cone, 0), comprimento=self.pt(PADRAO.comprimento_cone, 0), equipa_defesa=self.L.equipa_a_defender)
        linhas = textwrap.fill(s, 60)
        self.t_legenda.set_text(linhas)
        tamanho = 34 if len(linhas.splitlines()) == 1 else 29
        self.t_legenda.set_fontsize(tamanho)
        # proteção: uma legenda mais larga do que 92 % do vídeo (cortada nos lados) encolhe até caber
        largura_max = LARGURA_MAX_LEGENDA * self.fig.bbox.width
        while tamanho > 12 and self.t_legenda.get_window_extent(self.fig.canvas.get_renderer()).width > largura_max:
            tamanho -= 1
            self.t_legenda.set_fontsize(tamanho)
        if tamanho < (34 if len(linhas.splitlines()) == 1 else 29):
            self.legendas_reduzidas[chave] = tamanho
        self.t_legenda.set_color(mistura(self.cor["text"], self.cor["background"], janela(tp, dur - SAIDA_S, dur)))

    # -- passo 1: o jogo a correr
    def _contexto(self, tp: float) -> None:
        frame_jogo = self.percurso.frames[0] + (tp * VELOCIDADE_CONTEXTO) * 10
        pos, bola = posicoes_em(self.percurso, frame_jogo)
        pos["papel"] = pos["jogador"].map(self.papeis)
        pos["contorno"] = ~pos["detetado"]
        ESTILOS_JOGADOR[self.cfg.estilo_jogadores](self.ax, pos, self.cor)
        # a equipa que defende fica destacada: um anel laranja em cada defesa, de uma vez desde o início e a sair por mistura no fim
        defesas = pos[pos["papel"] == "defesa"]
        saida = janela(tp, DURACOES["contexto"] - SAIDA_S, DURACOES["contexto"])
        self.ax.scatter(defesas["x"], defesas["y"], s=520 * ESCALA_MARCADOR, facecolor="none", lw=2.8, zorder=6,
                        edgecolor=mistura(self.cor["primary"], self.cor["background"], saida))
        self._bola(bola)

    def _bola(self, bola: tuple[float, float]) -> None:
        if self.cfg.mostrar_bola and not np.isnan(bola[0]):
            self.ax.scatter([bola[0]], [bola[1]], s=130, color=self.cor["text"], edgecolor=self.cor["background"], lw=1.2, zorder=6)

    # -- passos 2 a 9: o frame congelado e o que se lhe acrescenta
    def _lance(self, nome: str, tp: float, p: float, passa: Callable) -> None:
        cor, ax = self.cor, self.ax
        bola_fase = nome == "tronco_vs_bola"
        # no passo 9 a primeira parte ainda mostra o lance pelo tronco; a partir de 0,25 passa para o da bola
        L = self.Lb if bola_fase and p >= 0.25 else self.L
        ox, oy = L.d["origem"]
        dur = DURACOES[nome]
        et_dados = L.d["jogadores"]
        # -- linhas de passe
        linhas = self._linhas(L, nome, tp, p, dur)
        for (x0, y0, x1, y1), estilo in linhas["tracos"]:
            ax.plot([x0, x1], [y0, y1], zorder=3, solid_capstyle="round", **estilo)
        # -- marcação
        if passa("marcacao"):
            k = 1.0 if nome != "marcacao" else suave(janela(tp, 0, 0.8))
            for _, j in L.marcam.iterrows():
                ax.plot([j["x"], j["x"] + k * (j["atacante_x"] - j["x"])], [j["y"], j["y"] + k * (j["atacante_y"] - j["y"])],
                        color=cor["save"], ls=MARCACAO_TRACO, lw=2.6 * ESCALA_TRACO, zorder=4)
                ax.scatter([j["x"]], [j["y"]], s=520 * ESCALA_MARCADOR, facecolor="none", edgecolor=cor["save"],
                           ls=MARCACAO_TRACO, lw=3.0 * k, zorder=6)
        # -- cones e setas
        self._cones_e_setas(L, nome, tp, p, passa)
        # -- jogadores
        jog = jogadores_congelados(L, self.papeis)
        ESTILOS_JOGADOR[self.cfg.estilo_jogadores](ax, jog, cor)
        self._bola((float(et_dados["bola_x"].iloc[0]), float(et_dados["bola_y"].iloc[0])))
        # -- portador em destaque
        r = 1.0 if nome != "pausa" else suave(janela(tp, 0, 0.9))
        ax.scatter([ox], [oy], s=620 * ESCALA_MARCADOR, facecolor="none", edgecolor=cor["primary"], lw=3.4 * r, zorder=6)
        # -- anéis dos defesas que contam (passo do valor)
        extra = {}
        if nome in ("valor", "tronco_vs_bola"):
            for jg, v in L.shapley.items():
                x, y = L.xy(jg)
                ax.scatter([x], [y], s=(300 + 900 * v / L.shapley.max()) * ESCALA_MARCADOR, facecolor="none",
                           edgecolor=cor["primary"], lw=3.0, zorder=6)
            if (nome == "valor" and p > 0.7) or (bola_fase and p > 0.85):
                extra = {jg: self.pt(v, 4) for jg, v in L.shapley.items()}
        # -- Jelacic em realce (passo da orientação)
        self._realce(L, nome, tp, p)
        # -- etiquetas
        et = Etiquetas(ax, list(zip(et_dados["x"], et_dados["y"])), procurar_mais_longe=True)
        if nome in ("valor", "tronco_vs_bola"):
            # os anéis dos defesas também ocupam espaço: sem isto o nome do portador cai por cima do anel mais próximo
            for jg, v in L.shapley.items():
                px, py = et._pt(*L.xy(jg))
                meio = np.sqrt((300 + 900 * v / L.shapley.max()) * ESCALA_MARCADOR) / 2
                et.ocupados.append((px - meio, py - meio, px + meio, py + meio))
        for destino, txt in linhas["rotulos"]:
            et.ponto_da_linha((ox, oy), destino, txt, 10.5 * ESCALA_TEXTO, cor["text"], cor["background"], FRACOES_ETIQUETA)
        quais = self._nomes_do_passo(L, nome)
        if quais:
            etiquetar_jogadores(ax, L, quais, extra, et, escala=ESCALA_TEXTO, outros=False)
        # -- painel
        if nome in ("valor", "tronco_vs_bola"):
            self._painel_valor(nome, tp, p)
        elif nome == "probabilidade":
            self._painel_chegada(tp, p)

    def _nomes_do_passo(self, L: Lance, nome: str) -> set:
        if nome == "pausa":
            return {L.portador}
        if nome == "linhas":
            return relevantes_do_passo(L, 1)
        if nome == "marcacao":
            return set(L.marcam["jogador"])  # só quem marca, não os atacantes marcados
        if nome == "corpo":
            return {L.portador, self.realce}
        if nome == "cones":
            return {L.portador} | set(L.marcam["jogador"])
        return relevantes_do_passo(L, 5 if nome == "probabilidade" else 7)

    # -- linhas de passe: o desenho de cada passo
    def _linhas(self, L: Lance, nome: str, tp: float, p: float, dur: float) -> dict:
        cor = self.cor
        r = L.recetores()
        ox, oy = L.d["origem"]
        maxima = max(r["valor_xt"].max(), 1e-9)
        cinza = mistura(cor["text_muted"], cor["background"], 0.35)
        tracos, rotulos = [], []
        n_linhas = len(r)
        for k, (_, linha) in enumerate(r.sort_values("valor_xt").reset_index(drop=True).iterrows()):
            destino = (linha["x"], linha["y"])
            perigosa = linha["valor_xt"] >= LIMITE_XT_ETIQUETA
            fecha_xt = (cor["primary"] if perigosa else cinza, (1.2 + 5.5 * linha["valor_xt"] / maxima) * ESCALA_TRACO)
            fecha_cinza = (cinza, 1.5 * ESCALA_TRACO)
            reveal, ls = 1.0, "-"
            if linha["excluida"]:
                ls = (0, (4, 3))
            if nome == "pausa":
                continue
            if nome == "linhas":
                reveal = com_estagio(k, n_linhas, min(p / 0.7, 1.0), 0.5)  # todas desenhadas aos 70 % do passo
                c, lw = fecha_xt
                if reveal >= 0.999 and perigosa:
                    rotulos.append((destino, self.pt(linha["valor_xt"], 4)))
            elif nome == "marcacao":
                c, lw = self._mistura_estilo(fecha_xt, fecha_cinza, janela(tp, 0, ENTRADA_S))
            elif nome in ("corpo", "cones"):
                c, lw = fecha_cinza
            else:  # probabilidade, valor, tronco_vs_bola
                sombra = linha["recetor"] in L.com_sombra
                if sombra:
                    v = linha["p_chegar"]
                    if nome == "probabilidade":
                        v = 1 - (1 - v) * suave(janela(p, 0.05, 0.6))
                    lw = (1.5 + 3.5 * v) * ESCALA_TRACO
                    c = cor["primary"]
                    if v < 0.95 + 1e-9 and linha["p_chegar"] < 0.95:
                        rotulos.append((destino, self.pt(v, 2)))
                else:
                    c, lw = fecha_cinza
            tracos.append(((ox, oy, ox + reveal * (destino[0] - ox), oy + reveal * (destino[1] - oy)),
                           {"color": c, "lw": lw, "ls": ls, "alpha": 1.0}))
        return {"tracos": tracos, "rotulos": rotulos}

    @staticmethod
    def _mistura_estilo(a: tuple, b: tuple, k: float) -> tuple:
        """Estilo (cor, largura) a passar de `a` para `b`; a cor mistura-se (nunca transparência)."""
        k = suave(k)
        return mistura(a[0], b[0], k), a[1] + k * (b[1] - a[1])

    # -- cones e setas do tronco (ou do corpo virado para a bola)
    def _cones_e_setas(self, L: Lance, nome: str, tp: float, p: float, passa: Callable) -> None:
        if not passa("corpo"):
            return
        cor, ax = self.cor, self.ax
        defesas = L.d["defesas"]
        sel = defesas[~defesas["a_marcar"] & defesas["orientacao"].notna()].reset_index(drop=True)
        n = len(sel)
        for k, j in sel.iterrows():
            # progresso da seta e do cone deste defesa
            if nome == "corpo":
                seta, cone = com_estagio(k, n, p, 0.5), 0.0
            elif nome == "cones":
                seta, cone = 1.0, com_estagio(k, n, p, 0.5)
            elif nome == "tronco_vs_bola":
                # primeiro encolhe o do tronco, depois crescem os da bola (a partir de 0,25)
                fase = janela(p, 0.25, 0.6)
                seta = cone = com_estagio(k, n, fase, 0.4) if p >= 0.25 else 1 - suave(janela(p, 0, 0.25))
            else:
                seta = cone = 1.0
            comprimento = COMPRIMENTO_SETA
            largura_seta = 2.4 * ESCALA_TRACO
            if j["jogador"] == self.realce and L is self.L:
                e = self._envelope_realce(nome, tp, p)
                comprimento = COMPRIMENTO_SETA + e * (COMPRIMENTO_SETA_REALCE - COMPRIMENTO_SETA)
                largura_seta = (2.4 + 2.2 * e) * ESCALA_TRACO
            relevante = j["jogador"] in L.shapley.index
            if cone > 0.01:
                ax.add_patch(Polygon(cone_ate(j["x"], j["y"], j["orientacao"], cone), closed=True, facecolor=to_rgba(cor["secondary"], OPACIDADE_CONE[relevante]),
                                     edgecolor=cor["secondary"], lw=LARGURA_CONTORNO_CONE[relevante],
                                     zorder=1))
            if seta * comprimento > 0.35:
                fx, fy = vetor(j["orientacao"])
                c = seta * comprimento
                ax.annotate("", xy=(j["x"] + c * fx, j["y"] + c * fy), xytext=(j["x"], j["y"]), zorder=7,
                            arrowprops={"arrowstyle": "-|>", "color": cor["primary"], "lw": largura_seta,
                                        "mutation_scale": 18 * ESCALA_TRACO})

    def _envelope_realce(self, nome: str, tp: float, p: float) -> float:
        """0–1: o Jelacic realçado durante o passo da orientação; larga-se no início dos cones."""
        if nome == "corpo":
            return suave(janela(p, 0.35, 0.6))
        if nome == "cones":
            return 1 - suave(janela(tp, 0, ENTRADA_S))
        return 0.0

    def _realce(self, L: Lance, nome: str, tp: float, p: float) -> None:
        if L is not self.L:
            return
        e = self._envelope_realce(nome, tp, p)
        if e <= 0:
            return
        x, y = L.xy(self.realce)
        self.ax.scatter([x], [y], s=900 * ESCALA_MARCADOR, facecolor="none", edgecolor=self.cor["primary"], lw=4.0 * e, zorder=6)
        if nome == "corpo":
            self._inset_esqueleto(L, p)

    def _inset_esqueleto(self, L: Lance, p: float) -> None:
        """Ombros, ancas e frente do tronco do jogador em realce, vistos de cima (um só frame da pose)."""
        cor, ax = self.cor, self.ax_inset
        linha = self.J["tabela"].query("frame == @self.frame and jogador == @self.realce").iloc[0]
        if not linha["tem_pose"]:
            return
        c = np.array([linha["midHip_x"], linha["midHip_y"]])
        pt_ = lambda n: np.array([linha[f"{n}_x"], linha[f"{n}_y"]]) - c
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect("equal")
        ax.add_patch(Rectangle((-0.6, -0.6), 1.2, 1.2, facecolor=cor["pitch"], edgecolor=cor["lines"], lw=1.5, transform=ax.transData))
        for a, b, n in (("lShoulder", "rShoulder", "inset_ombros"), ("lHip", "rHip", "inset_ancas")):
            pa, pb = pt_(a), pt_(b)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=cor["text"], lw=6, solid_capstyle="round")
            ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], s=120, color=cor["text"], zorder=3)
            m = (pa + pb) / 2
            fora = m - (pt_("lShoulder") + pt_("rShoulder") + pt_("lHip") + pt_("rHip")) / 4  # do tronco para fora
            fora = fora / max(np.hypot(*fora), 1e-9)
            lugar = m + 0.26 * fora  # afastada da linha, do lado de fora do tronco
            ax.text(lugar[0], lugar[1], texto(self.idioma, n), color=cor["text_muted"], fontsize=14, ha="center", va="center")
        espinha = np.array([pt_("neck"), pt_("midHip")])
        ax.plot(espinha[:, 0], espinha[:, 1], color=cor["text_muted"], lw=3)
        k = suave(janela(p, 0.45, 0.7))
        fx, fy = vetor(float(L.d["defesas"].set_index("jogador").loc[self.realce, "orientacao"]))
        topo = (pt_("lShoulder") + pt_("rShoulder") + pt_("lHip") + pt_("rHip")) / 4
        if k > 0.05:
            ax.annotate("", xy=(topo[0] + 0.5 * k * fx, topo[1] + 0.5 * k * fy), xytext=tuple(topo),
                        arrowprops={"arrowstyle": "-|>", "color": cor["primary"], "lw": 5, "mutation_scale": 28})
        ax.set_title(texto(self.idioma, "inset_titulo", nome=apelido(L.nomes, self.realce)), color=cor["text"], fontsize=18,
                     fontweight="bold", loc="left", pad=10)
        ax.axis("off")

    # -- painel da direita no passo do índice de chegada (o valor vem dos dados, o da linha com menor índice)
    def _painel_chegada(self, tp: float, p: float) -> None:
        minimo = self.numeros["tronco"]["chegada_minima"]
        if minimo is None:
            return
        cor, ax, i = self.cor, self.ax_painel, self.idioma
        valor, recetor = minimo
        saida = janela(tp, DURACOES["probabilidade"] - SAIDA_S, DURACOES["probabilidade"])
        fundo = cor["background"]
        ax.text(0, 0.97, texto(i, "painel_chegada_titulo"), fontsize=20, fontweight="bold", va="top",
                color=mistura(cor["text_muted"], fundo, saida))
        ax.text(0, 0.90, self.pt(valor, 2), fontsize=52, fontweight="bold", va="top", color=mistura(cor["primary"], fundo, saida))
        ax.text(0, 0.775, texto(i, "painel_chegada_para", recetor=recetor), fontsize=26, va="top",
                color=mistura(cor["text"], fundo, saida))
        ax.text(0, 0.68, texto(i, "painel_chegada_nota1").replace(" · ", "\n") + "\n" + textwrap.fill(texto(i, "painel_chegada_nota2"), 28),
                fontsize=15, va="top", color=mistura(cor["text_muted"], fundo, saida))

    # -- painel da direita: ameaça negada
    def _painel_valor(self, nome: str, tp: float, p: float) -> None:
        cor, ax, i = self.cor, self.ax_painel, self.idioma
        tr, bo = self.numeros["tronco"], self.numeros["bola"]
        if nome == "valor":
            numeros, rotulo, valor = tr, None, tr["negada_total_xt"] * suave(janela(p, 0.0, 0.55))
            barras = suave(janela(p, 0.45, 0.85))
        else:
            if p < 0.25:
                numeros, rotulo, valor, barras = tr, "painel_tronco", tr["negada_total_xt"], 1.0
            else:
                morfe = suave(janela(p, 0.3, 0.8))
                valor = tr["negada_total_xt"] + morfe * (bo["negada_total_xt"] - tr["negada_total_xt"])
                numeros, rotulo, barras = bo, "painel_bola", suave(janela(p, 0.45, 0.85))
        ax.text(0, 0.97, texto(i, rotulo) if rotulo else texto(i, "painel_valor"), fontsize=20, color=cor["text_muted"], va="top",
                fontweight="bold")
        ax.text(0, 0.90, self.pt(valor, 4), fontsize=52, fontweight="bold", color=cor["primary"], va="top")
        if nome == "valor" and p > 0.5:
            ax.text(0, 0.775, texto(i, "painel_perigo", pct=self.pt(numeros["pct_negada"], 1)), fontsize=26, color=cor["text"], va="top")
        if nome == "valor" and p > 0.5 or nome == "tronco_vs_bola":
            ax.text(0, 0.70 if nome == "valor" else 0.775, texto(i, "painel_perigo_total", perigo=self.pt(numeros["perigo_total_xt"], 3)), fontsize=16,
                    color=cor["text_muted"], va="top")
        partes = sorted(numeros["shapley"].items(), key=lambda kv: -kv[1])
        maximo = max(v for _, v in partes)
        y = 0.58
        for k, (quem, v) in enumerate(partes):
            f = suave(janela(barras, 0.0, 1.0))
            ax.text(0, y, quem, fontsize=20, color=cor["text"], va="bottom")
            ax.text(1.0, y, self.pt(v * f, 4), fontsize=20, color=cor["text"], va="bottom", ha="right")
            ax.add_patch(Rectangle((0, y - 0.035), 1.0 * v / maximo * f, 0.025, facecolor=cor["primary"], edgecolor="none"))
            y -= 0.115
        indice = self.cfg.extra.get("indice_realce")
        if nome == "valor" and indice and p > 0.75:
            leitura = leitura_do_indice(indice["indice"], indice["percentil"], self.idioma)
            ax.text(0, y + 0.02, texto(i, "painel_indice", nome=indice["nome"], percentil=leitura["percentil"]), fontsize=22,
                    fontweight="bold", color=cor["primary"], va="top")
            nota = texto(i, "painel_indice_nota", fase=indice["fase"], vezes=leitura["vezes"], indice=leitura["indice"])
            # não cabe numa linha do painel: parte no «·», sem baixar a letra
            ax.text(0, y - 0.045, nota.replace(" · ", "\n"), fontsize=15, color=cor["text_muted"], va="top")

    # -- gravação
    def animacao(self) -> animation.FuncAnimation:
        return animation.FuncAnimation(self.fig, lambda k: self.desenhar(k / FPS) or [], frames=self.n_frames, blit=False)

    def guardar_video(self, destino: Path) -> Path:
        plt.rcParams["savefig.dpi"] = "figure"
        return save_animation(self.animacao(), destino, fps=FPS)

    def guardar_imagem_do_passo(self, indice: int, destino: Path) -> Path:
        """PNG do fim do passo `indice` (0 a n-1): o estado em que o espectador o vê completo."""
        fim = self.plano[indice][2]
        self.desenhar(fim - SAIDA_S - 1 / FPS)  # antes de os textos começarem a sair
        destino.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(destino, dpi=DPI, facecolor=self.cor["background"])
        return destino


# Em linguagem corrente: este módulo transforma as figuras do artigo num filme. Primeiro mostra o jogo a
# correr (posições do tracking, 10 por segundo, esticadas para 25 por segundo em linha reta), depois pára no
# momento do lance e vai acrescentando, um passo de cada vez, o que a métrica faz: linhas de passe, quem marca,
# a frente do tronco, os cones, o índice de chegada e a ameaça negada. Nada é calculado aqui: os
# números vêm do mesmo cálculo das figuras. As durações e os textos estão no topo, para se mudarem sem tocar
# no desenho, e a vista (zoom) e a forma de desenhar os jogadores são parâmetros, para a segunda versão com
# esqueletos.
