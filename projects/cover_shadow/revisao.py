"""Fase 5 · figuras, clips e folha de revisão para a revisão por um treinador (C1 e C2).

Para cada lance escolhido gera uma figura completa (linhas de passe, quem marca, cones,
probabilidades e ameaça negada por defesa) e um clip de ~3 s centrado no momento. A folha
`revisao_lances.csv` tem uma linha por lance, com as respostas em branco para preencher.

O desenho de um frame (`desenhar_lance`) é o mesmo na figura e no clip. Os cálculos vêm de
`metrica.linhas`, os mesmos da métrica: nada é recalculado à parte.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Polygon
from metrica.linhas import avaliar_frame, recetores_com_valor
from metrica.parametros import PADRAO, Parametros
from metrica.sombra import contorno_do_cone
from metrica.xt import sentido_de_ataque
from mplsoccer import Pitch

from footy.metrics.orientacao import vetor
from footy.video.compose import save_animation
from footy.viz.theme import Brand

COLUNAS_FOLHA = ["lance", "ficheiros", "sombras_fazem_sentido", "marcacao_correta", "comentario"]
RESPOSTAS = ("sim", "em parte", "não")
JANELA_CLIP = 15  # frames de tracking (10 por segundo) para cada lado: ~3 s no total
FPS_CLIP = 5  # a tabela tem 5 frames alinhados por segundo: o clip passa em tempo real


def pt(valor: float, casas: int = 1) -> str:
    """Número com vírgula decimal, para o texto."""
    return f"{valor:.{casas}f}".replace(".", ",")


def dados_do_frame(tabela: pd.DataFrame, defesas: pd.DataFrame, match: dict, frame: int,
                   comprimento: float, largura: float, p: Parametros = PADRAO) -> dict:
    """Tudo o que o desenho de um frame precisa: jogadores, defesas, linhas e atribuições."""
    jogadores = tabela[tabela["frame"] == frame]
    defesas_frame = defesas[defesas["frame"] == frame].reset_index(drop=True)
    primeiro = jogadores.iloc[0]
    sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
    origem, recetores = recetores_com_valor(jogadores, sentido, comprimento, largura, p)
    linhas, atribuicoes = avaliar_frame(origem, recetores, defesas_frame, p)
    # v3.1 (D43): as linhas com mais de 35 m saem do lance (não se desenham nem contam como excluídas)
    linhas = linhas[linhas["motivo"] != "mais_de_35m"].reset_index(drop=True)
    return {"frame": frame, "jogadores": jogadores, "defesas": defesas_frame, "origem": origem,
            "linhas": linhas, "atribuicoes": atribuicoes, "equipa_com_bola": primeiro["equipa_com_bola"],
            "portador": primeiro["portador"], "periodo": int(primeiro["periodo"])}


def novo_campo(brand: Brand, comprimento: float, largura: float):
    return Pitch(pitch_type="skillcorner", pitch_length=comprimento, pitch_width=largura,
                 pitch_color=brand.colors["pitch"], line_color=brand.colors["lines"], linewidth=1.2)


def negada_por_defesa(d: dict) -> pd.Series:
    """Ameaça negada por cada defesa neste frame (soma das partes de Shapley das suas linhas)."""
    return d["atribuicoes"].groupby("defesa")["shapley"].sum().sort_values(ascending=False)


# Marcação: anel e linha a tracejado no tom do texto (D47). O vermelho-rosa anterior ficava perto do coral (ΔE ≈ 31; ≈ 21 em deuteranopia).
MARCACAO_TRACO = (0, (4, 2.5))

ETIQUETA_GR = "GR (excluído)"


def guarda_redes_defensor(d: dict) -> pd.DataFrame:
    """O guarda-redes da equipa que defende neste frame. Não projeta sombra nem marca (D25): desenha-se
    à parte, para ninguém o tomar por um defesa sem cone."""
    j = d["jogadores"]
    return j[(j["equipa"] != d["equipa_com_bola"]) & (j["posicao"] == "GK")]


ROTULOS_MOTIVO = {"sem_pose": "sem pose", "fora_de_jogo": "fora de jogo", "cruzamento": "cruzamento", "mais_de_35m": "mais de 35 m"}


def texto_das_excluidas(linhas: pd.DataFrame) -> str:
    """"3 excluídas (1 fora de jogo, 2 cruzamento)": as linhas que não se avaliam, por motivo."""
    excluidas = linhas[linhas["excluida"]]
    if excluidas.empty:
        return "0 excluídas"
    partes = [f"{n} {ROTULOS_MOTIVO[m]}" for m, n in excluidas["motivo"].value_counts().items()]
    return f"{len(excluidas)} excluídas ({', '.join(partes)})"


def painel_de_texto(ax, d: dict, nomes: dict, brand: Brand) -> None:
    """Coluna à direita: ameaça negada por defesa, quem marca e linhas excluídas."""
    ax.axis("off")
    negada = negada_por_defesa(d)
    marcam = d["defesas"][d["defesas"]["a_marcar"]]
    linhas = d["linhas"]
    avaliadas = linhas[~linhas["excluida"]]
    texto = [("Ameaça negada por defesa (xT)", "text", True)]
    texto += [(f"{nomes[j]}  {pt(v, 4)}", "text_muted", False) for j, v in negada.items()] or [("nenhuma", "text_muted", False)]
    texto += [("", "text", False), ("A marcar (fora da sombra)", "text", True)]
    texto += [(f"{nomes[j]}", "text_muted", False) for j in marcam["jogador"]] or [("ninguém", "text_muted", False)]
    texto += [("", "text", False), ("Linhas", "text", True),
              (f"{len(avaliadas)} avaliadas", "text_muted", False),
              (texto_das_excluidas(linhas), "text_muted", False),
              (f"perigo {pt(avaliadas['valor_xt'].sum(), 3)} · negada {pt(avaliadas['ameaca_negada'].sum(), 4)}",
               "text_muted", False)]
    y = 0.98
    for linha, cor, negrito in texto:
        ax.text(0, y, linha, transform=ax.transAxes, va="top", fontsize=9.5, color=brand.colors[cor],
                fontweight="bold" if negrito else "normal")
        y -= 0.05


def desenhar_lance(fig, ax_campo, ax_texto, d: dict, nomes: dict, brand: Brand, campo: Pitch, titulo: str,
                   vista: tuple, p: Parametros = PADRAO) -> None:
    """Desenha um frame: cones, jogadores, linhas de passe com a probabilidade de chegar, marcação."""
    ax = ax_campo
    campo.draw(ax=ax)
    ax.set_xlim(vista[0])
    ax.set_ylim(vista[1])
    cor = brand.colors
    defesas, jogadores, origem = d["defesas"], d["jogadores"], d["origem"]
    for _, j in defesas[~defesas["a_marcar"] & defesas["orientacao"].notna()].iterrows():
        ax.add_patch(Polygon(contorno_do_cone(j["x"], j["y"], j["orientacao"], p), closed=True,
                             facecolor=cor["secondary"], alpha=0.16, edgecolor=cor["secondary"], lw=0.8, zorder=1))
    maior_negada = max(d["linhas"]["ameaca_negada"].max(), 1e-9)
    for _, r in d["linhas"].iterrows():
        recetor = jogadores[jogadores["jogador"] == r["recetor"]].iloc[0]
        if r["excluida"]:
            ax.plot([origem[0], recetor["x"]], [origem[1], recetor["y"]], ls=(0, (3, 3)), color=cor["neutral"],
                    lw=1, zorder=2)
            continue
        com_sombra = r["n_sombras"] > 0
        ax.plot([origem[0], recetor["x"]], [origem[1], recetor["y"]], color=cor["primary_ink"] if com_sombra else cor["text_muted"],
                lw=1.2 + 4 * r["ameaca_negada"] / maior_negada if com_sombra else 0.9, alpha=0.95 if com_sombra else 0.55, zorder=3)
        if com_sombra:
            ax.annotate(pt(r["p_chegar"], 2), (origem[0] + 0.78 * (recetor["x"] - origem[0]),
                                               origem[1] + 0.78 * (recetor["y"] - origem[1])),
                        fontsize=8.5, color=cor["text"], ha="center", va="center", zorder=7,
                        bbox={"facecolor": cor["background"], "edgecolor": "none", "pad": 1.2})
    ataque = jogadores[jogadores["equipa"] == d["equipa_com_bola"]]
    ax.scatter(ataque["x"], ataque["y"], s=140, color=cor["secondary"], edgecolor=cor["background"], zorder=5)
    com_pose = defesas[defesas["orientacao"].notna()]
    sem_pose = defesas[defesas["orientacao"].isna()]
    ax.scatter(com_pose["x"], com_pose["y"], s=140, color=cor["text"], edgecolor=cor["background"], zorder=5)
    ax.scatter(sem_pose["x"], sem_pose["y"], s=140, facecolor="none", edgecolor=cor["text_muted"], lw=1.5, zorder=5)
    gr = guarda_redes_defensor(d)
    ax.scatter(gr["x"], gr["y"], s=140, marker="s", facecolor=cor["neutral"], edgecolor=cor["text_muted"], lw=1.5,
               zorder=5)
    for _, j in com_pose.iterrows():
        fx, fy = vetor(j["orientacao"])
        ax.annotate("", xy=(j["x"] + 2.0 * fx, j["y"] + 2.0 * fy), xytext=(j["x"], j["y"]), zorder=6,
                    arrowprops={"arrowstyle": "-|>", "color": cor["primary_ink"], "lw": 1.6})
    for _, j in defesas[defesas["a_marcar"]].iterrows():
        ax.plot([j["x"], j["atacante_x"]], [j["y"], j["atacante_y"]], color=cor["save"], ls=MARCACAO_TRACO, lw=1.8, zorder=4)
        ax.scatter([j["x"]], [j["y"]], s=380, facecolor="none", edgecolor=cor["save"], ls=MARCACAO_TRACO, lw=2.2, zorder=6)
    ax.scatter([origem[0]], [origem[1]], s=420, facecolor="none", edgecolor=cor["primary_ink"], lw=2.5, zorder=6)
    for _, j in jogadores.iterrows():
        nome = ETIQUETA_GR if j["jogador"] in set(gr["jogador"]) else nomes[j["jogador"]].split(". ")[-1]
        ax.annotate(nome, (j["x"], j["y"]), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=7, color=cor["text_muted"], zorder=8)
    ax.set_title(titulo, loc="left", color=cor["text"], fontsize=12)
    ax_texto.clear()
    painel_de_texto(ax_texto, d, nomes, brand)


def jogadores_da_vista(d: dict, raio_colegas: float = 30.0) -> pd.DataFrame:
    """Quem decide o enquadramento: o portador, todos os adversários e os colegas até `raio_colegas`
    do portador (um colega no meio-campo contrário esticaria o enquadramento e esconderia o lance)."""
    j, origem = d["jogadores"], d["origem"]
    perto = np.hypot(j["x"] - origem[0], j["y"] - origem[1]) <= raio_colegas
    return j[(j["equipa"] != d["equipa_com_bola"]) | perto]


def vista_em_torno(jogadores: pd.DataFrame, comprimento: float, largura: float, margem: float = 5.0,
                   proporcao: float = 1.4) -> tuple:
    """Enquadramento que inclui todos os jogadores dados, com margem e a proporção do campo
    desenhado (largura / altura), sem sair do relvado."""
    x0, x1 = jogadores["x"].min() - margem, jogadores["x"].max() + margem
    y0, y1 = jogadores["y"].min() - margem, jogadores["y"].max() + margem
    largura_vista = max(x1 - x0, (y1 - y0) * proporcao)
    altura_vista = largura_vista / proporcao
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    metade_x, metade_y = largura_vista / 2, altura_vista / 2
    cx = np.clip(cx, -comprimento / 2 + min(metade_x, comprimento / 2), comprimento / 2 - min(metade_x, comprimento / 2))
    cy = np.clip(cy, -largura / 2 + min(metade_y, largura / 2), largura / 2 - min(metade_y, largura / 2))
    return (cx - metade_x, cx + metade_x), (cy - metade_y, cy + metade_y)


def figura_base(brand: Brand):
    """Figura 12 x 7 polegadas (1200 x 700 px): campo à esquerda, painel de texto à direita."""
    fig = plt.figure(figsize=(12, 7), dpi=100, facecolor=brand.colors["background"])
    ax_campo = fig.add_axes((0.02, 0.07, 0.70, 0.86))
    ax_texto = fig.add_axes((0.74, 0.07, 0.25, 0.86))
    return fig, ax_campo, ax_texto


def rodape(fig, brand: Brand, fonte: str) -> None:
    fig.text(0.5, 0.012, brand.credit(fonte), ha="center", va="bottom", color=brand.colors["text_muted"], fontsize=9)


def guardar_figura(J: dict, frame: int, lance: str, titulo: str, brand: Brand, fonte: str, pasta: Path,
                   nome: str) -> Path:
    """Figura completa do lance. `J`: dados do jogo (tabela, defesas, match, campo, nomes)."""
    comprimento, largura = J["campo"]
    d = dados_do_frame(J["tabela"], J["defesas"], J["match"], frame, comprimento, largura)
    fig, ax_campo, ax_texto = figura_base(brand)
    desenhar_lance(fig, ax_campo, ax_texto, d, J["nomes"], brand, novo_campo(brand, comprimento, largura),
                   f"{lance} · {titulo}", vista_em_torno(jogadores_da_vista(d), comprimento, largura))
    rodape(fig, brand, fonte)
    destino = pasta / f"{nome}.png"
    pasta.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destino


def frames_do_clip(tabela: pd.DataFrame, frame: int, janela: int = JANELA_CLIP) -> list[int]:
    """Frames alinhados à volta do momento, enquanto a mesma equipa tem a bola."""
    centro = tabela[tabela["frame"] == frame].iloc[0]["equipa_com_bola"]
    perto = tabela[(tabela["frame"] >= frame - janela) & (tabela["frame"] <= frame + janela)]
    return sorted(perto[perto["equipa_com_bola"] == centro]["frame"].unique())


def guardar_clip(J: dict, frame: int, lance: str, titulo: str, brand: Brand, fonte: str, pasta: Path,
                 nome: str, janela: int = JANELA_CLIP) -> Path:
    """Clip centrado no momento (por omissão ~3 s; `janela` = frames de tracking para cada lado), com o mesmo
    desenho da figura e o enquadramento fixo."""
    comprimento, largura = J["campo"]
    quais = frames_do_clip(J["tabela"], frame, janela)
    dados = {f: dados_do_frame(J["tabela"], J["defesas"], J["match"], f, comprimento, largura) for f in quais}
    vista = vista_em_torno(pd.concat([jogadores_da_vista(d) for d in dados.values()]), comprimento, largura)
    campo = novo_campo(brand, comprimento, largura)
    fig, ax_campo, ax_texto = figura_base(brand)
    rodape(fig, brand, fonte)

    def atualizar(i: int):
        ax_campo.clear()
        f = quais[i]
        desenhar_lance(fig, ax_campo, ax_texto, dados[f], J["nomes"], brand, campo,
                       f"{lance} · {titulo} · {pt((f - frame) / 10, 1)} s", vista)
        return []

    anim = animation.FuncAnimation(fig, atualizar, frames=len(quais), blit=False)
    destino = save_animation(anim, pasta / f"{nome}.mp4", fps=FPS_CLIP)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------- a folha de revisão

def criar_folha(lances: pd.DataFrame, caminho: Path) -> bool:
    """Escreve a folha em branco. Devolve False sem tocar no ficheiro se ele já existir (nunca se
    perdem respostas já preenchidas). Separador `;` e UTF-8 com BOM, para abrir bem no Excel em português.
    `lances`: lance, ficheiros."""
    if caminho.exists():
        return False
    folha = lances[["lance", "ficheiros"]].assign(sombras_fazem_sentido="", marcacao_correta="", comentario="")
    folha[COLUNAS_FOLHA].to_csv(caminho, sep=";", index=False, encoding="utf-8-sig")
    return True


def ler_folha(caminho: Path) -> pd.DataFrame:
    return pd.read_csv(caminho, sep=";", encoding="utf-8-sig", dtype=str).fillna("")


def resumir_folha(folha: pd.DataFrame) -> pd.DataFrame | None:
    """Contagem das respostas (sim / em parte / não) por pergunta; None enquanto a folha estiver em branco."""
    colunas = ["sombras_fazem_sentido", "marcacao_correta"]
    respostas = folha[colunas].apply(lambda s: s.str.strip().str.lower())
    if not respostas.ne("").any(axis=None):
        return None
    contagem = pd.DataFrame({c: respostas[c].value_counts() for c in colunas}).reindex(
        list(RESPOSTAS) + [""]).fillna(0).astype(int)
    return contagem.rename(index={"": "por preencher"})


# Em linguagem corrente: este módulo desenha cada lance escolhido como um treinador o quereria ver: a
# vista de cima com os cones por trás de cada defesa, as linhas de passe (a laranja as que têm sombra,
# com a probabilidade de o passe chegar em cima), os defesas a marcar ligados ao seu homem a vermelho
# e, ao lado, a ameaça negada por cada defesa. O guarda-redes de quem defende aparece como um quadrado
# cinzento com a etiqueta "GR (excluído)": não faz sombra nem marca. O clip é o mesmo desenho a andar durante cerca de
# três segundos. A folha em CSV fica em branco para o treinador escrever "sim", "em parte" ou "não".
