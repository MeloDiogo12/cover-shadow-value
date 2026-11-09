"""Fase 8d · figuras, clips e folha da 4.ª ronda de revisão (C1 e C2), com a métrica v4.

Lances: os 8 da 3.ª ronda (W01–W08, para comparar) e os 3 momentos com a dissuasão (D) mais negativa (D01–D03), para ver se a
sombra empurrou o portador para uma opção mais perigosa. Cada lance tem uma figura com o valor de cada defesa na v4 (interceção
I, dissuasão D e total) e um clip de ~3 s com o mesmo desenho. Os valores vêm de `calcular_v4.avaliar_frame_v4`, os mesmos
da métrica: nada é recalculado à parte.

Com o fundo (D39) o `p` de cada linha já conta quem marca, quem não tem pose e o guarda-redes; os D01–D03 voltam a escolher-se
com a v4 com fundo. Cada imagem e a folha levam a legenda `LEGENDA` para os treinadores.

A seleção (`indice_dos_lances.csv`) guarda-se uma só vez e nunca se reescreve ao reexecutar o notebook, tal como a folha.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import calcular_v4 as c4
import escolha as esc
import matplotlib.pyplot as plt
import pandas as pd
import revisao as rev
from matplotlib import animation
from matplotlib.patches import Polygon
from metrica.pipeline import carregar_tabela, preparar_defesas
from metrica.sombra import contorno_do_cone
from metrica.xt import sentido_de_ataque
from mplsoccer import Pitch

from footy.ingest import skillcorner as sc
from footy.metrics.orientacao import vetor
from footy.video.compose import save_animation
from footy.viz.theme import Brand

RESPOSTAS = ("sim", "em parte", "não")
PERGUNTAS = (
    ("A ordem dos defesas por valor faz sentido?", "ordem_dos_defesas", "comentario_ordem"),
    ("A divisão entre cortar a linha (interceção) e fazer o portador mudar de ideias (dissuasão) faz sentido?",
     "divisao_intercecao_dissuasao", "comentario_divisao"),
)
LANCES_D = 3  # momentos de dissuasão mais negativa
LEGENDA = ("p = probabilidade de o passe chegar; s = probabilidade de o portador escolher esta linha, sem → com sombras; "
           "I = valor por cortar a linha; D = valor por fazer o portador mudar de escolha.")


# ---------------------------------------------------------------- escolha dos lances (guardada uma vez)

def momentos_com_dissuasao_mais_negativa(momentos: dict, excluir_posses: set, n: int = LANCES_D) -> pd.DataFrame:
    """Os `n` momentos com `D` (decomposição simétrica) mais negativo, de todos os jogos, com posses distintas entre si e
    fora de `excluir_posses` ((jogo, posse_id)). `momentos`: {jogo: tabela de momentos da v4}."""
    todos = pd.concat([m.assign(jogo=j) for j, m in momentos.items()], ignore_index=True)
    todos = todos[todos["n_defesas"] > 0].sort_values("D")
    escolhidos, vistas = [], set(excluir_posses)
    for r in todos.itertuples():
        if (r.jogo, r.posse_id) in vistas:
            continue
        vistas.add((r.jogo, r.posse_id))
        escolhidos.append(r)
        if len(escolhidos) == n:
            break
    return pd.DataFrame(escolhidos)[["jogo", "frame", "posse_id", "D", "I", "v"]]


def montar_indice(indice_v3: pd.DataFrame, momentos: dict) -> pd.DataFrame:
    """Índice da 4.ª ronda: os lances W01–W08 da 3.ª (mesmo jogo e frame) e D01–D03. Colunas: lance, grupo, jogo, frame,
    posse_id, periodo, nome. `momentos`: {jogo: momentos da v4}; os W têm de existir nos momentos da v4."""
    w = indice_v3[["lance", "grupo", "jogo", "frame", "posse_id", "periodo"]].copy()
    w["grupo"] = "ronda_3"
    d = momentos_com_dissuasao_mais_negativa(momentos, set(zip(w["jogo"], w["posse_id"])))
    d = d.assign(lance=[f"D{i:02d}" for i in range(1, len(d) + 1)], grupo="dissuasao_mais_negativa")
    d["periodo"] = [int(periodo_do_frame(j, f)) for j, f in zip(d["jogo"], d["frame"])]
    indice = pd.concat([w, d[["lance", "grupo", "jogo", "frame", "posse_id", "periodo"]]], ignore_index=True)
    indice["nome"] = indice["lance"] + "_" + indice["jogo"].astype(str) + "_" + indice["frame"].astype(str)
    return indice


def periodo_do_frame(jogo: int, frame: int) -> int:
    """Parte do jogo (1 ou 2) de um frame, da tabela da fase 3."""
    t = pd.read_parquet(caminho_da_tabela(jogo), columns=["frame", "periodo"])
    return int(t.loc[t["frame"] == frame, "periodo"].iloc[0])


def caminho_da_tabela(jogo: int) -> Path:
    from preparar_dados import caminho_tabela

    return caminho_tabela(jogo)


# ---------------------------------------------------------------- dados de um frame

def carregar_jogo(jogo: int) -> dict:
    """Tudo o que os desenhos do jogo precisam: tabela, defesas, campo, nomes e o que a v4 usa por frame."""
    match = sc.ler_match(jogo)
    tabela = carregar_tabela(jogo)
    defesas = preparar_defesas(tabela, match, c4.PADRAO)
    jogadores = c4.jogadores_do_jogo(match)
    marcados = esc.recetores_marcados(defesas)
    marcados_por_frame: dict = {}
    for frame, recetor in zip(marcados["frame"], marcados["recetor"]):
        marcados_por_frame.setdefault(frame, set()).add(recetor)
    dist = esc.distancia_ao_adversario_mais_proximo(tabela[["frame", "jogador", "x", "y", "defende"]])
    return {"jogo": jogo, "match": match, "tabela": tabela, "defesas": defesas, "campo": sc.dimensoes_campo(match),
            "nomes": jogadores.set_index("jogador")["nome"].to_dict(), "coef": c4.coeficientes_do_outro_jogo(jogo),
            "marcados_por_frame": marcados_por_frame, "guardas_por_frame": c4.guarda_redes_por_frame(tabela),
            "dist_por_frame": {f: dict(zip(g["recetor"], g["dist_adversario"])) for f, g in dist.groupby("frame")},
            "titulo": f"{match['home_team']['short_name']} – {match['away_team']['short_name']}"}


def dados_do_frame_v4(J: dict, frame: int) -> dict | None:
    """Jogadores, defesas, linhas e atribuições da v4 num frame (None se a v4 não avalia nenhuma linha)."""
    jogadores = J["tabela"][J["tabela"]["frame"] == frame]
    defesas = J["defesas"][J["defesas"]["frame"] == frame].reset_index(drop=True)
    primeiro = jogadores.iloc[0]
    sentido = sentido_de_ataque(J["match"], primeiro["equipa_com_bola"], primeiro["periodo"])
    r = c4.avaliar_frame_v4(jogadores, defesas, sentido, J["campo"], J["marcados_por_frame"].get(frame, set()),
                            J["dist_por_frame"][frame], J["coef"],
                            guarda_redes=J["guardas_por_frame"].get(frame))
    if r is None:
        return None
    linhas, momento, atribuicoes, _ = r
    pos = jogadores[jogadores["jogador"] == primeiro["portador"]].iloc[0]
    return {"frame": frame, "jogadores": jogadores, "defesas": defesas, "origem": (pos["x"], pos["y"]), "linhas": linhas,
            "momento": momento, "atribuicoes": atribuicoes, "equipa_com_bola": primeiro["equipa_com_bola"],
            "portador": primeiro["portador"]}


def valores_por_defesa(d: dict) -> pd.DataFrame:
    """Por defesa do frame: interceção (I), dissuasão (D) e total (Shapley, decomposição simétrica), do maior ao menor total."""
    a = d["atribuicoes"].set_index("defesa")
    return pd.DataFrame({"I": a["shapley_I"], "D": a["shapley_D"], "total": a["shapley"]}).sort_values("total", ascending=False)


# ---------------------------------------------------------------- desenho

def painel_v4(ax, d: dict, nomes: dict, brand: Brand) -> None:
    """Coluna à direita: valor de cada defesa (I, D, total), totais do momento e quem marca."""
    ax.axis("off")
    cor, m = brand.colors, d["momento"]
    texto = [("Valor por defesa (xT)", "text", True), ("                      I        D     total", "text_muted", False)]
    for j, v in valores_por_defesa(d).iterrows():
        texto.append((f"{nomes[j].split('. ')[-1][:12]:<12}{rev.pt(v['I'], 4):>8}{rev.pt(v['D'], 4):>9}{rev.pt(v['total'], 4):>9}",
                      "text_muted", False))
    if len(texto) == 2:
        texto.append(("nenhum defesa com sombra", "text_muted", False))
    marcam = d["defesas"][d["defesas"]["a_marcar"]]
    texto += [("", "text", False), ("Momento", "text", True),
              (f"E(∅) {rev.pt(m['E0'], 4)} · E(C) {rev.pt(m['EC'], 4)}", "text_muted", False),
              (f"I {rev.pt(m['I'], 4)} · D {rev.pt(m['D'], 4)} · v {rev.pt(m['v'], 4)}", "text_muted", False),
              (f"{m['n_linhas']} linhas ({m['n_linhas_sem_pose']} sem pose)", "text_muted", False),
              ("", "text", False), ("A marcar (fora da sombra)", "text", True)]
    texto += [(nomes[j], "text_muted", False) for j in marcam["jogador"]] or [("ninguém", "text_muted", False)]
    y = 0.98
    for linha, c, negrito in texto:
        ax.text(0, y, linha, transform=ax.transAxes, va="top", fontsize=7.6, color=cor[c], family="monospace",
                fontweight="bold" if negrito else "normal")
        y -= 0.047


def desenhar_lance_v4(ax_campo, ax_texto, d: dict, nomes: dict, brand: Brand, campo: Pitch, titulo: str, vista: tuple) -> None:
    """Um frame: cones, jogadores, linhas de passe (largura = ameaça negada da linha, com `p` de chegar e a escolha
    `s` sem e com sombras) e marcação."""
    ax, cor = ax_campo, brand.colors
    campo.draw(ax=ax)
    ax.set_xlim(vista[0])
    ax.set_ylim(vista[1])
    defesas, jogadores, origem, linhas = d["defesas"], d["jogadores"], d["origem"], d["linhas"]
    p = c4.PADRAO
    for _, j in defesas[~defesas["a_marcar"] & defesas["orientacao"].notna()].iterrows():
        ax.add_patch(Polygon(contorno_do_cone(j["x"], j["y"], j["orientacao"], p), closed=True, facecolor=cor["secondary"],
                             alpha=0.16, edgecolor=cor["secondary"], lw=0.8, zorder=1))
    avaliados = set(linhas["recetor"])
    ataque = jogadores[(jogadores["equipa"] == d["equipa_com_bola"]) & (jogadores["jogador"] != d["portador"])]
    for _, r in ataque[~ataque["jogador"].isin(avaliados)].iterrows():  # fora de jogo, cruzamento ou > 35 m
        ax.plot([origem[0], r["x"]], [origem[1], r["y"]], ls=(0, (3, 3)), color=cor["neutral"], lw=1, zorder=2)
    maior = max(linhas["negada_linha"].abs().max(), 1e-9)
    for _, r in linhas.iterrows():
        recetor = jogadores[jogadores["jogador"] == r["recetor"]].iloc[0]
        com_sombra = r["n_sombras"] > 0
        ax.plot([origem[0], recetor["x"]], [origem[1], recetor["y"]], color=cor["primary_ink"] if com_sombra else cor["text_muted"],
                lw=1.2 + 4 * abs(r["negada_linha"]) / maior if com_sombra else 0.9, alpha=0.95 if com_sombra else 0.55, zorder=3)
        if com_sombra:
            ax.annotate(f"p {rev.pt(r['pC'], 2)} · s {r['s0'] * 100:.0f}→{r['sC'] * 100:.0f}%",
                        (origem[0] + 0.75 * (recetor["x"] - origem[0]), origem[1] + 0.75 * (recetor["y"] - origem[1])),
                        fontsize=8, color=cor["text"], ha="center", va="center", zorder=7,
                        bbox={"facecolor": cor["background"], "edgecolor": "none", "pad": 1.2})
    ax.scatter(jogadores[jogadores["equipa"] == d["equipa_com_bola"]]["x"], jogadores[jogadores["equipa"] == d["equipa_com_bola"]]["y"],
               s=140, color=cor["secondary"], edgecolor=cor["background"], zorder=5)
    com_pose, sem_pose = defesas[defesas["orientacao"].notna()], defesas[defesas["orientacao"].isna()]
    ax.scatter(com_pose["x"], com_pose["y"], s=140, color=cor["text"], edgecolor=cor["background"], zorder=5)
    ax.scatter(sem_pose["x"], sem_pose["y"], s=140, facecolor="none", edgecolor=cor["text_muted"], lw=1.5, zorder=5)
    gr = rev.guarda_redes_defensor({"jogadores": jogadores, "equipa_com_bola": d["equipa_com_bola"]})
    ax.scatter(gr["x"], gr["y"], s=140, marker="s", facecolor=cor["neutral"], edgecolor=cor["text_muted"], lw=1.5, zorder=5)
    for _, j in com_pose.iterrows():
        fx, fy = vetor(j["orientacao"])
        ax.annotate("", xy=(j["x"] + 2.0 * fx, j["y"] + 2.0 * fy), xytext=(j["x"], j["y"]), zorder=6,
                    arrowprops={"arrowstyle": "-|>", "color": cor["primary_ink"], "lw": 1.6})
    for _, j in defesas[defesas["a_marcar"]].iterrows():
        ax.plot([j["x"], j["atacante_x"]], [j["y"], j["atacante_y"]], color=cor["save"], ls=rev.MARCACAO_TRACO, lw=1.8, zorder=4)
        ax.scatter([j["x"]], [j["y"]], s=380, facecolor="none", edgecolor=cor["save"], ls=rev.MARCACAO_TRACO, lw=2.2, zorder=6)
    ax.scatter([origem[0]], [origem[1]], s=420, facecolor="none", edgecolor=cor["primary_ink"], lw=2.5, zorder=6)
    for _, j in jogadores.iterrows():
        nome = rev.ETIQUETA_GR if j["jogador"] in set(gr["jogador"]) else nomes[j["jogador"]].split(". ")[-1]
        ax.annotate(nome, (j["x"], j["y"]), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=7,
                    color=cor["text_muted"], zorder=8)
    ax.set_title(titulo, loc="left", color=cor["text"], fontsize=12)
    ax_texto.clear()
    painel_v4(ax_texto, d, nomes, brand)


def _vista(J: dict, dados: list) -> tuple:
    comprimento, largura = J["campo"]
    vistas = [rev.jogadores_da_vista({"jogadores": d["jogadores"], "origem": d["origem"], "equipa_com_bola": d["equipa_com_bola"]})
              for d in dados]
    return rev.vista_em_torno(pd.concat(vistas), comprimento, largura)


def figura_base_v4(brand: Brand):
    """Como `rev.figura_base`, com espaço por baixo para a legenda dos treinadores (em duas linhas, antes do crédito)."""
    fig = plt.figure(figsize=(12, 7), dpi=100, facecolor=brand.colors["background"])
    ax_campo = fig.add_axes((0.02, 0.13, 0.70, 0.80))
    ax_texto = fig.add_axes((0.74, 0.13, 0.25, 0.80))
    meio = LEGENDA.index("I = ")
    for y, linha in ((0.075, LEGENDA[:meio].rstrip(" ;") + "."), (0.048, LEGENDA[meio:])):
        fig.text(0.5, y, linha, ha="center", va="bottom", color=brand.colors["text"], fontsize=9)
    return fig, ax_campo, ax_texto


def guardar_figura(J: dict, frame: int, lance: str, titulo: str, brand: Brand, fonte: str, pasta: Path, nome: str) -> Path:
    d = dados_do_frame_v4(J, frame)
    fig, ax_campo, ax_texto = figura_base_v4(brand)
    desenhar_lance_v4(ax_campo, ax_texto, d, J["nomes"], brand, rev.novo_campo(brand, *J["campo"]), f"{lance} · {titulo}", _vista(J, [d]))
    rev.rodape(fig, brand, fonte)
    destino = pasta / f"{nome}.png"
    pasta.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destino


def guardar_clip(J: dict, frame: int, lance: str, titulo: str, brand: Brand, fonte: str, pasta: Path, nome: str) -> Path:
    """Clip de ~3 s centrado no momento; os frames em que a v4 não avalia nenhuma linha ficam de fora."""
    dados = {f: dados_do_frame_v4(J, f) for f in rev.frames_do_clip(J["tabela"], frame)}
    dados = {f: d for f, d in dados.items() if d is not None}
    quais = sorted(dados)
    vista = _vista(J, list(dados.values()))
    campo = rev.novo_campo(brand, *J["campo"])
    fig, ax_campo, ax_texto = figura_base_v4(brand)
    rev.rodape(fig, brand, fonte)

    def atualizar(i: int):
        ax_campo.clear()
        f = quais[i]
        desenhar_lance_v4(ax_campo, ax_texto, dados[f], J["nomes"], brand, campo,
                          f"{lance} · {titulo} · {rev.pt((f - frame) / 10, 1)} s", vista)
        return []

    anim = animation.FuncAnimation(fig, atualizar, frames=len(quais), blit=False)
    destino = save_animation(anim, pasta / f"{nome}.mp4", fps=rev.FPS_CLIP)
    plt.close(fig)
    return destino


# ---------------------------------------------------------------- folha .xlsx (sem openpyxl)

def _celula(ref: str, texto: str, estilo: int) -> str:
    return f'<c r="{ref}" s="{estilo}" t="inlineStr"><is><t xml:space="preserve">{escape(texto)}</t></is></c>'


def criar_folha_xlsx(lances: pd.DataFrame, caminho: Path) -> bool:
    """Escreve a folha em branco: uma linha por lance, com os ficheiros e, por pergunta, uma resposta (lista sim / em parte /
    não) e um comentário. Devolve False sem tocar no ficheiro se ele já existir (nunca se perdem respostas preenchidas).
    `lances`: lance, ficheiros."""
    if caminho.exists():
        return False
    cabecalho = ["Lance", "Ficheiros", *(t for p in PERGUNTAS for t in (p[0], "Comentário"))]
    linhas = [f'<row r="1" ht="30" customHeight="1">{_celula("A1", "Revisão dos lances (4.ª ronda) — Cover Shadow Value, v4", 1)}</row>',
              f'<row r="2" ht="30" customHeight="1">{_celula("A2", "Para cada lance: abre a imagem e o clip, escolhe nas células amarelas e escreve um comentário curto quando a resposta for «em parte» ou «não».", 2)}</row>',
              f'<row r="3" ht="32" customHeight="1">{_celula("A3", "Legenda: " + LEGENDA, 2)}</row>',
              '<row r="4" ht="64" customHeight="1">' + "".join(_celula(f"{chr(65 + i)}4", h, 3) for i, h in enumerate(cabecalho)) + "</row>"]
    for n, r in enumerate(lances.itertuples(), start=5):
        linhas.append(f'<row r="{n}">{_celula(f"A{n}", r.lance, 0)}{_celula(f"B{n}", r.ficheiros, 0)}'
                      + "".join(f'<c r="{c}{n}" s="4"/>' for c in "CDEF") + "</row>")
    ultima = 4 + len(lances)
    validacao = ('<dataValidations count="1"><dataValidation type="list" allowBlank="1" showErrorMessage="1" '
                 f'sqref="C5:C{ultima} E5:E{ultima}"><formula1>"sim,em parte,não"</formula1></dataValidation></dataValidations>')
    folha = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             '<sheetViews><sheetView workbookViewId="0"><pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
             '<cols><col min="1" max="1" width="8" customWidth="1"/><col min="2" max="2" width="46" customWidth="1"/>'
             '<col min="3" max="3" width="26" customWidth="1"/><col min="4" max="4" width="50" customWidth="1"/>'
             '<col min="5" max="5" width="26" customWidth="1"/><col min="6" max="6" width="50" customWidth="1"/></cols>'
             f'<sheetData>{"".join(linhas)}</sheetData>'
             '<mergeCells count="3"><mergeCell ref="A1:F1"/><mergeCell ref="A2:F2"/><mergeCell ref="A3:F3"/></mergeCells>'
             f'{validacao}</worksheet>')
    estilos = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
               '<fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="14"/><name val="Calibri"/></font>'
               '<font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
               '<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
               '<fill><patternFill patternType="solid"><fgColor rgb="FFD9E1F2"/></patternFill></fill>'
               '<fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/></patternFill></fill></fills>'
               '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
               '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
               '<cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
               '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
               '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>'
               '<xf numFmtId="0" fontId="2" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf>'
               '<xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf></cellXfs>'
               '</styleSheet>')
    pacote = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                               '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                               '<Default Extension="xml" ContentType="application/xml"/>'
                               '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                               '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                               '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Revisão" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                                      '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": folha, "xl/styles.xml": estilos}
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as z:
        for nome, conteudo in pacote.items():
            z.writestr(nome, conteudo)
    return True


# Em linguagem corrente: este módulo prepara a 4.ª revisão do treinador. Escolhe os 8 lances da 3.ª ronda (para comparar)
# e os 3 instantes em que a dissuasão ficou mais negativa, desenha cada um como na v4 (o valor de cada defesa dividido em
# "cortou a linha" e "fez o portador mudar de ideias") e prepara uma folha Excel em branco com as duas perguntas.
# Nada disto é conclusão: são só as imagens e a folha para ele responder.
