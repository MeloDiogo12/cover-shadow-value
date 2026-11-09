"""Fase 9c · imagem de um momento com as três moedas (xT, linhas quebradas, risco) e o índice por fase de cada defesa.

Como `figuras_9b.figura_momento` (cones, linhas, marcação) com o painel de texto mais largo: por defesa, a ameaça negada em cada moeda e o
seu índice na fase defensiva do momento (100 = média da fase); no fim, os totais do momento com o percentil entre os momentos de blocos
alto e médio. Nas linhas com ameaça negada na moeda risco marca-se o ponto de corte (×) com o custo `c` (xT do adversário).
"""

from __future__ import annotations

from pathlib import Path

import figuras_agregados as ag
import matplotlib.pyplot as plt
import pandas as pd
import revisao as rev
from figuras_agregados import pt

from footy.viz.theme import Brand

ROTULOS = {"xt": "xT", "linhas": "linhas", "risco": "risco"}
CASAS = {"xt": 4, "linhas": 2, "risco": 4}


def figura_momento_3(J: dict, frame: int, rotulo: str, valores: dict, percentis: dict, indices: dict, cortes: pd.DataFrame, brand: Brand,
                     destino: Path, vista: tuple | None = None) -> Path:
    """`valores`: {moeda: `avaliacao_v31.negada_por_defesa_no_lance`}; `percentis`: {moeda: percentil (0–100) do momento};
    `indices`: {moeda: {nome do defesa: índice na fase}}; `cortes`: pontos de corte das linhas do momento (recetor, corte_x, corte_y,
    custo_xt, negada na moeda risco); `vista`: ((x0, x1), (y0, y1)) para aproximar o campo (por omissão, os jogadores do momento)."""
    comprimento, largura = J["campo"]
    d = rev.dados_do_frame(J["tabela"], J["defesas"], J["match"], frame, comprimento, largura)
    cor = brand.colors
    fig = plt.figure(figsize=(12, 7), dpi=100, facecolor=cor["background"])
    ax_campo, ax_texto = fig.add_axes((0.02, 0.07, 0.58, 0.86)), fig.add_axes((0.615, 0.07, 0.38, 0.86))
    rev.desenhar_lance(fig, ax_campo, ax_texto, d, J["nomes"], brand, rev.novo_campo(brand, comprimento, largura), rotulo,
                       vista or rev.vista_em_torno(rev.jogadores_da_vista(d), comprimento, largura))
    for _, c in cortes.iterrows():
        ax_campo.scatter([c["corte_x"]], [c["corte_y"]], marker="x", s=70, color=cor["primary_ink"], zorder=9, linewidths=2.2)
        ax_campo.text(c["corte_x"], c["corte_y"] + 1.1, pt(c["custo_xt"], 3), fontsize=7.5, color=cor["primary_ink"], ha="center", zorder=9)
    ax_texto.clear()
    ax_texto.axis("off")
    todos = sorted({n for m in valores.values() for n in m["shapley"]}, key=lambda n: -valores["risco"]["shapley"].get(n, 0.0))
    texto = [("Ameaça negada por defesa e índice na fase (100 = média)", "text", True)]
    for nome in todos[:6]:
        texto.append((nome, "text", False))
        texto.append(("   negada  " + " · ".join(f"{ROTULOS[m]} {pt(valores[m]['shapley'].get(nome, 0.0), CASAS[m])}" for m in ROTULOS), "text_muted", False))
        texto.append(("   índice  " + " · ".join(f"{ROTULOS[m]} {pt(indices[m].get(nome, float('nan')), 0)}" for m in ROTULOS), "text_muted", False))
    if not todos:
        texto.append(("nenhuma", "text_muted", False))
    texto += [("", "text", False), ("Momento (todas as linhas avaliadas)", "text", True)]
    for moeda, rot in (("xt", "xT"), ("linhas", "linhas quebradas"), ("risco", "risco")):
        v = valores[moeda]
        linha = (f"{rot}: valor {pt(v['perigo'], 3 if moeda != 'linhas' else 0)} · negada {pt(v['negada'], CASAS[moeda])}"
                 f" · percentil {pt(percentis[moeda], 0)} %")
        texto.append((linha, "text_muted", False))
    texto.append(("× = ponto de corte da linha (custo c, xT do adversário)", "text_muted", False))
    y = 0.98
    for linha, tom, negrito in texto:
        ax_texto.text(0, y, linha, transform=ax_texto.transAxes, va="top", fontsize=8.2, color=cor[tom], fontweight="bold" if negrito else "normal")
        y -= 0.036
    rev.rodape(fig, brand, ag.FONTE)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destino


# Em linguagem corrente: a mesma imagem de sempre (cones, linhas de passe, marcação) com um painel mais largo que mostra, para cada defesa
# com sombra no momento, quanto negou em cada uma das três moedas e quanto isso vale face à média da sua fase defensiva, e marca com um ×
# o sítio onde cada linha seria cortada.
