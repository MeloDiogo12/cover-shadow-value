"""A baliza vista de frente, com o estilo da marca.

Duas versões com a mesma interface, para se poderem trocar uma pela outra:
  - `baliza_simples`: três linhas (rápida de ler, boa para esboços)
  - `baliza_realista`: postes com volume, relvado e rede em perspetiva

As medidas são as reais em coordenadas StatsBomb (8 x 2,67 jardas), por isso os
remates desenhados por cima caem exatamente no sítio certo.

A rede nunca usa transparência: a sua cor é interpolada a partir do fundo. Sobre
o azul-noite da marca, qualquer `alpha` por cima do fundo suja a cor; misturar
as cores dá o mesmo desvanecimento sem sujar nada. É também assim que a rede sai
de cena quando uma grelha de zonas toma o seu lugar.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from footy.viz.theme import Brand

POSTE_ESQ, POSTE_DIR, BARRA = 36.0, 44.0, 2.67  # baliza em coordenadas StatsBomb
ESPESSURA = 0.13   # espessura visual do poste (~12 cm, como na realidade)
PROFUNDIDADE = 0.55  # quanto a rede "recua" na perspetiva


def mistura(c1: str, c2: str, t: float) -> str:
    """Mistura duas cores hex (t=0 -> c1, t=1 -> c2), sem transparência."""
    a = np.array([int(c1[i:i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(c2[i:i + 2], 16) for i in (1, 3, 5)])
    t = float(np.clip(t, 0, 1))
    r, g, azul = (a + (b - a) * t).round().astype(int)
    return f"#{r:02X}{g:02X}{azul:02X}"


def _rampa(prog: float, inicio: float, fim: float) -> float:
    """Progresso suavizado dentro de um troço de `prog` (arranca e trava devagar)."""
    x = float(np.clip((prog - inicio) / (fim - inicio), 0.0, 1.0))
    return x * x * (3 - 2 * x)


def baliza_simples(ax, brand: Brand, prog: float = 1.0, rede: float = 1.0,
                   tremor: float = 0.0) -> None:
    """A baliza em três linhas: `prog` 0→1 sobe os postes e corre a barra."""
    c = brand.colors
    x0, x1 = ax.get_xlim()
    ax.plot([x0, x1], [0, 0], color=c["lines"], lw=3, solid_capstyle="butt", zorder=1)

    postes, barra = _rampa(prog, 0.0, 0.6), _rampa(prog, 0.45, 1.0)
    if postes > 0:
        for x in (POSTE_ESQ, POSTE_DIR):
            ax.plot([x, x], [0, BARRA * postes], color=c["text"], lw=7,
                    solid_capstyle="round", zorder=4)
    if barra > 0:
        largura = (POSTE_DIR - POSTE_ESQ) * barra
        ax.plot([POSTE_ESQ, POSTE_ESQ + largura], [BARRA + tremor, BARRA + tremor],
                color=c["text"], lw=7, solid_capstyle="round", zorder=4)

    fade = _rampa(prog, 0.85, 1.0) * float(np.clip(rede, 0.0, 1.0))
    if fade > 0:
        cor = mistura(c["background"], c["lines"], fade)
        for x in np.arange(POSTE_ESQ + 0.5, POSTE_DIR, 0.5):
            ax.plot([x, x], [0, BARRA], color=cor, lw=0.7, zorder=1)
        for z in np.arange(0.4, BARRA, 0.4):
            ax.plot([POSTE_ESQ, POSTE_DIR], [z, z], color=cor, lw=0.7, zorder=1)


def _relvado(ax, brand: Brand) -> None:
    """Faixa de relva abaixo da linha de golo, para a baliza assentar no chão."""
    c = brand.colors
    x0, x1 = ax.get_xlim()
    z0 = ax.get_ylim()[0]
    relva = mistura(c["background"], c["lines"], 0.30)
    ax.add_patch(Rectangle((x0, z0), x1 - x0, -z0, color=relva, lw=0, zorder=0))
    ax.plot([x0, x1], [0, 0], color=mistura(relva, c["text"], 0.45), lw=2, zorder=1)


def _rede(ax, brand: Brand, fade: float) -> None:
    """Rede em perspetiva: pano ao fundo e malha a convergir dos postes e da barra."""
    c, bg = brand.colors, brand.colors["background"]
    fio = mistura(bg, mistura(bg, c["text"], 0.09), fade)
    lateral = mistura(bg, mistura(bg, c["text"], 0.16), fade)
    x0 = POSTE_ESQ + PROFUNDIDADE
    x1 = POSTE_DIR - PROFUNDIDADE
    topo = BARRA - PROFUNDIDADE * 0.35

    ax.add_patch(Polygon([(x0, 0), (x1, 0), (x1, topo), (x0, topo)], closed=True,
                         color=mistura(bg, mistura(bg, c["text"], 0.03), fade),
                         lw=0, zorder=1))
    for x in np.arange(x0, x1 + 1e-6, 0.22):
        ax.plot([x, x], [0, topo], color=fio, lw=0.6, zorder=1)
    for z in np.arange(0, topo + 1e-6, 0.22):
        ax.plot([x0, x1], [z, z], color=fio, lw=0.6, zorder=1)
    # malha lateral e de cima, a fugir para o fundo da baliza
    for t in np.linspace(0, 1, 9):
        ax.plot([POSTE_ESQ, x0], [BARRA * t, topo * t], color=lateral, lw=0.7, zorder=1)
        ax.plot([POSTE_DIR, x1], [BARRA * t, topo * t], color=lateral, lw=0.7, zorder=1)
    for t in np.linspace(0, 1, 25):
        ax.plot([POSTE_ESQ + (POSTE_DIR - POSTE_ESQ) * t, x0 + (x1 - x0) * t],
                [BARRA, topo], color=lateral, lw=0.7, zorder=1)
    for xs, zs in (([x0, x0], [0, topo]), ([x1, x1], [0, topo]), ([x0, x1], [topo, topo])):
        ax.plot(xs, zs, color=lateral, lw=1.2, zorder=1)


def baliza_realista(ax, brand: Brand, prog: float = 1.0, rede: float = 1.0,
                    tremor: float = 0.0) -> None:
    """A baliza com volume: relvado, rede em perspetiva e postes com sombra.

    `prog` 0→1 faz a baliza nascer (postes a subir, barra a correr, rede no fim).
    `rede` 0→1 regula a presença da rede por mistura de cor, sem transparência:
    nas cenas com grelha de zonas passa-se 1 menos o avanço da grelha, para a
    rede sair de cena à medida que as zonas entram e não haver duas malhas ao
    mesmo tempo. `tremor` desloca a barra na vertical (bolas que lá batem).
    """
    c, bg = brand.colors, brand.colors["background"]
    _relvado(ax, brand)

    postes, barra = _rampa(prog, 0.0, 0.6), _rampa(prog, 0.45, 1.0)
    fade = _rampa(prog, 0.85, 1.0) * float(np.clip(rede, 0.0, 1.0))
    if fade > 0:
        _rede(ax, brand, fade)

    branco, sombra = c["text"], mistura(c["text"], bg, 0.35)
    caixa = {"boxstyle": "round,pad=0,rounding_size=0.05", "lw": 0, "zorder": 4}
    if postes > 0:
        altura = (BARRA + ESPESSURA) * postes
        ax.add_patch(FancyBboxPatch((POSTE_ESQ - ESPESSURA, 0), ESPESSURA, altura,
                                    fc=branco, **caixa))
        ax.add_patch(FancyBboxPatch((POSTE_DIR, 0), ESPESSURA, altura, fc=branco, **caixa))
        # bisel interior: dá volume ao poste sem precisar de gradiente
        ax.add_patch(Rectangle((POSTE_ESQ - ESPESSURA * 0.35, 0), ESPESSURA * 0.35, altura,
                               fc=sombra, lw=0, zorder=4))
        ax.add_patch(Rectangle((POSTE_DIR + ESPESSURA * 0.65, 0), ESPESSURA * 0.35, altura,
                               fc=sombra, lw=0, zorder=4))
    if barra > 0:
        largura = ((POSTE_DIR - POSTE_ESQ) + 2 * ESPESSURA) * barra
        ax.add_patch(FancyBboxPatch((POSTE_ESQ - ESPESSURA, BARRA + tremor), largura,
                                    ESPESSURA, fc=branco, **caixa))
        ax.add_patch(Rectangle((POSTE_ESQ, BARRA + tremor),
                               max(0.0, largura - 2 * ESPESSURA), ESPESSURA * 0.35,
                               fc=sombra, lw=0, zorder=4))
