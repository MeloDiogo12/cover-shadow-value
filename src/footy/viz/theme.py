"""Tema visual: lê config/brand*.yaml e aplica-o ao matplotlib.

Uso típico:
    from footy.viz.theme import load_brand, apply_theme
    brand = load_brand()          # ou load_brand("brand_clube_x")
    apply_theme(brand)

Duas variantes de cor: `load_brand(variante="escura")` (redes sociais e vídeo; por omissão) e
`load_brand(variante="clara")` (artigo, fundo branco). Ver `config/brand.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import matplotlib as mpl
import yaml

from footy.paths import CONFIG


@dataclass
class Brand:
    name: str
    handle: str
    colors: dict[str, str]
    font: dict
    video: dict
    attribution: str = "Dados: {source}"
    extra: dict = field(default_factory=dict)
    variante: str = "escura"

    def __post_init__(self) -> None:
        # texto e linhas finas usam a «tinta»; sem ela (marcas antigas) é a própria cor de preenchimento
        self.colors.setdefault("primary_ink", self.colors["primary"])
        self.colors.setdefault("secondary_ink", self.colors["secondary"])

    def com_variante(self, variante: str) -> Brand:
        """A mesma marca noutra variante de cor (`escura` ou `clara`)."""
        return _brand_da_variante(self, variante)

    def credit(self, source: str) -> str:
        """Texto de rodapé com a fonte dos dados e o handle."""
        return f"{self.attribution.format(source=source)}  ·  {self.handle}"


VARIANTES = ("escura", "clara")


def _brand_da_variante(brand: Brand, variante: str) -> Brand:
    variantes = brand.extra.get("variants")
    if variante not in VARIANTES:
        raise ValueError(f"variante '{variante}' desconhecida; usar uma de {VARIANTES}")
    if not variantes:  # marca com uma só paleta (ex.: de um clube): a variante não muda nada
        return replace(brand, variante=variante)
    return replace(brand, colors=dict(variantes[variante]), variante=variante)


def load_brand(name: str = "brand", variante: str = "escura") -> Brand:
    with open(CONFIG / f"{name}.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if variante not in VARIANTES:
        raise ValueError(f"variante '{variante}' desconhecida; usar uma de {VARIANTES}")
    known = {"name", "handle", "font", "video", "attribution"}
    variantes = raw.get("variants")
    colors = variantes[variante] if variantes else raw["colors"]  # marca com uma só paleta: `colors`
    extra = {k: v for k, v in raw.items() if k not in known | {"colors"}}
    return Brand(**{k: raw[k] for k in known if k in raw}, colors=dict(colors), extra=extra, variante=variante)


def apply_theme(brand: Brand, variante: str | None = None) -> Brand:
    """Aplica cores e fontes da marca a todos os gráficos matplotlib seguintes.

    `variante` (opcional) aplica essa variante de cor em vez da da marca; devolve a marca usada."""
    if variante is not None and variante != brand.variante:
        brand = brand.com_variante(variante)
    c = brand.colors
    mpl.rcParams.update(
        {
            "figure.facecolor": c["background"],
            "axes.facecolor": c["background"],
            "savefig.facecolor": c["background"],
            "text.color": c["text"],
            "axes.labelcolor": c["text"],
            "axes.edgecolor": c["lines"],
            "xtick.color": c["text_muted"],
            "ytick.color": c["text_muted"],
            "font.family": brand.font["family"],
            "font.size": brand.font["body_size"],
            "axes.titlesize": brand.font["title_size"],
        }
    )
    return brand
