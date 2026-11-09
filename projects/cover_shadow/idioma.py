"""Fase 10a · idioma das figuras do artigo (`"pt"` por omissão, `"en"`).

Cada módulo de figuras tem o seu dicionário `TEXTOS` por idioma (como `animacoes.TEXTOS`); aqui ficam só as peças comuns: a
formatação dos números (vírgula decimal e espaço nos milhares em pt, ponto decimal e vírgula nos milhares em en) e os rótulos
que se repetem em várias figuras. Os números nunca mudam, só a forma de os escrever.
"""

from __future__ import annotations

IDIOMAS = ("pt", "en")

# «1.ª parte · minuto 21» → «first half · minute 21» (o rótulo de um lance vem em português de `figuras_artigo.C06` e `C03`)
PARTES = {"pt": {"1.ª parte": "1.ª parte", "2.ª parte": "2.ª parte"}, "en": {"1.ª parte": "first half", "2.ª parte": "second half"}}
MINUTO = {"pt": "minuto", "en": "minute"}

ROTULOS_CONTEXTO = {
    "pt": {"attacking_third": "terço ofensivo", "middle_third": "terço médio", "defensive_third": "terço defensivo",
           "high_block": "bloco alto", "medium_block": "bloco médio", "low_block": "bloco baixo", "chaotic": "caótica",
           "defending_direct": "defesa de jogo direto", "defending_set_play": "bola parada",
           "defending_quick_break": "contra-ataque", "defending_transition": "transição defensiva"},
    "en": {"attacking_third": "attacking third", "middle_third": "middle third", "defensive_third": "defensive third",
           "high_block": "high block", "medium_block": "mid block", "low_block": "low block", "chaotic": "chaotic",
           "defending_direct": "defending direct play", "defending_set_play": "set play",
           "defending_quick_break": "counter-attack", "defending_transition": "defensive transition"}}


# Palavras de cor que dependem da variante do tema (10c): na escura os defesas são pontos brancos, na clara são quase pretos.
# As legendas geram-se daqui, nunca à mão, para acompanharem `brand.variante`.
COR_DEFESA = {"pt": {"escura": "branco", "clara": "preto"}, "en": {"escura": "white", "clara": "black"}}


def cor_defesa(idioma: str, variante: str = "escura") -> str:
    """«branco»/«preto» (pt) ou «white»/«black» (en): a cor dos defesas na variante."""
    return COR_DEFESA[validar(idioma)][variante]


ATRIBUICAO = {"en": "Data: {source}"}  # o rodapé «Dados: …» da marca (config/brand.yaml) é em português


def marca(brand, idioma: str = "pt"):
    """A marca com o rodapé de crédito no idioma (em pt devolve a própria marca)."""
    from dataclasses import replace
    return replace(brand, attribution=ATRIBUICAO[idioma]) if idioma in ATRIBUICAO else brand


def validar(idioma: str) -> str:
    if idioma not in IDIOMAS:
        raise ValueError(f"idioma '{idioma}' desconhecido; usar um de {IDIOMAS}")
    return idioma


def numero(valor: float, casas: int = 1, idioma: str = "pt", menos: bool = False) -> str:
    """Número para o texto: «0,33» em pt, «0.33» em en. `menos=True` troca o hífen pelo sinal de menos (−)."""
    texto = f"{valor:.{casas}f}"
    if idioma == "pt":
        texto = texto.replace(".", ",")
    return texto.replace("-", "−") if menos else texto


def percentagem(valor: float, casas: int = 1, idioma: str = "pt", menos: bool = False) -> str:
    """«9,6 %» em pt (com espaço), «9.6%» em en."""
    return numero(valor, casas, idioma, menos) + (" %" if idioma == "pt" else "%")


def milhares(n: int, idioma: str = "pt") -> str:
    """«14 584» em pt, «14,584» em en."""
    texto = f"{int(n):,}"
    return texto.replace(",", " ") if idioma == "pt" else texto


def formatadores(idioma: str = "pt", menos: bool = False):
    """(num, pct, mil) já ligados ao idioma: `num(v, casas)`, `pct(v, casas)`, `mil(n)`."""
    validar(idioma)
    return (lambda v, casas=1: numero(v, casas, idioma, menos),
            lambda v, casas=1: percentagem(v, casas, idioma, menos),
            lambda n: milhares(n, idioma))


def rotulo_do_lance(rotulo: str, idioma: str = "pt") -> str:
    """Traduz «C06 · 1.ª parte · minuto 21» (só a parte e o minuto; o resto fica)."""
    if idioma == "pt":
        return rotulo
    for pt_, en in PARTES["en"].items():
        rotulo = rotulo.replace(pt_, en)
    return rotulo.replace(MINUTO["pt"], MINUTO["en"])


# Em linguagem corrente: as figuras escrevem os números sempre da mesma maneira, mas cada língua tem a sua: em português a vírgula
# separa os decimais e o espaço os milhares; em inglês é o ponto e a vírgula. Este módulo guarda essa regra num só sítio.
