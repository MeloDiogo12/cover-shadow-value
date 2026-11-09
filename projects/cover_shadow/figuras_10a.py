"""Fase 10a · as 16 figuras do artigo num idioma (`pt` ou `en`), com os mesmos números e o mesmo desenho.

Da raiz do repositório (o grupo `valor` não é preciso aqui):
    uv run python projects/cover_shadow/figuras_10a.py --idioma en                       # artigo_en/ (variante clara, + legendas_en.md)
    uv run python projects/cover_shadow/figuras_10a.py --idioma pt                       # artigo/ (variante clara)
    uv run python projects/cover_shadow/figuras_10a.py --idioma en --variante escura     # redes_en/  (e --idioma pt → redes_pt/)

Cada figura é desenhada pela mesma função do artigo português (`idioma="pt"` por omissão); aqui só se escolhe o idioma e a pasta.
Não toca em `outputs/cover_shadow/artigo/`: as figuras portuguesas só se regeneram com `figuras_9d.py` / `figuras_9e.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import figuras_9b
import figuras_9d as f9
import figuras_agregados as ag
import figuras_artigo as fa
import figuras_lance as fl
import gerar_candidatos as gc
import idioma as idi
import orientacao_defesas as od

from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

FIGURAS = ("agregado_versoes", "c06_passo4_cones_pelo_tronco", "c06_passo5_probabilidade_de_o_passe_chegar", "c06_passo8_resumo",
           "agregado_revisoes", "agregado_fase_e_terco", "agregado_c3_sombra_escolhida", "agregado_c5_tronco_vs_bola",
           "c06_corpo_vs_bola", "agregado_c4_estabilidade", "novo_c8_escolha_do_passe", "novo_moedas_por_fase",
           "novo_pressao_alta_goodwin", "novo_xpass_calibracao_e_gamma", "c03_lance", "orientacao_bola_vs_frente")  # número no artigo = posição + 1


def gerar(idioma: str, pasta: Path, variante: str = "clara") -> dict:
    """Desenha as 16 figuras em `pasta` e devolve os números de cada uma (os mesmos nos dois idiomas).
    `variante`: `clara` (artigo) ou `escura` (redes sociais)."""
    idi.validar(idioma)
    brand = idi.marca(load_brand(variante=variante), idioma)
    apply_theme(brand)
    pasta.mkdir(parents=True, exist_ok=True)
    numeros = f9.ler_numeros()
    JG = ag.carregar_jogos()
    ag.JG_TITULOS.update({j: JG[j]["titulo"] for j in JG})
    J6, J3, JP = gc.carregar(fa.C06["jogo"]), gc.carregar(fa.C03["jogo"]), gc.carregar(f9.PRESSAO_ALTA["jogo"])
    n: dict = {}
    n["versoes"] = ag.figura_versoes(brand, pasta / "agregado_versoes.png", ag.TEXTOS[idioma]["vs_nota_final"], idioma=idioma)
    n["c06_passos"] = fl.c06_passo_a_passo(J6, fa.C06["frame"], fa.C06["rotulo"], brand, pasta, moeda="risco", passos=(4, 5, 8), idioma=idioma)
    n["revisoes"] = ag.figura_revisoes(brand, pasta / "agregado_revisoes.png", idioma=idioma)
    n["fase_terco"] = f9.figura_fase_e_terco_risco(numeros, brand, pasta / "agregado_fase_e_terco.png", idioma)
    n["c3"] = ag.figura_c3(JG, brand, pasta / "agregado_c3_sombra_escolhida.png", idioma=idioma)
    n["c5"] = f9.figura_c5_risco(JG, numeros, brand, pasta / "agregado_c5_tronco_vs_bola.png", idioma)
    n["c06_corpo_bola"] = fl.c06_corpo_vs_bola(J6, fa.C06["frame"], fa.C06["rotulo"], brand, pasta, moeda="risco", idioma=idioma)
    n["c4"] = f9.figura_c4_risco(JG, numeros, brand, pasta / "agregado_c4_estabilidade.png", idioma)
    titulos = {j: JG[j]["titulo"] for j in JG}
    n["novas"] = {"c8": figuras_9b.figura_c8(brand, pasta / "novo_c8_escolha_do_passe.png", titulos, idioma),
                  "xpass": figuras_9b.figura_xpass(brand, pasta / "novo_xpass_calibracao_e_gamma.png", idioma),
                  "moedas": f9.figura_moedas_3(numeros, brand, pasta / "novo_moedas_por_fase.png", idioma)}
    n["pressao_alta"] = f9.figura_pressao_alta(JP, numeros, brand, pasta / "novo_pressao_alta_goodwin.png", idioma)
    n["c03"] = fl.c03_lance(J3, fa.C03["frame"], fa.C03["rotulo"], brand, pasta, "Klimala", "Isaias", moeda="risco", idioma=idioma)
    n["bola_frente"] = od.figura_bola_vs_frente(od.defesas_com_angulo(JG), brand, pasta / "orientacao_bola_vs_frente.png", idioma)
    return n


def hashes(pasta: Path) -> dict[str, str]:
    return {f: hashlib.sha256((pasta / f"{f}.png").read_bytes()).hexdigest() for f in FIGURAS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idioma", choices=idi.IDIOMAS, default="en")
    parser.add_argument("--variante", choices=("clara", "escura"), default="clara", help="clara = artigo; escura = redes sociais")
    parser.add_argument("--destino", type=Path, default=None, help="por omissão: artigo (pt) / artigo_en (en) na variante clara; redes_pt / redes_en na escura")
    args = parser.parse_args()
    if args.destino is None:
        pasta = {("clara", "pt"): "artigo", ("clara", "en"): "artigo_en", ("escura", "pt"): "redes_pt", ("escura", "en"): "redes_en"}
        args.destino = output_dir("cover_shadow") / pasta[(args.variante, args.idioma)]
    n = gerar(args.idioma, args.destino, args.variante)
    if (args.idioma, args.variante) != ("pt", "clara"):  # `artigo/` já tem `numeros_das_figuras.json` (fase 9d)
        (args.destino / "numeros_das_figuras_10a.json").write_text(json.dumps(n, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if args.idioma == "en" and args.variante == "clara":
        import legendas_en
        legendas_en.escrever(args.destino, n)
    for f, h in hashes(args.destino).items():
        print(h[:16], f)
    print(f"16 figuras ({args.idioma}) em {args.destino}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script desenha as 16 figuras do artigo no idioma pedido. Para o inglês escreve-as numa pasta própria com
# os mesmos nomes de ficheiro e uma legenda por figura; para o português serve para provar que o código novo desenha exatamente as
# mesmas imagens de antes (os hashes têm de coincidir).
