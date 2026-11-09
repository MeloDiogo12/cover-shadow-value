"""Fase 7a · gera a animação do C06 (Brisbane Roar – Perth Glory, 21') e uma imagem de cada passo.

Da raiz do repositório:
    uv run python projects/cover_shadow/animar_c06.py                 # vídeo + imagens
    uv run python projects/cover_shadow/animar_c06.py --so-imagens    # só os PNG (rápido, para rever)
    uv run python projects/cover_shadow/animar_c06.py --segundos 8    # só os primeiros 8 s do vídeo (teste)
"""

from __future__ import annotations

import argparse
import json

import gerar_candidatos as gc
from animacoes import APELIDO_REALCE, FPS, Quadro
from figuras_agregados import ROTULOS_CONTEXTO

from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

C06 = {"jogo": 1925299, "frame": 12660, "rotulo": "C06 · 1.ª parte · minuto 21"}


def indice_do_realce() -> dict | None:
    """Índice do defesa em destaque (Jelacic) na fase defensiva do lance, em risco: lido de `artigo_v31.json` (secção `risco.c06`, `numeros_9d.py`)."""
    ficheiro = output_dir("cover_shadow") / "artigo_v31.json"
    if not ficheiro.exists():
        return None
    c06 = json.loads(ficheiro.read_text(encoding="utf-8")).get("risco", {}).get("c06")
    if not c06:
        return None
    achados = {n: v for n, v in c06["por_defesa"].items() if APELIDO_REALCE.lower() in n.lower()}
    if not achados or c06["fase"] not in ROTULOS_CONTEXTO:
        return None
    return {"nome": APELIDO_REALCE, "fase": ROTULOS_CONTEXTO[c06["fase"]], "indice": next(iter(achados.values()))["indice_na_fase"],
            "percentil": next(iter(achados.values()))["percentil_na_fase"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so-imagens", action="store_true")
    ap.add_argument("--segundos", type=float, default=None, help="limita o vídeo a estes segundos (teste)")
    ap.add_argument("--idioma", default="pt")
    ap.add_argument("--nome", default="c06_pt_16x9")
    args = ap.parse_args()
    brand = load_brand(variante="escura")  # animação: variante escura (vídeo)
    apply_theme(brand)
    from animacoes import Configuracao
    quadro = Quadro(gc.carregar(C06["jogo"]), C06["frame"], C06["rotulo"], brand, Configuracao(idioma=args.idioma, extra={"indice_realce": indice_do_realce()}))
    pasta = output_dir("cover_shadow") / "animacoes"
    zero = quadro.plano[0][0] == "abertura"  # o cartão de abertura é o passo 0; os restantes mantêm a numeração anterior
    for i in range(len(quadro.plano)):
        quadro.guardar_imagem_do_passo(i, pasta / f"{args.nome}_passo{i + 1 - zero}.png")
    print(f"{len(quadro.plano)} imagens em {pasta}")
    print(f"legendas reduzidas para caber: {quadro.legendas_reduzidas or 'nenhuma'}")
    if args.so_imagens:
        return
    if args.segundos:
        quadro.n_frames = int(args.segundos * FPS)
    destino = quadro.guardar_video(pasta / f"{args.nome}.mp4")
    print(f"vídeo ({quadro.n_frames / FPS:.0f} s) em {destino}")


if __name__ == "__main__":
    main()
