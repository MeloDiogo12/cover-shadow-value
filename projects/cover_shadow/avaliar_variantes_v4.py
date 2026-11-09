"""Fase 8d · corre a métrica v4 para cada variante e jogo e guarda as tabelas em data/processed/cover_shadow/avaliacao_v4/.

Pressupõe a tabela da fase 3 (`preparar_dados.py`) e a de escolhas (`escolha.py`). Cada execução leva ~3 min por jogo;
as execuções são independentes e correm em paralelo (cada uma usa ~1 GB). O que já está calculado não se repete
(apagar os ficheiros para forçar). Da raiz do projeto:
    uv run python projects/cover_shadow/avaliar_variantes_v4.py --trabalhadores 6
    uv run python projects/cover_shadow/avaliar_variantes_v4.py --so base bola
    uv run python projects/cover_shadow/avaliar_variantes_v4.py --listar            # sem calcular
    uv run python projects/cover_shadow/avaliar_variantes_v4.py --fundo             # com o fundo (D39): só as `a4.VARIANTES_FUNDO`

A `base` calcula também, sobre as mesmas linhas e sombras, os coeficientes do M2 ajustados com os dois jogos e 50
conjuntos tirados da distribuição estimada (a do ajuste no outro jogo), para a incerteza dos coeficientes (C4).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import avaliacao_v4 as a4
import calcular_v4 as c4

from footy.ingest import skillcorner as sc


def alternativos_da_base(jogo: int) -> dict:
    """{nome: coeficientes}: o ajuste com os dois jogos e os conjuntos da incerteza (D38, 8d). Cada conjunto vem do
    ajuste do M2 no outro jogo, o mesmo cujos coeficientes a base usa."""
    conjuntos = c4.conjuntos_de_coeficientes(c4.ajuste_do_outro_jogo(jogo), a4.N_CONJUNTOS)
    return {"ambos": c4.coeficientes_dos_dois_jogos(), **{f"inc{k:02d}": c for k, c in enumerate(conjuntos)}}


def correr(jogo: int, nome: str, com_fundo: bool = False) -> tuple[int, str, float]:
    """Calcula e guarda uma variante. Com `com_fundo` (D39) vai para `a4.PASTA_FUNDO`, sem os conjuntos alternativos."""
    v = a4.variantes_v4()[nome]
    inicio = time.time()
    alt = alternativos_da_base(jogo) if nome == "base" and not com_fundo else None
    r = c4.calcular_jogo_v4(jogo, v.parametros, v.orientacao, v.limite, alt, fundo="completo" if com_fundo else "sem_fundo")
    a4.guardar(r, jogo, nome, a4.PASTA_FUNDO if com_fundo else a4.PASTA)
    return jogo, nome, time.time() - inicio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trabalhadores", type=int, default=4)
    ap.add_argument("--so", nargs="*")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--fundo", action="store_true", help="v4 com o fundo (D39), só as variantes essenciais, noutra pasta")
    args = ap.parse_args()
    pasta = a4.PASTA_FUNDO if args.fundo else a4.PASTA
    nomes = args.so or (list(a4.VARIANTES_FUNDO) if args.fundo else list(a4.variantes_v4()))
    por_calcular = [(j, n) for n in nomes for j in sc.JOGOS_COM_POSE if not a4.variante_calculada(j, n, pasta)]
    print(f"{len(nomes)} variantes x {len(sc.JOGOS_COM_POSE)} jogos; por calcular: {len(por_calcular)}")
    if args.listar:
        print("\n".join(nomes))
        return
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=args.trabalhadores) as pool:
        futuros = [pool.submit(correr, j, n, args.fundo) for j, n in por_calcular]
        for f in as_completed(futuros):
            jogo, nome, segundos = f.result()
            print(f"{time.time() - inicio:6.0f} s  {jogo} {nome} ({segundos:.0f} s)", flush=True)


if __name__ == "__main__":
    main()


# Em linguagem corrente: este script corre a v4 inteira várias vezes, cada uma com um parâmetro diferente (um de cada
# vez), mais uma com o cone virado para a bola, e guarda cada resultado. Na versão base corre ainda a escolha do passe
# com 51 conjuntos de coeficientes (os estimados com os dois jogos e 50 sorteados da incerteza), aproveitando as mesmas
# sombras. Corre em paralelo e, se for interrompido, volta a pegar onde ficou.
