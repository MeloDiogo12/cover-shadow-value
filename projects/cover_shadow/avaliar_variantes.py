"""Fase 5 · corre a métrica para cada variante de parâmetros e jogo e guarda os resultados.

Pressupõe a tabela da fase 3 de cada jogo (`preparar_dados.py`). Cada execução completa demora
cerca de 100 s por jogo; as execuções são independentes e correm em paralelo. O que já está
calculado não se repete (apagar o ficheiro para forçar). Da raiz do projeto:
    uv run python projects/cover_shadow/avaliar_variantes.py --trabalhadores 6
    uv run python projects/cover_shadow/avaliar_variantes.py --so base bola      # só estas variantes
    uv run python projects/cover_shadow/avaliar_variantes.py --listar            # sem calcular
    uv run python projects/cover_shadow/avaliar_variantes.py --versao v3         # base v3 + bloqueio por raio

Escreve em `data/processed/cover_shadow/avaliacao_<versao>/<jogo>_<variante>_{linhas,atribuicoes}.parquet`
(`--versao v2`, por omissão: sensibilidade completa; `--versao v3`: só a base e as variantes do bloqueio).
A primeira avaliação (v1, antes das decisões D25 a D30) está em `.../avaliacao/` e não se recalcula.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from avaliacao import (
    PASTAS,
    guardar_variante,
    variante_calculada,
    variantes,
    variantes_v3,
    variantes_v31,
)
from metrica.pipeline import calcular_jogo

from footy.ingest import skillcorner as sc

VARIANTES = {"v2": variantes, "v3": variantes_v3, "v3_1": variantes_v31}


def correr(jogo: int, nome: str, versao: str) -> tuple[int, str, float]:
    parametros, orientacao = VARIANTES[versao]()[nome]
    inicio = time.time()
    guardar_variante(calcular_jogo(jogo, parametros, orientacao), jogo, nome, PASTAS[versao])
    return jogo, nome, time.time() - inicio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trabalhadores", type=int, default=4, help="execuções em paralelo (cada uma usa ~1 GB)")
    ap.add_argument("--so", nargs="*", help="só estas variantes (nomes como em `--listar`)")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--versao", choices=list(VARIANTES), default="v2")
    args = ap.parse_args()
    nomes = args.so or list(VARIANTES[args.versao]())
    por_calcular = [(j, n) for n in nomes for j in sc.JOGOS_COM_POSE
                    if not variante_calculada(j, n, PASTAS[args.versao])]
    print(f"{len(nomes)} variantes x {len(sc.JOGOS_COM_POSE)} jogos; por calcular: {len(por_calcular)}")
    if args.listar:
        print("\n".join(nomes))
        return
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=args.trabalhadores) as pool:
        futuros = [pool.submit(correr, j, n, args.versao) for j, n in por_calcular]
        for f in as_completed(futuros):
            jogo, nome, segundos = f.result()
            print(f"{time.time() - inicio:6.0f} s  {jogo} {nome} ({segundos:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

# Em linguagem corrente: este script corre a métrica inteira várias vezes, cada uma com um
# parâmetro diferente (um de cada vez), mais uma com o cone virado para a bola, e guarda cada
# resultado. Como as corridas não dependem umas das outras, correm em paralelo; e se for
# interrompido, volta a pegar onde ficou.
