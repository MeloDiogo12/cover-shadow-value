"""Fase 4 · calcula a Cover Shadow Value de um jogo e guarda as tabelas em data/processed/.

Pressupõe a tabela da fase 3 (`preparar_dados.py --jogo <id>`). Com os parâmetros de
`metrica/parametros.py` e sem alterar nenhum, da raiz do projeto:
    uv run python projects/cover_shadow/calcular_metrica.py --jogo 1925299
    uv run python projects/cover_shadow/calcular_metrica.py --jogo 1996435

Escreve `<jogo>_csv_linhas.parquet` (uma linha por linha de passe e frame) e
`<jogo>_csv_atribuicoes.parquet` (parte de Shapley de cada defesa).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from metrica.agregacao import resumo_do_jogo
from metrica.parametros import (
    PADRAO_V3 as PADRAO,  # a v3 (as tabelas `_csv_`); a v3.1 sai de `calcular_v31.py`
)
from metrica.pipeline import calcular_jogo

from footy.ingest import skillcorner as sc
from footy.paths import DATA_PROCESSED

PASTA_SAIDA = DATA_PROCESSED / "cover_shadow"


def caminhos(jogo: int) -> tuple[Path, Path]:
    return PASTA_SAIDA / f"{jogo}_csv_linhas.parquet", PASTA_SAIDA / f"{jogo}_csv_atribuicoes.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogo", type=int, default=sc.JOGO_PADRAO, choices=sc.JOGOS_COM_POSE)
    jogo = ap.parse_args().jogo
    inicio = time.time()
    resultado = calcular_jogo(jogo, PADRAO)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino_linhas, destino_atribuicoes = caminhos(jogo)
    resultado.linhas.to_parquet(destino_linhas, index=False)
    resultado.atribuicoes.to_parquet(destino_atribuicoes, index=False)
    print(f"{time.time() - inicio:.0f} s; frames sem portador: {resultado.frames_sem_portador}")
    print(resumo_do_jogo(resultado.linhas))


if __name__ == "__main__":
    main()

# Em linguagem corrente: este script corre a métrica inteira para um jogo e guarda o resultado, para
# o notebook não ter de recalcular. O número do jogo é um argumento e os parâmetros são sempre os
# de `metrica/parametros.py`, por isso dois jogos correm exatamente com as mesmas regras.
