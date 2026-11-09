"""Fase 3 · constrói a tabela de trabalho do Cover Shadow Value em data/interim/.

Lê do data/raw/skillcorner/ (sem o alterar) o match.json, os eventos dinâmicos, o tracking e o zip
da pose do jogo inteiro, e escreve uma tabela com uma linha por jogador e frame alinhado
(5 por segundo) durante as posses de bola (`player_possession`, decisão D6).

Antes, descarregar os dados (uma vez):
    uv run python -m footy.ingest.skillcorner --tracking --pose

Depois, da raiz do projeto (demora uns minutos: a pose tem 3,3 GB lida linha a linha):
    uv run python projects/cover_shadow/preparar_dados.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from footy.ingest import skillcorner as sc
from footy.paths import DATA_INTERIM
from footy.transform.skillcorner_pose import (
    frames_alinhados_das_posses,
    jogadores_do_jogo,
    linhas_da_pose,
    linhas_do_tracking,
    posses,
    tabela_de_trabalho,
)

PASTA_SAIDA = DATA_INTERIM / "cover_shadow"


def caminho_tabela(jogo: int = sc.JOGO_PADRAO) -> Path:
    return PASTA_SAIDA / f"{jogo}_posses_pose.parquet"


def construir(jogo: int = sc.JOGO_PADRAO) -> Path:
    inicio = time.time()
    momentos = frames_alinhados_das_posses(posses(sc.ler_eventos(jogo)))
    quais = set(momentos["frame"])
    print(f"{len(quais)} frames alinhados em posse")

    tracking = linhas_do_tracking(sc.frames(sc.caminho(jogo, "tracking_extrapolated.jsonl")), quais)
    print(f"tracking lido ({time.time() - inicio:.0f} s)")
    pose = linhas_da_pose(sc.frames(sc.caminho_pose(jogo)), quais)
    print(f"pose lida ({time.time() - inicio:.0f} s)")

    tabela = tabela_de_trabalho(tracking, pose, momentos, jogadores_do_jogo(sc.ler_match(jogo)))
    destino = caminho_tabela(jogo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_parquet(destino, index=False)
    print(f"{len(tabela)} linhas, {tabela['frame'].nunique()} frames -> {destino}")
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogo", type=int, default=sc.JOGO_PADRAO, choices=sc.JOGOS_COM_POSE)
    construir(ap.parse_args().jogo)


if __name__ == "__main__":
    main()

# Em linguagem corrente: este script faz a ponte entre os dados brutos e o notebook da fase 3. Vê
# nos eventos quando é que houve um jogador com a bola, guarda os instantes (5 por segundo) em que
# tracking e pose coincidem, lê o tracking e depois a pose só nesses instantes, e junta tudo numa
# tabela parquet. Como só lê, nunca estraga os originais; se se apagar a tabela, volta-se a correr.
