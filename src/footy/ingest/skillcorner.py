"""Leitor dos dados abertos da SkillCorner (A-League 2024/25, licença MIT).

Fontes (creditar sempre a SkillCorner):
- github.com/SkillCorner/opendata: por jogo, `match.json` (plantéis, dimensões do campo),
  tracking a 10 fps (Git LFS, ~97 MB), eventos dinâmicos (CSV) e fases de jogo (CSV).
- huggingface.co/datasets/SkillCorner/opendata-bodypose: pose corporal a 25 fps, só 2 jogos
  (1925299 e 1996435). Cada jogo é um zip de ~600 MB (`raw/<id>.jsonl.zip`) com um único
  `<id>.jsonl` de ~3,3 GB. O zip não se descompacta: lê-se linha a linha lá de dentro.
  No GitHub há também uma amostra pequena (12,3 s do jogo 1925299).

Coordenadas: metros, origem no centro do campo, x ao longo do campo, y ao longo da largura.
O campo NÃO é sempre 105x68: lê-se de `match.json` (`pitch_length`, `pitch_width`). O sentido de
ataque troca à ida para o 2.º tempo, por isso nunca se assume um sentido fixo.

Este módulo DESCARREGA e LÊ; não interpreta nada. Tudo vai para data/raw/skillcorner/ tal como
chegou (imutável), e a preparação vive noutro sítio.

Uso, da raiz do projeto:
    uv run python -m footy.ingest.skillcorner                     # jogo 1925299 sem tracking + amostra de pose
    uv run python -m footy.ingest.skillcorner --tracking          # junta o tracking (~97 MB)
    uv run python -m footy.ingest.skillcorner --tracking --pose   # e a pose do jogo inteiro (~595 MB)
    uv run python -m footy.ingest.skillcorner --jogo 1996435
"""

from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import requests

from footy.paths import DATA_RAW

PASTA = DATA_RAW / "skillcorner"
GITHUB = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
# O tracking está em Git LFS: o endereço "raw" devolveria só um ponteiro de 133 bytes.
GITHUB_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"
# "resolve" devolve o ficheiro em si (o Hugging Face redireciona para o servidor dos ficheiros).
HUGGINGFACE = "https://huggingface.co/datasets/SkillCorner/opendata-bodypose/resolve/main"

JOGO_PADRAO = 1925299  # Brisbane Roar - Perth Glory, 21/12/2024 (tem body pose)
JOGOS_COM_POSE = (1925299, 1996435)
AMOSTRA_POSE = "sample_1925299_phase406.jsonl.gz"

# Ficheiros pequenos de cada jogo (o tracking é à parte, por ser pesado).
FICHEIROS_LEVES = ["match.json", "dynamic_events.csv", "phases_of_play.csv"]


def pasta_do_jogo(jogo: int) -> Path:
    return PASTA / "matches" / str(jogo)


def caminho(jogo: int, ficheiro: str) -> Path:
    return pasta_do_jogo(jogo) / f"{jogo}_{ficheiro}"


def descarregar(url: str, destino: Path) -> Path:
    """Descarrega para `destino` se ainda não existir. Escreve num .part e só depois renomeia,
    para um download cortado a meio nunca passar por ficheiro completo."""
    if destino.exists():
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".part")
    with requests.get(url, stream=True, timeout=60) as resposta:
        resposta.raise_for_status()
        with open(parcial, "wb") as saida:
            saida.writelines(resposta.iter_content(chunk_size=1 << 20))
    parcial.replace(destino)
    return destino


def descarregar_jogo(jogo: int = JOGO_PADRAO, tracking: bool = False) -> None:
    for ficheiro in FICHEIROS_LEVES:
        descarregar(f"{GITHUB}/matches/{jogo}/{jogo}_{ficheiro}", caminho(jogo, ficheiro))
    if tracking:
        nome = "tracking_extrapolated.jsonl"
        descarregar(f"{GITHUB_LFS}/matches/{jogo}/{jogo}_{nome}", caminho(jogo, nome))


def descarregar_amostra_pose() -> Path:
    return descarregar(f"{GITHUB}/bodypose/{AMOSTRA_POSE}", PASTA / "bodypose" / AMOSTRA_POSE)


def caminho_pose(jogo: int = JOGO_PADRAO) -> Path:
    """Zip da pose do jogo inteiro, tal como vem do Hugging Face."""
    return PASTA / "bodypose" / f"{jogo}.jsonl.zip"


def descarregar_pose(jogo: int = JOGO_PADRAO) -> Path:
    if jogo not in JOGOS_COM_POSE:
        raise ValueError(f"O jogo {jogo} não tem body pose nos dados abertos ({JOGOS_COM_POSE}).")
    return descarregar(f"{HUGGINGFACE}/raw/{jogo}.jsonl.zip", caminho_pose(jogo))


def ler_match(jogo: int = JOGO_PADRAO) -> dict:
    with open(caminho(jogo, "match.json"), encoding="utf-8") as f:
        return json.load(f)


def dimensoes_campo(match: dict) -> tuple[float, float]:
    """(comprimento, largura) em metros, do próprio jogo."""
    return match["pitch_length"], match["pitch_width"]


def ler_eventos(jogo: int = JOGO_PADRAO) -> pd.DataFrame:
    return pd.read_csv(caminho(jogo, "dynamic_events.csv"), low_memory=False)


def ler_fases(jogo: int = JOGO_PADRAO) -> pd.DataFrame:
    return pd.read_csv(caminho(jogo, "phases_of_play.csv"), low_memory=False)


def frames(ficheiro: Path) -> Iterator[dict]:
    """Um frame de cada vez (um objeto JSON por linha), sem carregar o jogo todo para memória.
    Serve para tracking (.jsonl) e para pose (.jsonl, .jsonl.gz ou o .jsonl.zip do jogo inteiro)."""
    if ficheiro.suffix == ".zip":
        yield from _frames_do_zip(ficheiro)
        return
    abrir = gzip.open if ficheiro.suffix == ".gz" else open
    with abrir(ficheiro, "rt", encoding="utf-8") as f:
        for linha in f:
            yield json.loads(linha)


def _frames_do_zip(ficheiro: Path) -> Iterator[dict]:
    """Lê o .jsonl de dentro do zip sem o extrair (3,3 GB descompactado). O zip da SkillCorner
    traz também uma pasta `__MACOSX/` (lixo do macOS), que se ignora."""
    with zipfile.ZipFile(ficheiro) as z:
        nome = next(n for n in z.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX"))
        with z.open(nome) as f:
            for linha in f:
                yield json.loads(linha)


def jogadores_por_frame(frames_: Iterator[dict]) -> pd.DataFrame:
    """Uma linha por jogador e frame: posição, se foi detetado e se tem pose.

    `tem_pose` conta quantos jogadores o frame tem com articulações; `joints` é null quando a
    pose não foi resolvida (fora de câmara, tapado, aglomeração), embora x/y possam existir.
    """
    linhas = []
    for frame in frames_:
        for jogador in frame["player_data"]:
            linhas.append({
                "frame": frame["frame"],
                "periodo": frame["period"],
                "jogador": jogador["player_id"],
                "x": jogador["x"],
                "y": jogador["y"],
                "detetado": jogador["is_detected"],
                "tem_pose": bool(jogador.get("joints")),
            })
    return pd.DataFrame(linhas)


def qualidade_da_pose(tabela: pd.DataFrame) -> dict:
    """Números de qualidade para a fase de Data Understanding (por jogador-frame)."""
    total = len(tabela)
    return {
        "frames": tabela["frame"].nunique(),
        "jogador_frames": total,
        "detetados": float(tabela["detetado"].mean()),
        "com_pose": float(tabela["tem_pose"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogo", type=int, default=JOGO_PADRAO)
    ap.add_argument("--tracking", action="store_true", help="inclui o tracking (~97 MB)")
    ap.add_argument("--pose", action="store_true",
                    help="inclui a pose do jogo inteiro, do Hugging Face (~600 MB)")
    args = ap.parse_args()
    descarregar_jogo(args.jogo, tracking=args.tracking)
    descarregar_amostra_pose()
    if args.pose:
        descarregar_pose(args.jogo)
    print(f"Pronto em {PASTA}")


if __name__ == "__main__":
    main()

# Em linguagem corrente: este ficheiro vai buscar os dados abertos da SkillCorner e guarda-os em
# data/raw/skillcorner/ sem mexer em nada. Tem funções pequenas para abrir o match.json (quem joga,
# tamanho do campo), os eventos e as fases, e para percorrer o tracking ou a pose frame a frame.
# `jogadores_por_frame` transforma cada frame numa tabela simples (quem está onde, se foi detetado,
# se tem pose), e `qualidade_da_pose` conta quantos jogadores têm pose, que é a pergunta central da
# fase 2. Com `--pose` vai também buscar ao Hugging Face a pose do jogo inteiro: é um zip grande, e
# `frames` lê-o linha a linha lá de dentro, sem o descompactar. Não calcula orientações nem
# sombras: isso vive em footy.transform e footy.metrics.
