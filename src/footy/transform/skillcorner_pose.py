"""Junta o tracking (10 fps) e a pose (25 fps) da SkillCorner nos frames em que coincidem.

As duas fontes cobrem o mesmo tempo de relógio com numerações diferentes:
`frame_pose = 2,5 × frame_tracking`. Só quando `frame_pose` é múltiplo de 5 (e o frame de
tracking é par) é que os dois caem exatamente no mesmo instante; nos outros seria preciso
interpolar. Por isso a tabela de trabalho fica com 5 frames por segundo, sem interpolação.

Os momentos vêm dos eventos `player_possession` (dinâmicos da SkillCorner): do primeiro ao
último frame de cada posse individual, que é quando o adversário pode passar.

Este módulo não escolhe nada sobre a métrica (o que é marcar, que corpo orienta o cone):
só alinha, filtra e organiza.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

# As articulações que a orientação do corpo e da cabeça podem usar (as outras 18 ficam de fora).
ARTICULACOES = (
    "nose", "neck", "lEye", "rEye", "lEar", "rEar",
    "lShoulder", "rShoulder", "midHip", "lHip", "rHip",
)
PASSO_TRACKING = 2  # entre dois frames alinhados há 2 frames de tracking (0,2 s)
FPS_ALINHADO = 5


def frame_tracking_da_pose(frame_pose: int) -> int | None:
    """Frame de tracking que coincide com este frame de pose, ou None se cair entre dois."""
    if frame_pose % 5:
        return None
    return frame_pose * 2 // 5


def alinhado(frame_tracking: int) -> bool:
    """Um frame de tracking tem pose no mesmo instante quando é par."""
    return frame_tracking % PASSO_TRACKING == 0


def posses(eventos: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por posse individual (`player_possession`): quem tem a bola, de que equipa,
    e de que frame a que frame (tracking, inclusive)."""
    p = eventos[eventos["event_type"] == "player_possession"]
    return pd.DataFrame({
        "posse_id": p["event_id"],
        "periodo": p["period"],
        "frame_inicio": p["frame_start"].astype(int),
        "frame_fim": p["frame_end"].astype(int),
        "portador": p["player_id"].astype(int),
        "equipa_com_bola": p["team_id"].astype(int),
    }).reset_index(drop=True)


def frames_alinhados_das_posses(tabela_posses: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por frame alinhado dentro de cada posse. Uma posse de 0 s num frame ímpar
    fica sem nenhum, e isso é contado no notebook. Se duas posses partilharem um frame
    (uma acaba onde a outra começa), o frame fica com a primeira."""
    linhas = [
        {"frame": f, "posse_id": p.posse_id, "portador": p.portador,
         "equipa_com_bola": p.equipa_com_bola}
        for p in tabela_posses.itertuples()
        for f in range(p.frame_inicio, p.frame_fim + 1)
        if alinhado(f)
    ]
    return pd.DataFrame(linhas).drop_duplicates("frame").reset_index(drop=True)


def linhas_do_tracking(frames_: Iterable[dict], quais: set[int]) -> pd.DataFrame:
    """Posição de cada jogador (e da bola) nos frames de tracking pedidos."""
    linhas = []
    for frame in frames_:
        if frame["frame"] not in quais:
            continue
        bola = frame["ball_data"]
        for jogador in frame["player_data"]:
            linhas.append({
                "frame": frame["frame"], "periodo": frame["period"],
                "jogador": jogador["player_id"], "x": jogador["x"], "y": jogador["y"],
                "detetado": jogador["is_detected"], "bola_x": bola["x"], "bola_y": bola["y"],
            })
    return pd.DataFrame(linhas)


def _ate_ao_ultimo(frames_pose: Iterable[dict], ultimo_tracking: int) -> Iterator[dict]:
    """Para de ler a pose assim que passa o último frame que interessa (poupa minutos)."""
    for frame in frames_pose:
        if frame["frame"] > 2.5 * ultimo_tracking:
            return
        yield frame


def _articulacoes_em_colunas(joints: dict | None) -> dict:
    """{'lShoulder_x': ..., 'lShoulder_y': ...} para as ARTICULACOES; vazio se não há pose.
    Fica só x e y: o z da SkillCorner é relativo ao centro do jogador, não é altura."""
    if not joints:
        return {}
    colunas = {}
    for nome in ARTICULACOES:
        if nome in joints:
            colunas[f"{nome}_x"], colunas[f"{nome}_y"] = joints[nome]["xyz"][:2]
    return colunas


def linhas_da_pose(frames_pose: Iterable[dict], quais: set[int]) -> pd.DataFrame:
    """Pose de cada jogador nos frames de pose que coincidem com os frames de tracking pedidos.
    `x_pose`/`y_pose`/`detetado_pose` vêm do ficheiro da pose, que traz a sua própria posição a
    25 fps; nem sempre coincide com a do ficheiro de tracking (o notebook 03 mede quanto)."""
    linhas = []
    for frame in _ate_ao_ultimo(frames_pose, max(quais)):
        numero = frame_tracking_da_pose(frame["frame"])
        if numero not in quais:
            continue
        for jogador in frame["player_data"]:
            linhas.append({
                "frame": numero, "frame_pose": frame["frame"], "jogador": jogador["player_id"],
                "x_pose": jogador["x"], "y_pose": jogador["y"],
                "detetado_pose": jogador.get("is_detected"),
                "tem_pose": bool(jogador.get("joints")),
                **_articulacoes_em_colunas(jogador.get("joints")),
            })
    return pd.DataFrame(linhas)


def jogadores_do_jogo(match: dict) -> pd.DataFrame:
    """Equipa e posição (sigla, ex.: GK, LCB) de cada jogador, a partir do `match.json`."""
    return pd.DataFrame([
        {"jogador": p["id"], "equipa": p["team_id"], "posicao": p["player_role"]["acronym"]}
        for p in match["players"]
    ])


def tabela_de_trabalho(
    tracking: pd.DataFrame, pose: pd.DataFrame, momentos: pd.DataFrame, jogadores: pd.DataFrame
) -> pd.DataFrame:
    """Uma linha por jogador e frame alinhado em posse. A base é o tracking (os jogadores em
    campo); a pose junta-se por cima e, quando falta, `tem_pose` fica False."""
    tabela = (
        tracking.merge(momentos, on="frame", how="inner")
        .merge(jogadores, on="jogador", how="left")
        .merge(pose, on=["frame", "jogador"], how="left")
    )
    tabela["tem_pose"] = tabela["tem_pose"].fillna(False).astype(bool)
    tabela["defende"] = tabela["equipa"] != tabela["equipa_com_bola"]
    return tabela.sort_values(["frame", "jogador"]).reset_index(drop=True)


# Em linguagem corrente: o tracking tem 10 fotografias por segundo e a pose tem 25. Só de 5 em 5
# fotografias de pose é que as duas foram tiradas exatamente no mesmo instante, por isso fica-se só
# com essas (5 por segundo). Dos eventos da SkillCorner tiram-se as posses de bola (quem tem a bola,
# de quando a quando), e só entram os instantes dentro de uma posse. O resultado é uma tabela com
# uma linha por jogador e instante: onde está segundo o tracking, se foi detetado pela câmara, se
# tem pose, as articulações da cabeça, dos ombros e das ancas, a bola, e quem tem a posse.
