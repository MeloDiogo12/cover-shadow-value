"""Agregação da ameaça negada: por lance, por posse e por jogador (D6, D17).

- **Lance** = um frame alinhado (0,2 s). Ameaça negada por cada defesa = soma, sobre as linhas
  desse frame, da parte de Shapley que lhe cabe.
- **Posse**: média ao longo dos frames da posse em que o defesa está em campo (D6): uma linha fechada
  durante toda a posse vale mais do que meio segundo e as posses longas não inflacionam o total.
  Os frames em que o defesa não negou nada contam como zero.
- **Jogador**: soma das médias por posse, em xT. `pct_negada` = o que negou a dividir pela soma, nas
  mesmas posses, das médias da ameaça disponível (linhas avaliadas).
- **Zona e fase (D32)**: a ameaça negada também se divide pelo terço e corredor da posse e pela fase
  defensiva (`phases_of_play` da SkillCorner), para comparar contextos parecidos.
- Linhas excluídas ficam fora da ameaça disponível e contam-se à parte, por motivo: sem pose (D13),
  fora de jogo (D26) e cruzamento (D27).
"""

from __future__ import annotations

import pandas as pd

MOTIVOS = ("sem_pose", "fora_de_jogo", "cruzamento", "mais_de_35m")


def com_motivo(linhas: pd.DataFrame) -> pd.DataFrame:
    """Garante a coluna `motivo`. Resultados da primeira avaliação (v1) não a têm: nessa altura a única
    razão para excluir uma linha era a falta de pose."""
    if "motivo" in linhas:
        return linhas
    return linhas.assign(motivo=linhas["excluida"].map({True: "sem_pose", False: ""}))


def excluidas_por_motivo(linhas: pd.DataFrame) -> dict:
    """Quantas linhas ficaram de fora por cada motivo (todos os motivos aparecem, com 0 se não houve)."""
    linhas = com_motivo(linhas)
    contagem = linhas.loc[linhas["excluida"], "motivo"].value_counts()
    return {m: int(contagem.get(m, 0)) for m in MOTIVOS}


def por_lance(linhas: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por frame: linhas avaliadas e excluídas, ameaça disponível e negada, em xT.
    `linhas` tem `frame`, `posse_id`, `excluida`, `valor_xt`, `ameaca_negada`."""
    avaliadas = linhas[~linhas["excluida"]]
    por_frame = linhas.groupby("frame").agg(posse_id=("posse_id", "first"), n_linhas=("recetor", "size"),
                                            n_excluidas=("excluida", "sum"))
    por_frame["n_avaliadas"] = por_frame["n_linhas"] - por_frame["n_excluidas"]
    motivos = pd.crosstab(linhas["frame"], com_motivo(linhas)["motivo"]).reindex(columns=list(MOTIVOS), fill_value=0)
    por_frame = por_frame.join(motivos.add_prefix("n_"))
    por_frame["disponivel"] = avaliadas.groupby("frame")["valor_xt"].sum()
    por_frame["negada"] = avaliadas.groupby("frame")["ameaca_negada"].sum()
    por_frame = por_frame.fillna({"disponivel": 0.0, "negada": 0.0})
    por_frame["pct_negada"] = (100 * por_frame["negada"] / por_frame["disponivel"]).where(por_frame["disponivel"] > 0)
    return por_frame.reset_index()


def por_lance_e_defesa(atribuicoes: pd.DataFrame) -> pd.DataFrame:
    """Ameaça negada por cada defesa em cada frame (D17, nível lance)."""
    soma = atribuicoes.groupby(["frame", "defesa"], as_index=False)["shapley"].sum()
    return soma.rename(columns={"shapley": "negada"})


def por_posse_e_defesa(atribuicoes: pd.DataFrame, presenca: pd.DataFrame, lances: pd.DataFrame) -> pd.DataFrame:
    """Média por posse, por defesa: `negada` e `disponivel` (ameaça disponível média nos frames
    em que o defesa estava em campo). `presenca`: frame, posse_id, jogador (os defesas)."""
    negada = por_lance_e_defesa(atribuicoes).rename(columns={"defesa": "jogador"})
    base = presenca.merge(lances[["frame", "disponivel"]], on="frame", how="left")
    base = base.merge(negada, on=["frame", "jogador"], how="left").fillna({"negada": 0.0, "disponivel": 0.0})
    return base.groupby(["posse_id", "jogador"], as_index=False).agg(
        frames=("frame", "size"), negada=("negada", "mean"), disponivel=("disponivel", "mean"))


def por_jogador(por_posse: pd.DataFrame, jogadores: pd.DataFrame) -> pd.DataFrame:
    """Soma das médias por posse e % da ameaça disponível negada (D17). `jogadores`: jogador,
    equipa, posicao. Comparar sempre dentro da mesma posição."""
    t = por_posse.groupby("jogador", as_index=False).agg(
        posses=("posse_id", "size"), negada=("negada", "sum"), disponivel=("disponivel", "sum"))
    t["pct_negada"] = (100 * t["negada"] / t["disponivel"]).where(t["disponivel"] > 0)
    return t.merge(jogadores, on="jogador", how="left").sort_values("negada", ascending=False, ignore_index=True)


def fase_por_frame(fases: pd.DataFrame, frames, coluna: str) -> pd.Series:
    """Valor de `coluna` da fase de jogo (SkillCorner `phases_of_play`) em que cada frame cai. Cada fase vai
    de `frame_start` a `frame_end`; frames fora de qualquer fase ficam com NaN. Série indexada pelo frame."""
    frames = pd.Series(sorted(set(frames)), name="frame")
    f = fases[["frame_start", "frame_end", coluna]].sort_values("frame_start")
    junto = pd.merge_asof(frames.to_frame(), f, left_on="frame", right_on="frame_start")
    dentro = junto["frame"] <= junto["frame_end"]
    return junto[coluna].where(dentro).set_axis(junto["frame"])


def fase_defensiva_por_frame(fases: pd.DataFrame, frames) -> pd.Series:
    """Fase defensiva (`team_out_of_possession_phase_type`: bloco alto, médio, baixo…) em cada frame."""
    return fase_por_frame(fases, frames, "team_out_of_possession_phase_type")


def contexto_dos_frames(tabela: pd.DataFrame, fases: pd.DataFrame) -> pd.DataFrame:
    """Um frame por linha com o `terco` e o `corredor` da posse (início da posse, como em D27), a `fase`
    defensiva e a `fase_posse` de quem tem a bola (`build_up` = 1.ª fase de construção, `create` = 2.ª, …).
    `tabela`: tabela de trabalho com `frame`, `terco` e `corredor`."""
    por_frame = tabela.groupby("frame")[["terco", "corredor"]].first()
    por_frame["fase"] = fase_defensiva_por_frame(fases, por_frame.index)
    por_frame["fase_posse"] = fase_por_frame(fases, por_frame.index, "team_in_possession_phase_type")
    return por_frame.reset_index()


def por_zona_ou_fase(lances: pd.DataFrame, contexto: pd.DataFrame, por: str | list) -> pd.DataFrame:
    """Ameaça negada agrupada por `por` ("terco", "corredor", "fase" ou uma lista, p. ex. ["terco", "corredor"]).
    `lances`: saída de `por_lance`. Soma de `negada` e `disponivel` nos lances (frames) de cada grupo,
    `pct_negada` = 100 x negada / disponivel e `lances` = frames com alguma linha avaliada."""
    t = lances[lances["n_avaliadas"] > 0].merge(contexto, on="frame", how="left")
    g = t.groupby(por, dropna=False).agg(lances=("frame", "size"), disponivel=("disponivel", "sum"),
                                         negada=("negada", "sum")).reset_index()
    g["pct_negada"] = (100 * g["negada"] / g["disponivel"]).where(g["disponivel"] > 0)
    return g.sort_values("negada", ascending=False, ignore_index=True)


def resumo_do_jogo(linhas: pd.DataFrame) -> dict:
    """Contagem de linhas avaliadas e excluídas (também por motivo) e ameaça total (para o notebook)."""
    avaliadas = linhas[~linhas["excluida"]]
    return {
        **{f"excl_{m}": n for m, n in excluidas_por_motivo(linhas).items()},
        "linhas": len(linhas), "avaliadas": len(avaliadas), "excluidas": int(linhas["excluida"].sum()),
        "pct_excluidas": 100 * linhas["excluida"].mean(),
        "disponivel": avaliadas["valor_xt"].sum(), "negada": avaliadas["ameaca_negada"].sum(),
        "pct_negada": 100 * avaliadas["ameaca_negada"].sum() / avaliadas["valor_xt"].sum(),
    }


# Em linguagem corrente: a cada instante soma-se o que cada defesa negou nas linhas que fechou. Dentro
# de uma posse faz-se a média por instante (os instantes em que não fechou nada contam como zero),
# e no fim do jogo somam-se as médias das posses de cada jogador. A percentagem compara o que um
# jogador negou com o perigo total que havia nas linhas avaliadas enquanto esteve em campo.
# Para ver onde e em que fase a ameaça se nega, cada instante recebe a zona do campo onde a posse
# começou e a fase defensiva do jogo (bloco alto, médio, baixo…), e somam-se os lances de cada grupo.
