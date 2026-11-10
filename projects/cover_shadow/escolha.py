"""Fase 8a · tabela de escolhas do passe (uma linha por passe e por opção de passe), sem modelo.

Para cada passe real (`player_possession` com `end_type == "pass"`) olha-se para o último frame
alinhado da posse (5 por segundo) e listam-se as opções que a métrica define nesse frame: todos
os colegas, menos os em fora de jogo (D26) e os cruzamentos (D27). A opção que o jogador escolheu
é o recetor visado (`player_targeted_id`, a intenção, não só quem recebeu). A cada opção juntam-se
as variáveis da linha (comprimento, avanço, xT, marcação do recetor) e a sombra da v3 (cone pelo
tronco) e da variante "bola", já calculadas em `data/processed/cover_shadow/avaliacao_v3/`.

A tabela alimenta o logit condicional da fase 8b. Aqui só se monta e se descreve.

Duas escolhas de leitura dos dados, para a fase 8b decidir:
- `high_pass` só existe em passes completos (`pass_outcome` falhado ou fora de jogo ficam sem valor),
  por isso "rasteiro" não é conhecido para os passes falhados.
- As variáveis da SkillCorner (`xpass_sc`, `score_sc`) só existem para as opções que ela detetou
  (`passing_option`), não para todos os colegas.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from avaliacao import PASTA_AVALIACAO_V3
from entradas import exigir
from metrica.linhas import avaliar_frame, recetores_com_valor
from metrica.parametros import (
    PADRAO_V3 as PADRAO,  # a tabela de escolhas é da v3 (D37 aplica o limite de 35 m depois)
)
from metrica.pipeline import ORIENTACOES, carregar_tabela, preparar_defesas
from metrica.xt import sentido_de_ataque, xt_da_posicao

from footy.ingest import skillcorner as sc
from footy.paths import DATA_PROCESSED

PASTA_SAIDA = DATA_PROCESSED / "cover_shadow"
SEGUNDOS_POR_FRAME = 0.1  # o tracking tem 10 frames por segundo
LIMITE_INTERVALO_S = 0.4  # proposta: máximo entre o frame da escolha e o passe (a confirmar)
FAIXAS_DE_COMPRIMENTO = (0, 10, 20, 30, 35, 40, np.inf)
MOTIVOS_FORA_DO_CONJUNTO = ("fora_de_jogo", "cruzamento")  # D26 e D27; "sem_pose" fica no conjunto, marcada


# ---------------------------------------------------------------- passes e frame da escolha

def passes_do_jogo(eventos: pd.DataFrame) -> pd.DataFrame:
    """Um passe por linha: quem passa, recetor visado (a intenção), tipo e resultado.
    `comprimento_evento` é o `pass_distance` da SkillCorner, que não existe nos passes falhados."""
    p = eventos[(eventos["event_type"] == "player_possession") & (eventos["end_type"] == "pass")]
    return pd.DataFrame({
        "posse_id": p["event_id"].to_numpy(), "periodo": p["period"].to_numpy(), "equipa": p["team_id"].to_numpy(),
        "frame_passe": p["frame_end"].to_numpy(), "passador": p["player_id"].to_numpy(),
        "recetor_visado": p["player_targeted_id"].to_numpy(),
        "high_pass": pd.array(p["high_pass"].to_numpy(), dtype="boolean"),
        "pass_outcome": p["pass_outcome"].to_numpy(), "start_type": p["start_type"].to_numpy(),
        "comprimento_evento": p["pass_distance"].to_numpy(),
    })


def juntar_frame_da_escolha(passes: pd.DataFrame, tabela: pd.DataFrame) -> pd.DataFrame:
    """Último frame alinhado da posse (na tabela da fase 3) e segundos que o separam do passe.
    Passes sem nenhum frame alinhado (posses de 0 s num frame ímpar) ficam com `frame` vazio."""
    ultimo = tabela.groupby("posse_id")["frame"].max()
    frame = passes["posse_id"].map(ultimo)
    return passes.assign(frame=frame, intervalo_s=(passes["frame_passe"] - frame) * SEGUNDOS_POR_FRAME)


# ---------------------------------------------------------------- variáveis das opções

def distancia_ao_adversario_mais_proximo(posicoes: pd.DataFrame) -> pd.DataFrame:
    """Por frame, a distância (m) de cada jogador de ataque ao adversário mais próximo, guarda-redes
    incluído (aqui interessa o espaço do recetor, não quem projeta sombra). `posicoes`: frame,
    jogador, x, y, defende (True nos adversários da equipa com bola)."""
    partes = []
    for frame, jogadores in posicoes.groupby("frame"):
        ataque, defesa = jogadores[~jogadores["defende"]], jogadores[jogadores["defende"]]
        if ataque.empty or defesa.empty:
            continue
        dx = ataque["x"].to_numpy()[:, None] - defesa["x"].to_numpy()[None, :]
        dy = ataque["y"].to_numpy()[:, None] - defesa["y"].to_numpy()[None, :]
        partes.append(pd.DataFrame({"frame": frame, "recetor": ataque["jogador"].to_numpy(),
                                    "dist_adversario": np.hypot(dx, dy).min(axis=1)}))
    return pd.concat(partes, ignore_index=True)


def recetores_marcados(defesas: pd.DataFrame) -> pd.DataFrame:
    """(frame, recetor) de quem tem algum defesa a marcá-lo pela regra da D33 (`a_marcar`)."""
    marcados = defesas.loc[defesas["a_marcar"], ["frame", "atacante"]].drop_duplicates()
    return marcados.rename(columns={"atacante": "recetor"}).assign(recetor_marcado=True)


def opcoes_da_skillcorner(eventos: pd.DataFrame, passes: pd.DataFrame) -> pd.DataFrame:
    """`xpass_completion` e `passing_option_score` da SkillCorner para cada (posse, recetor), só quando ela
    detetou essa opção. Se houver mais de uma deteção, fica a que cobre o frame do passe (ou a mais próxima)."""
    o = eventos[eventos["event_type"] == "passing_option"]
    o = pd.DataFrame({
        "posse_id": o["associated_player_possession_event_id"].to_numpy(), "recetor": o["player_id"].to_numpy(),
        "frame_inicio": o["frame_start"].to_numpy(), "frame_fim": o["frame_end"].to_numpy(),
        "xpass_sc": o["xpass_completion"].to_numpy(), "score_sc": o["passing_option_score"].to_numpy(),
        "visado_sc": o["targeted"].to_numpy(),
    }).merge(passes[["posse_id", "frame_passe"]], on="posse_id")
    o["afastamento"] = np.maximum(np.maximum(o["frame_inicio"] - o["frame_passe"], o["frame_passe"] - o["frame_fim"]), 0)
    o = o.sort_values(["afastamento", "frame_fim"], ascending=[True, False]).drop_duplicates(["posse_id", "recetor"])
    return o[["posse_id", "recetor", "xpass_sc", "score_sc", "visado_sc"]]


# ---------------------------------------------------------------- tabela de escolhas

def versoes_da_sombra(tronco: pd.DataFrame, bola: pd.DataFrame) -> pd.DataFrame:
    """Uma linha de passe por (frame, recetor) com a sombra (1 − p_chegar) e `n_sombras` das duas versões.
    A sombra fica vazia nas linhas sem pose (a métrica não as avalia)."""
    b = bola[["frame", "recetor", "p_chegar", "n_sombras"]].rename(
        columns={"p_chegar": "p_chegar_bola", "n_sombras": "n_sombras_bola"})
    t = tronco[["frame", "posse_id", "recetor", "motivo", "valor_xt", "p_chegar", "n_sombras"]].rename(
        columns={"p_chegar": "p_chegar_tronco", "n_sombras": "n_sombras_tronco"})
    v = t.merge(b, on=["frame", "recetor"], how="left", validate="1:1")
    v["sombra_tronco"] = 1 - v["p_chegar_tronco"]
    v["sombra_bola"] = 1 - v["p_chegar_bola"]
    return v.drop(columns=["p_chegar_tronco", "p_chegar_bola"])


def construir_opcoes(passes: pd.DataFrame, sombras: pd.DataFrame, posicoes: pd.DataFrame, marcados: pd.DataFrame,
                     skillcorner: pd.DataFrame, campo: tuple[float, float]) -> pd.DataFrame:
    """Passe × opção, com todas as variáveis. Inclui as opções excluídas (`motivo`), para se saber porque é que
    o recetor visado ficou de fora; `conjunto_de_escolhas` tira-as. `passes` tem `frame` e `sentido`;
    `posicoes` tem frame, jogador, x, y, defende no frame da escolha."""
    comprimento_campo, largura_campo = campo
    o = passes.dropna(subset=["frame"]).merge(sombras, on=["frame", "posse_id"])
    o["escolhida"] = (o["recetor"] == o["recetor_visado"]).astype(int)

    pos = posicoes[["frame", "jogador", "x", "y"]]
    o = o.merge(pos.rename(columns={"jogador": "recetor", "x": "x_recetor", "y": "y_recetor"}), on=["frame", "recetor"], how="left")
    o = o.merge(pos.rename(columns={"jogador": "passador", "x": "x_passador", "y": "y_passador"}), on=["frame", "passador"], how="left")
    dx, dy = o["x_recetor"] - o["x_passador"], o["y_recetor"] - o["y_passador"]
    o["comprimento"] = np.hypot(dx, dy)
    o["avanco"] = dx * o["sentido"]  # positivo = em direção à baliza adversária
    xt_origem = xt_da_posicao(o["x_passador"], o["y_passador"], o["sentido"], comprimento_campo, largura_campo)
    xt_destino = xt_da_posicao(o["x_recetor"], o["y_recetor"], o["sentido"], comprimento_campo, largura_campo)
    o["ganho_xt"] = xt_destino - xt_origem  # com sinal; `valor_xt` é só a parte positiva

    o = o.merge(marcados, on=["frame", "recetor"], how="left")
    o["recetor_marcado"] = o["recetor_marcado"].eq(True).astype(int)
    o = o.merge(distancia_ao_adversario_mais_proximo(posicoes), on=["frame", "recetor"], how="left")
    return o.merge(skillcorner, on=["posse_id", "recetor"], how="left")


def conjunto_de_escolhas(opcoes: pd.DataFrame) -> pd.DataFrame:
    """As opções que a métrica avalia ou marca como sem pose: sem fora de jogo (D26) nem cruzamentos (D27)."""
    return opcoes[~opcoes["motivo"].isin(MOTIVOS_FORA_DO_CONJUNTO)]


def marcar_passes(passes: pd.DataFrame, opcoes: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a cada passe o que se passou com as suas opções: se tem linhas calculadas, se o recetor
    visado está no conjunto (ou porque saiu: fora de jogo, cruzamento, ausente do frame) e quantas
    opções ficam sem pose."""
    no_conjunto = conjunto_de_escolhas(opcoes)
    visado = opcoes[opcoes["escolhida"] == 1].set_index("posse_id")["motivo"]
    por_passe = pd.DataFrame({
        "n_opcoes": no_conjunto.groupby("posse_id").size(),
        "n_sem_pose": no_conjunto[no_conjunto["motivo"] == "sem_pose"].groupby("posse_id").size(),
    })
    p = passes.join(por_passe, on="posse_id")
    p["tem_linhas"] = p["posse_id"].isin(opcoes["posse_id"])
    p["n_opcoes"] = p["n_opcoes"].fillna(0).astype(int)
    p["n_sem_pose"] = p["n_sem_pose"].fillna(0).astype(int)
    p["motivo_visado"] = p["posse_id"].map(visado)
    p.loc[p["tem_linhas"] & p["motivo_visado"].isna(), "motivo_visado"] = "ausente"  # nem tem linha no frame
    p["visado_no_conjunto"] = p["motivo_visado"].isin(["", "sem_pose"])
    p["conjunto_completo"] = p["visado_no_conjunto"] & (p["n_sem_pose"] == 0)
    return p


def comprimento_do_passe(passes: pd.DataFrame, opcoes: pd.DataFrame) -> pd.Series:
    """Comprimento aproximado de cada passe: o `pass_distance` da SkillCorner e, nos passes falhados (sem esse
    valor), a distância do passador ao recetor visado no frame da escolha."""
    linha_visada = opcoes[opcoes["escolhida"] == 1].drop_duplicates("posse_id").set_index("posse_id")["comprimento"]
    return passes["comprimento_evento"].fillna(passes["posse_id"].map(linha_visada))


# ---------------------------------------------------------------- contagens e descrições

def e_rasteiro(passes: pd.DataFrame, criterio: str = "conhecidos") -> pd.Series:
    """`conhecidos`: `high_pass` é False. `nao_ar`: tudo o que não é `high_pass` True (os falhados contam
    como rasteiros). `todos`: sem filtro."""
    ar = passes["high_pass"]
    if criterio == "conhecidos":
        return ar.eq(False).fillna(False).astype(bool)
    if criterio == "nao_ar":
        return ~ar.eq(True).fillna(False).astype(bool)
    if criterio == "todos":
        return pd.Series(True, index=passes.index)
    raise ValueError("criterio deve ser conhecidos, nao_ar ou todos")


def funil(passes: pd.DataFrame, limite_s: float = LIMITE_INTERVALO_S, criterio: str = "conhecidos") -> pd.Series:
    """Passes que sobrevivem a cada passo, por ordem. `passes` já passou por `marcar_passes`."""
    ok = pd.Series(True, index=passes.index)
    passos = {"passes": ok}
    passos["rasteiros"] = passos["passes"] & e_rasteiro(passes, criterio)
    passos["frame da escolha no limite, com linhas"] = passos["rasteiros"] & passes["tem_linhas"] & (passes["intervalo_s"] <= limite_s)
    passos["recetor visado no conjunto"] = passos["frame da escolha no limite, com linhas"] & passes["visado_no_conjunto"]
    passos["todas as linhas avaliáveis"] = passos["recetor visado no conjunto"] & passes["conjunto_completo"]
    return pd.Series({nome: int(m.sum()) for nome, m in passos.items()}, name="passes")


def faixas_de_comprimento(passes: pd.DataFrame) -> pd.DataFrame:
    """Passes por faixa de comprimento: nº, nº pelo ar, nº sem informação de `high_pass` e % pelo ar entre
    os que a têm. `passes` tem `comprimento_passe`."""
    faixa = pd.cut(passes["comprimento_passe"], FAIXAS_DE_COMPRIMENTO, right=False)
    g = passes.groupby(faixa, observed=False)["high_pass"]
    t = pd.DataFrame({"passes": g.size(), "pelo_ar": g.apply(lambda s: int(s.eq(True).sum())),
                      "sem_informacao": g.apply(lambda s: int(s.isna().sum()))})
    t["pct_pelo_ar"] = 100 * t["pelo_ar"] / (t["passes"] - t["sem_informacao"]).replace(0, np.nan)
    return t


VARIAVEIS = ("comprimento", "avanco", "valor_xt", "ganho_xt", "recetor_marcado", "dist_adversario",
             "sombra_tronco", "sombra_bola", "n_sombras_tronco", "n_sombras_bola", "xpass_sc", "score_sc")


def medias_por_escolha(conjunto: pd.DataFrame) -> pd.DataFrame:
    """Média de cada variável nas opções escolhidas e nas não escolhidas (ignora valores em falta)."""
    return conjunto.groupby("escolhida")[list(VARIAVEIS)].mean().T.rename(columns={0: "não escolhidas", 1: "escolhidas"})


def correlacao_sombra_marcacao(conjunto: pd.DataFrame) -> pd.DataFrame:
    """Correlação (Pearson) da sombra de cada versão com o recetor marcado, o comprimento e a distância ao
    adversário, nas opções avaliadas: mostra o risco de confundir sombra com marcação."""
    avaliadas = conjunto.dropna(subset=["sombra_tronco", "sombra_bola"])
    return avaliadas[["sombra_tronco", "sombra_bola"]].apply(
        lambda s: avaliadas[["recetor_marcado", "comprimento", "dist_adversario", "avanco"]].corrwith(s))


def sombras_so_com_defesas_com_pose(tabela: pd.DataFrame, defesas: pd.DataFrame, orientacao: str, match: dict,
                                    frames) -> pd.DataFrame:
    """Sombra (1 − p_chegar) de cada linha dos `frames`, contando só os defesas que têm pose: onde a métrica
    exclui a linha por falta de pose (D13), isto dá um valor mínimo (faltam as sombras dos defesas sem pose).
    `defesas` vem de `preparar_defesas` com a mesma `orientacao`; devolve frame, recetor e `sombra_<orientacao>_pose`."""
    comprimento, largura = sc.dimensoes_campo(match)
    com_pose = defesas[defesas["orientacao"].notna() & defesas["frame"].isin(frames)]
    por_frame = dict(tuple(com_pose.groupby("frame")))
    partes = []
    for frame, jogadores in tabela[tabela["frame"].isin(frames)].groupby("frame"):
        if not (jogadores["jogador"] == jogadores["portador"]).any():
            continue
        primeiro = jogadores.iloc[0]
        sentido = sentido_de_ataque(match, primeiro["equipa_com_bola"], primeiro["periodo"])
        origem, recetores = recetores_com_valor(jogadores, sentido, comprimento, largura, PADRAO)
        linhas, _ = avaliar_frame(origem, recetores, por_frame.get(frame, com_pose.iloc[:0]), PADRAO)
        partes.append(pd.DataFrame({"frame": frame, "recetor": linhas["recetor"].to_numpy(),
                                    f"sombra_{orientacao}_pose": 1 - linhas["p_chegar"].to_numpy()}))
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------- um jogo inteiro

def escolhas_do_jogo(jogo: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(passes, opções) de um jogo, com a v3 já calculada. Recalcula só a marcação (`preparar_defesas`)."""
    match, eventos, tabela = sc.ler_match(jogo), sc.ler_eventos(jogo), carregar_tabela(jogo)
    corre = "avaliar_variantes.py --versao v3 --so base bola"
    tronco = pd.read_parquet(exigir(PASTA_AVALIACAO_V3 / f"{jogo}_base_linhas.parquet", corre))
    bola = pd.read_parquet(exigir(PASTA_AVALIACAO_V3 / f"{jogo}_bola_linhas.parquet", corre))

    passes = juntar_frame_da_escolha(passes_do_jogo(eventos), tabela)
    passes["sentido"] = [sentido_de_ataque(match, e, per) for e, per in zip(passes["equipa"], passes["periodo"])]
    posicoes = tabela[tabela["frame"].isin(passes["frame"].dropna())][["frame", "jogador", "x", "y", "defende"]]
    defesas = {o: preparar_defesas(tabela, match, PADRAO, o) for o in ORIENTACOES}
    marcados = recetores_marcados(defesas["tronco"])

    opcoes = construir_opcoes(passes, versoes_da_sombra(tronco, bola), posicoes, marcados,
                              opcoes_da_skillcorner(eventos, passes), sc.dimensoes_campo(match))
    frames = set(opcoes["frame"])
    for orientacao in ORIENTACOES:
        so_com_pose = sombras_so_com_defesas_com_pose(tabela, defesas[orientacao], orientacao, match, frames)
        opcoes = opcoes.merge(so_com_pose, on=["frame", "recetor"], how="left", validate="m:1")
    passes = marcar_passes(passes, opcoes)
    passes["comprimento_passe"] = comprimento_do_passe(passes, opcoes)
    return passes.assign(jogo=jogo), conjunto_de_escolhas(opcoes).assign(jogo=jogo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jogos", type=int, nargs="+", default=list(sc.JOGOS_COM_POSE))
    jogos = ap.parse_args().jogos
    todos = []
    for jogo in jogos:
        inicio = time.time()
        todos.append(escolhas_do_jogo(jogo))
        print(f"{jogo}: {len(todos[-1][0])} passes, {len(todos[-1][1])} opções ({time.time() - inicio:.0f} s)")
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    passes = pd.concat([p for p, _ in todos], ignore_index=True)
    opcoes = pd.concat([o for _, o in todos], ignore_index=True)
    # cada opção leva as marcas do seu passe, para a fase 8b filtrar sem voltar à tabela de passes
    marcas = ["posse_id", "jogo", "visado_no_conjunto", "conjunto_completo", "comprimento_passe"]
    opcoes = opcoes.merge(passes[marcas], on=["posse_id", "jogo"])
    opcoes.to_parquet(PASTA_SAIDA / "escolhas_v3.parquet", index=False)
    passes.to_parquet(PASTA_SAIDA / "passes_escolha_v3.parquet", index=False)


if __name__ == "__main__":
    main()


# Em linguagem corrente: para cada passe que um jogador deu, vai-se ao último instante em que ainda
# tinha a bola e lista-se a quem a podia passar (os colegas que a métrica considera). Marca-se a
# quem a passou mesmo e, para cada colega, junta-se o que o descreve: a que distância está, se
# está à frente, quanto perigo traz, se está marcado, e quanto a sombra de um defesa tapa essa
# linha (com o cone pelo tronco e com o cone virado para a bola). Ainda não se tira nenhuma
# conclusão: é a tabela de partida para ver, na fase 8b, o que explica a escolha do passe.
