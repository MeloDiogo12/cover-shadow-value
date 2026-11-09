"""Características candidatas para dizer se um defesa está a marcar (decisão A1, em aberto).

Calcula, para cada defesa em cada frame alinhado, medidas que uma regra de marcação poderia usar.
Os limiares da regra da marcação (D14) vêm de `metrica.parametros`; `esta_a_marcar` aplica-a.

- distância ao atacante mais próximo (e quem é);
- lado da baliza: o defesa está mais perto da própria baliza do que esse atacante?
- persistência: durante quanto tempo seguido (na mesma posse) esse atacante é o mais próximo;
- corpo virado para o atacante e para o portador: ângulo entre a orientação (cada uma das três
  versões de `footy.metrics.orientacao`) e a direção para cada um deles. 0° = de frente.

A entrada é a tabela de trabalho da fase 3 (`footy.transform.skillcorner_pose.tabela_de_trabalho`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.metrics.orientacao import VERSOES, angulo_para, diferenca_angular, todas_as_orientacoes
from footy.transform.skillcorner_pose import FPS_ALINHADO, PASSO_TRACKING
from metrica.parametros import Parametros


def baliza_propria_x(match: dict, equipa: int, periodo: int) -> float:
    """x da linha de golo que a equipa defende nesse período. `home_team_side` diz, por parte,
    para onde ataca a equipa da casa; quem ataca "left_to_right" defende a baliza de x negativo."""
    lado_casa = match["home_team_side"][periodo - 1]
    da_casa = equipa == match["home_team"]["id"]
    ataca_para_x_positivo = (lado_casa == "left_to_right") == da_casa
    meio_campo = match["pitch_length"] / 2
    return -meio_campo if ataca_para_x_positivo else meio_campo


def atacante_mais_proximo(tabela: pd.DataFrame, sem_portador: bool = False) -> pd.DataFrame:
    """Para cada defesa e frame: o atacante mais próximo, a distância e a posição dele.
    Com `sem_portador`, o portador da bola não conta como candidato (D20)."""
    partes = []
    for _, frame in tabela.groupby("frame"):
        defesas, atacantes = frame[frame["defende"]], frame[~frame["defende"]]
        if sem_portador:
            atacantes = atacantes[atacantes["jogador"] != atacantes["portador"]]
        if defesas.empty or atacantes.empty:
            continue
        dx = defesas["x"].to_numpy()[:, None] - atacantes["x"].to_numpy()[None, :]
        dy = defesas["y"].to_numpy()[:, None] - atacantes["y"].to_numpy()[None, :]
        distancias = np.hypot(dx, dy)
        mais_perto = distancias.argmin(axis=1)
        partes.append(pd.DataFrame({
            "frame": defesas["frame"].to_numpy(),
            "jogador": defesas["jogador"].to_numpy(),
            "atacante": atacantes["jogador"].to_numpy()[mais_perto],
            "dist_atacante": distancias[np.arange(len(defesas)), mais_perto],
            "atacante_x": atacantes["x"].to_numpy()[mais_perto],
            "atacante_y": atacantes["y"].to_numpy()[mais_perto],
        }))
    return pd.concat(partes, ignore_index=True)


def vantagem_para_a_baliza(defesas: pd.DataFrame, match: dict) -> pd.Series:
    """Metros que o defesa está mais perto do centro da própria baliza do que o atacante mais próximo
    (positivo = entre o atacante e a baliza; negativo = o atacante está mais perto da baliza)."""
    baliza_x = [baliza_propria_x(match, e, p) for e, p in zip(defesas["equipa"], defesas["periodo"])]
    defesa_a_baliza = np.hypot(defesas["x"] - baliza_x, defesas["y"])
    atacante_a_baliza = np.hypot(defesas["atacante_x"] - baliza_x, defesas["atacante_y"])
    return atacante_a_baliza - defesa_a_baliza


def persistencia(defesas: pd.DataFrame) -> pd.Series:
    """Segundos seguidos, dentro da mesma posse, em que o atacante mais próximo é o mesmo.
    Cada frame recebe a duração do trecho inteiro a que pertence (também os frames seguintes)."""
    d = defesas.sort_values(["jogador", "frame"])
    quebra = (
        (d["jogador"] != d["jogador"].shift())
        | (d["posse_id"] != d["posse_id"].shift())
        | (d["atacante"] != d["atacante"].shift())
        | (d["frame"].diff() != PASSO_TRACKING)
    )
    trecho = quebra.cumsum()
    frames_no_trecho = d.groupby(trecho)["frame"].transform("size")
    return (frames_no_trecho / FPS_ALINHADO).reindex(defesas.index)


def posicao_do_portador(tabela: pd.DataFrame) -> pd.DataFrame:
    portador = tabela[tabela["jogador"] == tabela["portador"]]
    return portador[["frame", "x", "y"]].rename(columns={"x": "portador_x", "y": "portador_y"})


def desvios_de_orientacao(defesas: pd.DataFrame) -> pd.DataFrame:
    """Ângulo (0–180°) entre cada versão da orientação e a direção para o atacante e para o
    portador. NaN quando o defesa não tem pose."""
    orientacoes = todas_as_orientacoes(defesas)
    para_atacante = angulo_para(defesas["x"], defesas["y"], defesas["atacante_x"], defesas["atacante_y"])
    para_portador = angulo_para(defesas["x"], defesas["y"], defesas["portador_x"], defesas["portador_y"])
    colunas = {}
    for nome in VERSOES:
        o = orientacoes[f"orient_{nome}"].to_numpy()
        colunas[f"orient_{nome}"] = o
        colunas[f"desvio_atacante_{nome}"] = diferenca_angular(o, para_atacante.to_numpy())
        colunas[f"desvio_portador_{nome}"] = diferenca_angular(o, para_portador.to_numpy())
    return pd.DataFrame(colunas, index=defesas.index)


def caracteristicas_de_marcacao(tabela: pd.DataFrame, match: dict, sem_portador: bool = False) -> pd.DataFrame:
    """Uma linha por defesa e frame, com todas as características candidatas. Com `sem_portador`,
    o atacante mais próximo (e a distância, o lado da baliza e a persistência que dele dependem)
    ignora o portador da bola (D20): quem o pressiona é avaliado pela sombra que faz."""
    defesas = (
        tabela[tabela["defende"]]
        .merge(atacante_mais_proximo(tabela, sem_portador), on=["frame", "jogador"])
        .merge(posicao_do_portador(tabela), on="frame", how="left")
    )
    defesas["dist_portador"] = np.hypot(defesas["x"] - defesas["portador_x"],
                                        defesas["y"] - defesas["portador_y"])
    defesas["vantagem_baliza"] = vantagem_para_a_baliza(defesas, match)
    defesas["lado_da_baliza"] = defesas["vantagem_baliza"] > 0
    defesas["persistencia_s"] = persistencia(defesas)
    return pd.concat([defesas, desvios_de_orientacao(defesas)], axis=1)


def do_lado_da_baliza(c: pd.DataFrame, p: Parametros) -> pd.Series:
    """Defesa do lado da baliza, com a tolerância de `tolerancia_lado_baliza` m (D33): lado a lado conta.
    Sem a coluna `vantagem_baliza` (tabelas antigas) vale o critério estrito."""
    if "vantagem_baliza" not in c:
        return c["lado_da_baliza"]
    return c["vantagem_baliza"] > -p.tolerancia_lado_baliza


def _entra_na_marcacao(c: pd.DataFrame, p: Parametros) -> pd.Series:
    """Começa a marcar (D14, D33): atacante mais próximo a menos de `distancia_marcacao`, defesa do lado
    da baliza (com tolerância) e esse atacante é o mais próximo há pelo menos `tempo_marcacao` segundos."""
    return (
        (c["dist_atacante"] <= p.distancia_marcacao)
        & do_lado_da_baliza(c, p)
        & (c["persistencia_s"] >= p.tempo_marcacao)
    )


def _pode_continuar(c: pd.DataFrame, p: Parametros) -> pd.Series:
    """Continua a marcar (D29): ainda a menos de `distancia_saida_marcacao` do atacante mais
    próximo e do lado da baliza."""
    return (c["dist_atacante"] <= p.distancia_saida_marcacao) & do_lado_da_baliza(c, p)


def esta_a_marcar(c: pd.DataFrame, p: Parametros) -> pd.Series:
    """Regra da marcação: D14 para começar, D29 para continuar, D28 para excluir quem pressiona.
    Quem marca fica de fora da sombra nesse momento (D5).
    - Pressão (D28): a menos de `distancia_pressao` do portador nunca se está a marcar.
    - Continuidade (D29): depois de começar, marca até se afastar mais de `distancia_saida_marcacao`
      (ou até pressionar o portador, ou perder o lado da baliza, ou se cortar a sequência de frames).
    `c`: características por defesa e frame, com `jogador`, `frame` e `dist_portador`."""
    livre_de_pressao = c["dist_portador"] > p.distancia_pressao
    entra = _entra_na_marcacao(c, p) & livre_de_pressao
    continua = _pode_continuar(c, p) & livre_de_pressao
    d = c.assign(entra=entra, continua=continua).sort_values(["jogador", "frame"])
    # um trecho é uma sequência de frames seguidos do mesmo defesa em que pode continuar a marcar;
    # atravessa a mudança de portador, porque quem marca um atacante continua a marcá-lo depois de um passe
    quebra = (
        (d["jogador"] != d["jogador"].shift()) | (d["frame"].diff() != PASSO_TRACKING)
        | ~d["continua"] | ~d["continua"].shift(fill_value=False)
    )
    marca = d.groupby(quebra.cumsum())["entra"].cummax() & d["continua"]
    return marca.reindex(c.index)


# Em linguagem corrente: para cada defesa, em cada instante de uma posse adversária, procura-se o
# atacante que está mais perto dele e mede-se a distância. Depois vê-se se o defesa está entre esse
# atacante e a própria baliza (mais perto dela do que o atacante), durante quanto tempo seguido é
# esse o atacante mais próximo, e se o corpo do defesa aponta para ele e para quem tem a bola (um
# ângulo pequeno quer dizer que está de frente). `esta_a_marcar` aplica depois a regra: começa a marcar
# quem está perto do atacante (4 m), do lado da baliza (ou lado a lado, até 1 m) e há tempo suficiente; continua a marcar até se
# afastar a mais de 5 m (assim a marcação não liga e desliga a cada passo); e quem está colado ao
# portador (menos de 4 m) está a pressionar, não a marcar. Os limiares vêm de `parametros`.
