"""A sombra de cobertura de um defesa: cone por trás do corpo (D2, D4, D10, D19).

Geometria (vista de cima, ângulos em graus como em `footy.metrics.orientacao`: 0° = +x, anti-horário):
- o eixo da sombra aponta para o lado oposto ao da frente do tronco (as costas do defesa);
- a sombra começa junto ao corpo com a largura `largura_junto_ao_corpo` (não num ponto) e abre
  `angulo_cone` no total até ao `comprimento_cone`: um trapézio.

O cone é só geométrico (D19). O alcance do defesa não o corta: entra depois, como probabilidade,
em `metrica.chegada`, através da `margem_de_tempo` (quanto tempo o defesa tem de avanço sobre a bola).

Todas as funções recebem arrays numpy e devolvem arrays com a mesma forma (por ponto).
"""

from __future__ import annotations

import numpy as np

from metrica.parametros import Parametros


def eixo_da_sombra(orientacao_graus) -> tuple[np.ndarray, np.ndarray]:
    """Vetor unitário das costas do defesa: oposto ao da frente do tronco."""
    radianos = np.radians(np.asarray(orientacao_graus, dtype=float) + 180)
    return np.cos(radianos), np.sin(radianos)


def dentro_do_cone(px, py, defesa_x, defesa_y, orientacao_graus, p: Parametros) -> np.ndarray:
    """True nos pontos (px, py) que estão dentro do cone atrás do defesa."""
    ex, ey = eixo_da_sombra(orientacao_graus)
    rx, ry = np.asarray(px) - defesa_x, np.asarray(py) - defesa_y
    ao_longo = rx * ex + ry * ey
    de_lado = np.abs(-rx * ey + ry * ex)
    meia_largura = p.largura_junto_ao_corpo / 2 + ao_longo * np.tan(np.radians(p.angulo_cone / 2))
    return (ao_longo >= 0) & (ao_longo <= p.comprimento_cone) & (de_lado <= meia_largura)


def contorno_do_cone(defesa_x, defesa_y, orientacao_graus, p: Parametros) -> np.ndarray:
    """Os 4 cantos do trapézio da sombra (para desenhar): dois junto ao corpo, com a largura de
    `largura_junto_ao_corpo`, e dois ao fim do `comprimento_cone`, já abertos pelo ângulo."""
    ex, ey = eixo_da_sombra(orientacao_graus)
    normal = np.array([-ey, ex])
    eixo = np.array([ex, ey])
    meia_base = p.largura_junto_ao_corpo / 2
    meia_ponta = meia_base + p.comprimento_cone * np.tan(np.radians(p.angulo_cone / 2))
    corpo = np.array([defesa_x, defesa_y], dtype=float)
    return np.array([corpo + meia_base * normal, corpo - meia_base * normal,
                     corpo + p.comprimento_cone * eixo - meia_ponta * normal,
                     corpo + p.comprimento_cone * eixo + meia_ponta * normal])


def margem_de_tempo(px, py, portador_x, portador_y, defesa_x, defesa_y, p: Parametros,
                    defesa_vx=0.0, defesa_vy=0.0) -> np.ndarray:
    """Segundos de avanço do defesa sobre a bola em cada ponto: positivo = chega antes da bola.
    Tempo da bola: distância desde o portador / velocidade da bola (em linha reta, rasteiro).
    Tempo do defesa (modelo de Shaw, que segue Spearman): durante o tempo de reação continua à
    velocidade que trazia, e só depois corre em linha reta a `velocidade_defesa`; ou seja,
    reação + distância desde `posição + velocidade x reação` / velocidade do defesa."""
    tempo_bola = np.hypot(np.asarray(px) - portador_x, np.asarray(py) - portador_y) / p.velocidade_bola
    parte_x = defesa_x + defesa_vx * p.tempo_reacao
    parte_y = defesa_y + defesa_vy * p.tempo_reacao
    tempo_defesa = p.tempo_reacao + np.hypot(np.asarray(px) - parte_x, np.asarray(py) - parte_y) / p.velocidade_defesa
    return tempo_bola - tempo_defesa


# Em linguagem corrente: imagina-se uma lanterna apontada para as costas do defesa. A luz sai com a
# largura do corpo e abre um pouco (60° no total) até 10 m. Tudo o que ela ilumina é sombra, sem
# exceções: se o defesa chega tarde a esse ponto, isso não apaga a sombra, só faz com que valha pouco
# (a conta do tempo está em `margem_de_tempo` e transforma-se em probabilidade em `chegada`). No tempo
# de reação o defesa ainda segue o movimento que trazia; só depois corre direto ao ponto.
