"""Fora de jogo de um colega como destino de passe (D26).

Um colega está em posição de fora de jogo quando está mais perto da baliza adversária do que a bola
e do que o penúltimo adversário (o guarda-redes conta como adversário), já no meio-campo adversário.
Mede-se com o portador a fazer de bola e a posição de cada jogador nesse frame: pergunta-se "se o
passe saísse agora, o colega estaria em fora de jogo?". Uma linha para ele não é um passe válido.
"""

from __future__ import annotations

import numpy as np


def linha_de_fora_de_jogo(adversarios_x, sentido: int) -> float:
    """x (já virado para quem ataca ir para +x) do penúltimo adversário; -inf se há menos de dois."""
    ao_ataque = np.sort(np.asarray(adversarios_x, dtype=float) * sentido)
    return float(ao_ataque[-2]) if len(ao_ataque) >= 2 else -np.inf


def esta_em_fora_de_jogo(colegas_x, portador_x: float, adversarios_x, sentido: int) -> np.ndarray:
    """True nos colegas mais adiantados do que a bola, o penúltimo adversário e o meio-campo."""
    limite = max(portador_x * sentido, linha_de_fora_de_jogo(adversarios_x, sentido), 0.0)
    return np.asarray(colegas_x, dtype=float) * sentido > limite


# Em linguagem corrente: um colega está adiantado de forma ilegal se, na direção da baliza
# adversária, está à frente da bola e à frente de quase todos os adversários (só pode haver um
# adversário, normalmente o guarda-redes, entre ele e a baliza). Quando isso acontece não faz
# sentido contar uma linha de passe para ele, porque o passe não podia sair.
