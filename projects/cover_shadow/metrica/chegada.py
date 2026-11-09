"""Probabilidade de o passe chegar e repartição da ameaça negada (D8, D9, D16).

Pressupostos (explícitos, a validar na avaliação):
1. Um defesa com sombra sobre a linha tem, em cada ponto da linha dentro do cone, a probabilidade
   `1 / (1 + exp(-(pi / raiz(3)) * margem / incerteza))` de lá chegar antes da bola (a forma de
   Spearman, 2018, e de Shaw); a da linha é o máximo ao longo desses pontos. O alcance entra só aqui
   (D19): um ponto a que o defesa chega tarde dá uma probabilidade baixa, perto de 0 se muito tarde.
2. Os defesas intercetam de forma **independente**: a probabilidade de o passe chegar é o produto
   das probabilidades de cada um falhar, `prod(1 - p_d)`. Sem sombras, chega com probabilidade 1.
3. Ameaça negada por um conjunto T de defesas: `valor * (1 - prod_{d em T}(1 - p_d))`. O valor de
   Shapley reparte a do conjunto completo; com um só defesa dá `valor * p_d`.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np


def probabilidade_de_chegar_ao_ponto(margem, incerteza: float) -> np.ndarray:
    """Probabilidade de o defesa chegar antes da bola, dado o avanço em segundos (logística).
    O fator pi / raiz(3) é o da implementação de Shaw: faz de `incerteza` o desvio-padrão da curva."""
    return 1 / (1 + np.exp(-np.pi / np.sqrt(3) * np.asarray(margem, dtype=float) / incerteza))


def probabilidade_do_defesa(margem, no_cone, incerteza: float) -> float:
    """Melhor probabilidade do defesa ao longo dos pontos da linha que estão dentro do seu cone
    (0 se o cone não cobre a linha)."""
    if not np.any(no_cone):
        return 0.0
    return float(probabilidade_de_chegar_ao_ponto(np.asarray(margem)[no_cone], incerteza).max())


def probabilidade_de_o_passe_chegar(p_defesas) -> float:
    """Chance de nenhum dos defesas lá chegar primeiro (independentes)."""
    return float(np.prod(1 - np.asarray(p_defesas, dtype=float)))


def ameaca_negada(valor: float, p_defesas) -> float:
    return valor * (1 - probabilidade_de_o_passe_chegar(p_defesas))


def valores_de_shapley(valor: float, p_defesas) -> np.ndarray:
    """Parte da ameaça negada que cabe a cada defesa. Exato: média, sobre todas as ordens de
    entrada, do que cada um acrescenta (o número de defesas por linha é pequeno)."""
    p = np.asarray(p_defesas, dtype=float)
    n = len(p)
    partes = np.zeros(n)
    for i in range(n):
        outros = [j for j in range(n) if j != i]
        for k in range(n):
            peso = factorial(k) * factorial(n - k - 1) / factorial(n)
            for grupo in combinations(outros, k):
                sem = ameaca_negada(valor, p[list(grupo)])
                com = ameaca_negada(valor, p[list(grupo) + [i]])
                partes[i] += peso * (com - sem)
    return partes


# Em linguagem corrente: para cada defesa com a linha à sombra calcula-se a chance de ele apanhar a
# bola (alta se chega bem antes dela, baixa se chega tarde). Se houver vários, supõe-se que cada um tenta sem saber do outro, e o passe só chega se
# todos falharem. A ameaça negada é o perigo da linha vezes a chance de o passe NÃO chegar. Para
# dividir isso entre os defesas usa-se o valor de Shapley: cada um recebe o que acrescenta, em média,
# a todas as ordens possíveis de entrada em cena, e as partes somam sempre o total.
