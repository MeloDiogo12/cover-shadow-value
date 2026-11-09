"""Parâmetros da Cover Shadow Value, todos num só sítio (decisões A1 a A4 e A11 de DECISOES.md).

São os valores de partida. Nenhum está fixo no resto do código: a fase 5 (avaliação) testa-os
por sensibilidade, trocando só este objeto (`dataclasses.replace(PADRAO, angulo_cone=45)`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Parametros:
    # Forma do cone (D2, D4; A2, A3)
    angulo_cone: float = 60.0  # abertura total, em graus (a original)
    comprimento_cone: float = 10.0  # m; até onde não se passa por cima do defesa (D4)

    largura_junto_ao_corpo: float = 2.0  # m; o cone começa com esta largura, não num ponto (D3)

    # Alcance do defesa e da bola (D16, D19, D21; A4). Valores de Spearman (2018), "Beyond Expected
    # Goals", MIT Sloan Sports Analytics Conference, e da implementação de pitch control de Laurie
    # Shaw (Friends of Tracking), que a segue.
    tempo_reacao: float = 0.7  # s até o defesa começar a mexer-se
    velocidade_defesa: float = 5.0  # m/s a que o defesa se desloca
    velocidade_bola: float = 15.0  # m/s, velocidade média da bola
    incerteza_tempo: float = 0.45  # s; desvio-padrão do tempo de interceção (tti_sigma de Shaw)
    velocidade_maxima_tracking: float = 12.0  # m/s; acima disto a velocidade é erro (maxspeed de Shaw)
    pontos_por_linha: int = 120  # amostras ao longo de cada linha de passe

    # Marcação (D5, D14; A1)
    distancia_marcacao: float = 4.0  # m ao atacante mais próximo (D33; era 3 m)
    tolerancia_lado_baliza: float = 1.0  # m; "lado da baliza" também vale lado a lado, até esta margem (D33)
    tempo_marcacao: float = 1.0  # s seguidos a ser o mais próximo
    distancia_saida_marcacao: float = 5.0  # m; quem já marca só deixa de marcar acima disto (D29)
    distancia_pressao: float = 4.0  # m ao portador; abaixo disto o defesa pressiona e nunca marca (D28)

    # Bloqueio pelo corpo (D30, D34): probabilidade 1 até `bloqueio_pleno` m do defesa, a descer
    # linearmente até 0 a `bloqueio_nulo` m. Com `bloqueio_gradual=False` usa-se o raio fixo.
    bloqueio_gradual: bool = True
    raio_bloqueio_corpo: float = 1.0  # m; só para o raio fixo: mais perto do defesa que isto, dentro do cone, fica tapada
    bloqueio_pleno: float = 0.3  # m
    bloqueio_nulo: float = 1.2  # m

    # Zona de cruzamento (D27): linhas para a área só contam se forem passe atrasado ou curto
    comprimento_passe_curto: float = 12.0  # m
    avanco_maximo_atrasado: float = 1.0  # m; ganho máximo em direção à baliza adversária para ser "atrasado"

    # Linhas muito longas (D37, D43): acima deste comprimento a linha não se avalia (quase nunca se joga pelo chão).
    # None = sem limite (a definição v3).
    comprimento_maximo_linha: float | None = None  # m


PADRAO_V3 = Parametros()  # a definição v3 (D33, D34): sem limite de comprimento das linhas

# A v3.1 (D43), a versão do artigo e a omissão: a v3 mais a exclusão das linhas com mais de 35 m (D37).
PADRAO = replace(PADRAO_V3, comprimento_maximo_linha=35.0)

# Moeda do valor de cada linha (D44). A v3.1 mede em «risco» por omissão: valor negado = (p(∅) − p(C)) · (g + c), com g o ganho de xT da
# linha e c o xT do adversário no ponto onde a linha seria cortada. As outras duas moedas continuam disponíveis como análise de
# sensibilidade (`xt`: só o ganho g; `linhas`: adversários ultrapassados). A moeda não altera as sombras nem as probabilidades:
# vive em `avaliacao_risco.moeda_risco` e `avaliacao_v31.moeda_linhas`, sobre as tabelas da v3.1.
MOEDA_PADRAO = "risco"
MOEDAS_DISPONIVEIS = ("risco", "xt", "linhas")

# A definição v2 (D25 a D31), com que se calculou a avaliação em `avaliacao_v2/`: marcação a 3 m sem
# tolerância e bloqueio por raio fixo de 1 m. Serve para reproduzir a v2 e para a comparar com a v3.
PADRAO_V2 = replace(PADRAO_V3, distancia_marcacao=3.0, tolerancia_lado_baliza=0.0, bloqueio_gradual=False)


# Em linguagem corrente: esta é a folha de ajustes da métrica. Diz quão aberto e quão comprido é o
# cone, a que velocidade se supõe que o defesa e a bola se movem, e quando se considera que um
# defesa está a marcar alguém ou só a pressionar o portador, e a partir de que passe se
# deixa de contar como cruzamento. Como estão todos aqui, mudar um número e voltar a calcular chega
# para ver o que acontece à métrica.
