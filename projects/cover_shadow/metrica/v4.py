"""Métrica v4 (D38): ameaça negada pesada pela probabilidade de escolha, separada em interceção e dissuasão.

A v3 trata todas as linhas de passe como igualmente prováveis. A v4 pesa cada linha pela probabilidade de ser
escolhida (o M2 da fase 8b) e separa duas formas de a sombra tirar perigo:
- **interceção**: o passe é feito e a bola não chega (a escolha é a que o portador faria sem sombras);
- **dissuasão**: o portador muda de escolha por causa da sombra.

Para um momento com L linhas e um conjunto C de defesas que projetam sombra, seja, para cada coligação S de C:
- `p_l(S)` = probabilidade de o passe da linha l chegar com as sombras de S (produto de `1 − p_{l,d}`);
- `s_l(S)` = probabilidade de escolha da linha l (softmax de `eta_l + beta * (1 − p_l(S))`);
- `x_l` = perigo da linha (valor de xT).

Perigo esperado `E(S) = Σ_l s_l(S) p_l(S) x_l`; ameaça negada `v(S) = E(∅) − E(S)`;
interceção e dissuasão. Como `v = Σ x (s0·p0 − s·p)` é a variação de um produto, a sua decomposição depende da
ordem em que se mudam os dois fatores (`p0` = 1 sem fundo; ver D39 mais abaixo):
- **ordem 1**: `I1 = Σ s0 (1 − p) x` (a escolha é a sem sombras) e `D1 = Σ (s0 − s) p x` (a mudança de escolha
  avalia-se com os passes já cortados);
- **ordem 2**: `D2 = Σ (s0 − s) x` (a mudança de escolha avalia-se sem interceções) e `I2 = Σ s (1 − p) x`
  (o que se corta nos passes que de facto seriam escolhidos);
- **simétrica** (por omissão): `I = (I1 + I2)/2`, `D = (D1 + D2)/2`, o Shapley entre os dois fatores.
Em todas, `I + D = v` exatamente. O Shapley reparte I e D (cada uma com valor 0 na coligação vazia) pelos
defesas de C, por enumeração exata das coligações; é linear, por isso o da simétrica é a média das duas ordens.

**Fundo (D39):** os defesas fora de C (a marcar, sem pose, guarda-redes) não dão sombra mas podem cortar o passe. Com
`p_fundo_l` = probabilidade de o passe chegar considerando só eles, `p_l(S) = p_fundo_l × p_sombras_l(S)` e `p_l(∅) = p_fundo_l`.
O fundo é igual em todas as coligações, por isso só as sombras de C geram valor; a escolha continua a depender só das
sombras (`1 − p_sombras`), como no modelo ajustado. Sem `p_fundo` (ou com 1 em todas as linhas) é a v4 anterior.

As funções trabalham sobre arrays pequenos (um momento), por isso testam-se sem tracking nem pose.
"""

from __future__ import annotations

from math import factorial
from typing import NamedTuple

import numpy as np

MAX_DEFESAS = 10  # 2^10 coligações por momento; um jogo tem no máximo 10 jogadores de campo a defender


def membros_das_coligacoes(n: int) -> np.ndarray:
    """Matriz (2^n × n) booleana: a linha k diz que defesas pertencem à coligação k (bit i de k = defesa i)."""
    return ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def p_chegar_por_coligacao(p_defesa: np.ndarray) -> np.ndarray:
    """(2^n × L): probabilidade de cada linha chegar com as sombras de cada coligação. `p_defesa`: (L × n), a
    probabilidade de o defesa d chegar à linha l (0 se não a afeta). A coligação vazia dá 1 em todas as linhas."""
    membros = membros_das_coligacoes(p_defesa.shape[1])
    falha = np.where(membros[:, None, :], 1 - p_defesa[None, :, :], 1.0)  # só contam os defesas da coligação
    return falha.prod(axis=2)


def probabilidade_de_escolha(eta: np.ndarray, beta: float, sombra: np.ndarray) -> np.ndarray:
    """Softmax, ao longo das linhas, de `eta + beta * sombra`. `eta`: (L,); `sombra`: (K × L) → (K × L)."""
    pontuacao = eta[None, :] + beta * sombra
    e = np.exp(pontuacao - pontuacao.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


ORDENS = ("ordem1", "ordem2", "simetrica")
ORDEM_PADRAO = "simetrica"  # a escolha final entre as três é do autor (D38)


class Valores(NamedTuple):
    """Por coligação (índice = máscara de bits): perigo esperado e interceção/dissuasão nas duas ordens."""
    E: np.ndarray
    I1: np.ndarray
    D1: np.ndarray
    I2: np.ndarray
    D2: np.ndarray
    p_chegar: np.ndarray  # (2^n × L)
    s: np.ndarray  # (2^n × L)

    def decompor(self, ordem: str = ORDEM_PADRAO) -> tuple[np.ndarray, np.ndarray]:
        """(I, D) por coligação na ordem pedida; a simétrica é a média das duas ordens."""
        if ordem == "ordem1":
            return self.I1, self.D1
        if ordem == "ordem2":
            return self.I2, self.D2
        if ordem == "simetrica":
            return (self.I1 + self.I2) / 2, (self.D1 + self.D2) / 2
        raise ValueError(f"ordem desconhecida: {ordem!r} (use uma de {ORDENS})")


def p_com_fundo(p_sombras: np.ndarray, p_fundo) -> np.ndarray:
    """Probabilidade de chegar com as sombras e o fundo (D39): produto (independentes). `p_fundo`: (L,) ou None (= 1)."""
    return p_sombras if p_fundo is None else p_sombras * np.asarray(p_fundo, float)[None, :]


def valores_por_coligacao(x: np.ndarray, eta: np.ndarray, beta: float, p_defesa: np.ndarray, p_fundo=None,
                          custo=None, p_defesa_escolha=None, p_coligacoes=None) -> Valores:
    """E, I e D (nas duas ordens) para todas as coligações de um momento. `x`: (L,) perigo de cada linha; `eta`: (L,) pontuação da
    escolha sem sombra; `beta`: coeficiente da sombra no modelo de escolha; `p_defesa`: (L × n); `p_fundo`: (L,) probabilidade
    de o passe chegar contando só os defesas de fundo (D39); por omissão 1 (sem fundo).
    D40: `custo` (L,) é o custo de perder a linha: o valor esperado de escolhê-la passa a ser `p·x − (1 − p)·custo` (a moeda
    "risco") em vez de `p·x`; `p_defesa_escolha` (L × n) são as probabilidades dos defesas que alimentam o modelo de escolha
    (a variável com que foi ajustado), se diferirem das de `p_defesa` (que dão o p da linha).
    D42: `p_coligacoes` (2^n × L) substitui o p da linha por coligação (o xPass com a sombra: `σ(base + γ · sombra(S))`); a escolha
    continua a depender das sombras de `p_defesa_escolha` (ou de `p_defesa`)."""
    p_sombras = p_chegar_por_coligacao(p_defesa)
    s = probabilidade_de_escolha(eta, beta, 1 - (p_sombras if p_defesa_escolha is None else p_chegar_por_coligacao(p_defesa_escolha)))
    p = p_com_fundo(p_sombras, p_fundo) if p_coligacoes is None else np.asarray(p_coligacoes, float)
    s0, p0 = s[0], p[0]  # coligação vazia: sem sombras (p0 = p_fundo)
    ev, ev0 = (p * x, p0 * x) if custo is None else (p * x - (1 - p) * custo, p0 * x - (1 - p0) * custo)
    return Valores(E=(s * ev).sum(axis=1),
                   I1=(s0 * (ev0 - ev)).sum(axis=1), D1=((s0 - s) * ev).sum(axis=1),
                   I2=(s * (ev0 - ev)).sum(axis=1), D2=((s0 - s) * ev0).sum(axis=1), p_chegar=p, s=s)


def shapley_exato(valor: np.ndarray) -> np.ndarray:
    """Valor de Shapley de cada jogador, por enumeração exata. `valor[k]`: valor da coligação k (máscara de bits),
    com `valor[0]` = 0 para a soma dar `valor[-1]`."""
    n = int(np.log2(len(valor)))
    mascaras = np.arange(len(valor))
    tamanho = membros_das_coligacoes(n).sum(axis=1)
    peso = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)]) if n else np.array([])
    partes = np.zeros(n)
    for i in range(n):
        sem_i = mascaras[(mascaras >> i) & 1 == 0]
        partes[i] = (peso[tamanho[sem_i]] * (valor[sem_i | (1 << i)] - valor[sem_i])).sum()
    return partes


class Decomposicao(NamedTuple):
    """Interceção e dissuasão do momento (coligação completa) e o Shapley de cada uma, numa das ordens."""
    I: float
    D: float
    shapley_I: np.ndarray
    shapley_D: np.ndarray


class Momento(NamedTuple):
    """Resultado de um momento. Os Shapley estão pela ordem das colunas de `p_defesa`. `I`, `D` e os Shapley são os
    da ordem pedida em `avaliar_momento`; `por_ordem` tem as três."""
    E0: float
    EC: float
    I: float
    D: float
    shapley_I: np.ndarray
    shapley_D: np.ndarray
    s0: np.ndarray  # escolha sem sombras, por linha
    sC: np.ndarray  # escolha com as sombras de todos os defesas de C, por linha
    pC: np.ndarray  # probabilidade de o passe chegar com todos os defesas de C (e o fundo), por linha
    por_ordem: dict
    p0: np.ndarray | None = None  # probabilidade de o passe chegar sem sombras (= p_fundo), por linha

    @property
    def v(self) -> float:
        return self.E0 - self.EC


def avaliar_momento(x, eta, beta: float, p_defesa, ordem: str = ORDEM_PADRAO, p_fundo=None, custo=None,
                    p_defesa_escolha=None, p_coligacoes=None) -> Momento:
    """Ameaça negada do momento (coligação completa) e Shapley de I e de D por defesa, nas três ordens; os campos
    `I`, `D`, `shapley_I` e `shapley_D` são os da `ordem` pedida. `p_fundo`, `custo` e `p_defesa_escolha`: ver
    `valores_por_coligacao` (D39, D40, D42)."""
    x, eta, p_defesa = np.asarray(x, float), np.asarray(eta, float), np.asarray(p_defesa, float)
    v = valores_por_coligacao(x, eta, beta, p_defesa, p_fundo, custo, p_defesa_escolha, p_coligacoes)
    todos = 2**p_defesa.shape[1] - 1
    # o Shapley é linear: calcula-se nas duas ordens e a simétrica sai da média
    base = {}
    for o in ("ordem1", "ordem2"):
        I, D = v.decompor(o)
        base[o] = Decomposicao(float(I[todos]), float(D[todos]), shapley_exato(I), shapley_exato(D))
    a, b = base["ordem1"], base["ordem2"]
    base["simetrica"] = Decomposicao((a.I + b.I) / 2, (a.D + b.D) / 2, (a.shapley_I + b.shapley_I) / 2,
                                     (a.shapley_D + b.shapley_D) / 2)
    d = base[ordem] if ordem in base else v.decompor(ordem)  # decompor levanta o erro de ordem desconhecida
    return Momento(E0=float(v.E[0]), EC=float(v.E[todos]), I=d.I, D=d.D, shapley_I=d.shapley_I, shapley_D=d.shapley_D,
                   s0=v.s[0], sC=v.s[todos], pC=v.p_chegar[todos], por_ordem=base, p0=v.p_chegar[0])


class Total(NamedTuple):
    """Versão leve de `avaliar_momento`: só o perigo esperado e o Shapley da ameaça negada total (I + D)."""
    E0: float
    EC: float
    shapley: np.ndarray

    @property
    def v(self) -> float:
        return self.E0 - self.EC


def avaliar_total(x, eta, beta: float, p_defesa, p_fundo=None, custo=None, p_defesa_escolha=None) -> Total:
    """Ameaça negada total e o seu Shapley por defesa, sem separar I de D. Como o Shapley é linear e I + D = v em todas
    as coligações, o resultado coincide com `shapley_I + shapley_D` de `avaliar_momento` (em qualquer ordem). Serve as
    corridas com muitos conjuntos de coeficientes (incerteza, D38/fase 8d), em que só interessa o total."""
    x, eta, p_defesa = np.asarray(x, float), np.asarray(eta, float), np.asarray(p_defesa, float)
    E = valores_por_coligacao(x, eta, beta, p_defesa, p_fundo, custo, p_defesa_escolha).E
    return Total(E0=float(E[0]), EC=float(E[-1]), shapley=shapley_exato(E[0] - E))


# Em linguagem corrente: sem sombras, o portador escolhe entre os colegas com certas probabilidades e cada
# passe chega com a probabilidade que o fundo (quem marca, quem não tem pose, o guarda-redes) deixa.
# Com as sombras, duas coisas mudam: alguns passes já não chegam (interceção) e o portador
# passa a preferir outros colegas (dissuasão). O perigo que se perde é a soma das duas coisas. Para saber quanto
# cabe a cada defesa, experimentam-se todas as combinações de defesas presentes e mede-se, em média, o que cada
# um acrescenta quando entra (Shapley), separadamente para a interceção e para a dissuasão. A dissuasão pode
# ser negativa: a sombra de um defesa pode empurrar o portador para uma linha mais perigosa que ficou livre.
# Como o perigo é "probabilidade de escolha × probabilidade de chegar × valor", a divisão entre as duas partes
# depende de qual dos dois fatores se muda primeiro. Mudar primeiro a chegada (ordem 1) avalia a mudança de
# escolha com os passes já cortados, o que puxa a dissuasão para baixo; mudar primeiro a escolha (ordem 2) faz
# o contrário. A versão simétrica faz a média das duas, sem favorecer nenhuma.
