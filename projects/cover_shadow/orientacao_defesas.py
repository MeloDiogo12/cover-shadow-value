"""Fases 6d e 6e · orientação dos jogadores da equipa sem bola face à bola (v3, dois jogos, durante as posses).

Mede o ângulo entre a frente do tronco de cada jogador da equipa sem bola com pose (sem guarda-redes) e a direção para o
portador: 0° = virado para a bola, 180° = de costas. Três perguntas, só números:
1. distribuição do ângulo (`distribuicao`);
2. o mesmo por faixa de distância à bola, grupo de posição e marcação (`por_faixa`, `por_grupo`);
3. as linhas em que a sombra muda entre o cone pelo tronco e o cone virado para a bola (`mudancas_de_defesa`);
4. (6e) bola vs frente: o ângulo tronco–frente (frente = perpendicular à linha da própria baliza, para longe dela) e a
   separação bola–frente, para distinguir "virado para a bola" de "virado para a frente" (`bola_vs_frente`).

Da raiz do repositório:
    uv run python projects/cover_shadow/orientacao_defesas.py
"""

from __future__ import annotations

import json

import figuras_agregados as ag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from entradas import exigir
from metrica.xt import sentido_de_ataque

from footy.metrics.orientacao import angulo_para, diferenca_angular
from footy.paths import output_dir
from footy.viz.theme import apply_theme, load_brand

FAIXAS_DISTANCIA = ((0, 10), (10, 20), (20, 30), (30, 1000))  # m, [inf, sup)
ROTULOS_FAIXA = ("0–10 m", "10–20 m", "20–30 m", "30+ m")
CLASSES_ANGULO = ((0, 45), (45, 90), (90, 135), (135, 180.001))  # graus, [inf, sup)
ROTULOS_ANGULO = ("até 45°", "45–90°", "90–135°", "acima de 135°")
FAIXAS_SEPARACAO = ((0, 45), (45, 90), (90, 180.001))  # graus entre a direção da bola e a da frente, [inf, sup)
ROTULOS_SEPARACAO = ("0–45°", "45–90°", "90°+")
SECCAO = "Orientação dos jogadores da equipa sem bola"

# Textos da figura 16 (`figura_bola_vs_frente`), por idioma.
TEXTOS = {
    "pt": {
        "titulo": "Virado para a bola ou virado para a frente?",
        "sub": ("Mediana do ângulo do tronco face ao portador e face à frente (direção para longe da própria baliza), "
                "por separação entre as duas direções · dois jogos"),
        "perto": "mais perto da bola em {pct}\nn = {n}",
        "separacao": "separação {s}",
        "graus": "graus",
        "tronco_portador": "ângulo tronco–portador",
        "tronco_frente": "ângulo tronco–frente",
        "nota": ("Separação = ângulo entre a direção do portador e a da frente (só geometria). Cada observação é um jogador da equipa "
                 "sem bola num instante da posse adversária (5 por segundo)."),
    },
    "en": {
        "titulo": "Facing the ball or facing forward?",
        "sub": ("Median trunk angle to the ball carrier and to the forward direction (away from own goal), "
                "by separation between the two · both matches"),
        "perto": "closer to the ball in {pct}\nn = {n}",
        "separacao": "separation {s}",
        "graus": "degrees",
        "tronco_portador": "trunk–ball carrier angle",
        "tronco_frente": "trunk–forward angle",
        "nota": ("Separation = angle between the direction of the ball carrier and that of the forward direction (geometry only). Each observation is a "
                 "player of the team out of possession at one instant of the opponent's possession (5 per second)."),
    },
}
GRUPOS = {"centrais": ("CB", "LCB", "RCB"), "laterais": ("LB", "RB", "LWB", "RWB"),
          "médios": ("DM", "LDM", "RDM", "CM", "LCM", "RCM", "AM", "LM", "RM"),
          "avançados": ("LW", "RW", "CF", "LF", "RF", "SS")}


def grupo_de_posicao(posicao: str) -> str:
    """Grupo da posição da ficha do jogo; suplentes (`SUB`) não têm posição conhecida."""
    for grupo, siglas in GRUPOS.items():
        if posicao in siglas:
            return grupo
    return "sem posição"


def frente_de_jogo(d: pd.DataFrame, match: dict) -> pd.Series:
    """Direção (0° ou 180°) perpendicular à linha da própria baliza, a apontar para longe dela: o sentido em que a
    equipa sem bola ataca, que é o oposto ao da equipa com bola nesse período (o mesmo `sentido_de_ataque` do pipeline)."""
    sentido_com_bola = pd.Series([sentido_de_ataque(match, e, p) for e, p in zip(d["equipa_com_bola"], d["periodo"])], index=d.index)
    return pd.Series(np.where(sentido_com_bola == 1, 180.0, 0.0), index=d.index)


def defesas_com_angulo(JG: dict) -> pd.DataFrame:
    """Jogadores da equipa sem bola com pose e portador (sem guarda-redes, D25), com o ângulo tronco–portador
    (`angulo`), a distância à bola, o grupo de posição e a marcação, e (6e) o ângulo tronco–frente (`angulo_frente`),
    a separação bola–frente (`separacao`) e `mais_perto_da_bola`. Uma linha por jogador e frame."""
    partes = []
    for jogo, J in JG.items():
        d = J["defesas"]
        d = d[d["desvio_portador_tronco"].notna() & d["dist_portador"].notna()]
        partes.append(d.assign(jogo=jogo, frente=frente_de_jogo(d, J["match"])))
    d = pd.concat(partes, ignore_index=True)
    d["angulo"] = d["desvio_portador_tronco"]
    d["angulo_frente"] = diferenca_angular(d["orient_tronco"], d["frente"])
    para_a_bola = angulo_para(d["x"], d["y"], d["portador_x"], d["portador_y"]).to_numpy()
    d["separacao"] = diferenca_angular(para_a_bola, d["frente"])
    d["mais_perto_da_bola"] = d["angulo"] < d["angulo_frente"]
    d["faixa_separacao"] = pd.cut(d["separacao"], [f[0] for f in FAIXAS_SEPARACAO] + [FAIXAS_SEPARACAO[-1][1]],
                                  right=False, labels=ROTULOS_SEPARACAO)
    d["faixa"] = pd.cut(d["dist_portador"], [f[0] for f in FAIXAS_DISTANCIA] + [FAIXAS_DISTANCIA[-1][1]],
                        right=False, labels=ROTULOS_FAIXA)
    d["grupo"] = d["posicao"].map(grupo_de_posicao)
    return d


def resumo(angulo: pd.Series) -> dict:
    """Mediana e percentagem até 45°, até 90° e acima de 135° (n = número de observações)."""
    return {"n": len(angulo), "mediana": float(angulo.median()) if len(angulo) else np.nan,
            "pct_ate_45": 100 * float((angulo <= 45).mean()) if len(angulo) else np.nan,
            "pct_ate_90": 100 * float((angulo <= 90).mean()) if len(angulo) else np.nan,
            "pct_acima_135": 100 * float((angulo > 135).mean()) if len(angulo) else np.nan}


def distribuicao(d: pd.DataFrame) -> dict:
    return {"todos": resumo(d["angulo"]), "a_marcar": resumo(d.loc[d["a_marcar"], "angulo"]),
            "sem_marcar": resumo(d.loc[~d["a_marcar"], "angulo"])}


def por_faixa(d: pd.DataFrame) -> pd.DataFrame:
    """Por faixa de distância: resumo e percentagem em cada classe de ângulo (soma 100 %)."""
    linhas = []
    for faixa in ROTULOS_FAIXA:
        a = d.loc[d["faixa"] == faixa, "angulo"]
        classes = {f"classe_{i}": 100 * float(((a >= lo) & (a < hi)).mean()) for i, (lo, hi) in enumerate(CLASSES_ANGULO)}
        linhas.append({"faixa": faixa, **resumo(a), **classes})
    return pd.DataFrame(linhas)


def por_grupo(d: pd.DataFrame) -> pd.DataFrame:
    """Resumo por grupo de posição, faixa de distância e marcação (a marcar / sem marcar)."""
    linhas = []
    for grupo in GRUPOS:
        for faixa in ROTULOS_FAIXA:
            for marca, nome in ((True, "a marcar"), (False, "sem marcar")):
                a = d.loc[(d["grupo"] == grupo) & (d["faixa"] == faixa) & (d["a_marcar"] == marca), "angulo"]
                linhas.append({"grupo": grupo, "faixa": faixa, "marcacao": nome, **resumo(a)})
    return pd.DataFrame(linhas)


def por_grupo_agregado(d: pd.DataFrame) -> pd.DataFrame:
    """Resumo por grupo de posição com todas as faixas de distância e a marcação juntas: n, mediana, % até 45° e % acima de 135°."""
    return pd.DataFrame([{"grupo": g, **resumo(d.loc[d["grupo"] == g, "angulo"])} for g in GRUPOS])


def resumo_bola_frente(g: pd.DataFrame) -> dict:
    """n, mediana do ângulo tronco–portador, mediana do ângulo tronco–frente e % em que o tronco está mais perto da
    direção do portador do que da frente."""
    vazio = len(g) == 0
    return {"n": len(g), "mediana_portador": np.nan if vazio else float(g["angulo"].median()),
            "mediana_frente": np.nan if vazio else float(g["angulo_frente"].median()),
            "pct_mais_perto_da_bola": np.nan if vazio else 100 * float(g["mais_perto_da_bola"].mean())}


def bola_vs_frente(d: pd.DataFrame, por: str | None = None) -> pd.DataFrame:
    """Por faixa de separação bola–frente (e, se `por` for "faixa" ou "grupo", também por faixa de distância ou grupo)."""
    outras = [None] if por is None else list(ROTULOS_FAIXA if por == "faixa" else GRUPOS)
    linhas = []
    for outra in outras:
        base = d if por is None else d[d[por] == outra]
        for rotulo in ROTULOS_SEPARACAO:
            linha = {"separacao": rotulo, **resumo_bola_frente(base[base["faixa_separacao"] == rotulo])}
            linhas.append({por: outra, **linha} if por else linha)
    return pd.DataFrame(linhas)


def mudancas_de_defesa(JG: dict, d: pd.DataFrame, versao: str = "v3") -> pd.DataFrame:
    """Um registo por (linha, defesa) que faz sombra só com um dos cones: `ganha_com_bola` = só com o cone virado para
    a bola, `ganha_com_tronco` = só com o cone pelo tronco. Traz a distância à bola, o ângulo tronco–portador e a
    marcação desse defesa nesse frame, e `valor_xt` da linha."""
    chave = ["frame", "recetor", "defesa"]
    partes = []
    for jogo in JG:
        tronco_l, tronco_a = ag.ler(jogo, "base", versao)
        _, bola_a = ag.ler(jogo, "bola", versao)
        t, b = tronco_a[chave].assign(tronco=True), bola_a[chave].assign(bola=True)
        m = t.merge(b, on=chave, how="outer")
        m = m[m["tronco"].isna() | m["bola"].isna()].copy()
        m["tipo"] = np.where(m["bola"].notna(), "só com o cone para a bola", "só com o cone pelo tronco")
        valores = tronco_l[["frame", "recetor", "valor_xt"]]
        m = m[chave + ["tipo"]].merge(valores, on=["frame", "recetor"], how="left")
        info = d[d["jogo"] == jogo][["frame", "jogador", "dist_portador", "angulo", "a_marcar"]].rename(columns={"jogador": "defesa"})
        partes.append(m.merge(info, on=["frame", "defesa"], how="left").assign(jogo=jogo))
    return pd.concat(partes, ignore_index=True)


def resumo_das_mudancas(m: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for tipo, g in m.groupby("tipo"):
        g = g.dropna(subset=["angulo", "dist_portador"])
        linhas.append({"tipo": tipo, "registos": len(g), "dist_mediana": float(g["dist_portador"].median()),
                       "dist_q1": float(g["dist_portador"].quantile(0.25)), "dist_q3": float(g["dist_portador"].quantile(0.75)),
                       **{k: v for k, v in resumo(g["angulo"]).items() if k != "n"},
                       "pct_a_marcar": 100 * float(g["a_marcar"].mean())})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- figura

def mistura(cor_a: str, cor_b: str, t: float) -> tuple:
    a, b = np.array(plt.matplotlib.colors.to_rgb(cor_a)), np.array(plt.matplotlib.colors.to_rgb(cor_b))
    return tuple(a * (1 - t) + b * t)


def figura_angulo_por_distancia(d: pd.DataFrame, brand, destino) -> dict:
    """Barras 100 % empilhadas: proporção de jogadores da equipa sem bola em cada classe de ângulo tronco–portador, por faixa de distância."""
    cor = brand.colors
    t = por_faixa(d)
    tons = [mistura(cor["background"], cor["secondary"], x) for x in (1.0, 0.7, 0.42, 0.2)]
    tinta = [cor["background"], cor["background"], cor["text"], cor["text"]]
    fig = ag.nova_figura(brand, "Orientação dos jogadores da equipa sem bola face à bola, por distância",
                         "Ângulo entre a frente do tronco e a direção do portador (0° = virado para a bola) · "
                         "com pose, sem guarda-redes, durante as posses · dois jogos")
    ax = fig.add_axes((0.10, 0.20, 0.62, 0.60))
    ag.estilo_eixo(ax, brand, grelha="")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    n_faixas = len(t)
    for i, r in t.iterrows():
        y = n_faixas - 1 - i
        esquerda = 0.0
        for k in range(len(CLASSES_ANGULO)):
            v = r[f"classe_{k}"]
            ax.barh(y, v, left=esquerda, height=0.62, color=tons[k], edgecolor=cor["background"], linewidth=2.5)
            if v >= 5:
                ax.text(esquerda + v / 2, y, f"{ag.pt(v, 0)} %", ha="center", va="center", fontsize=14, fontweight="bold", color=tinta[k])
            esquerda += v
        ax.text(101, y, f"mediana {ag.pt(r['mediana'], 0)}° · n = {r['n']:,}".replace(",", " "), va="center", fontsize=12, color=cor["text_muted"])
    ax.set_yticks(range(n_faixas), list(reversed(t["faixa"])))
    ax.tick_params(axis="y", labelsize=14, colors=cor["text"])
    ax.set_xlim(0, 100)
    x = 0.10
    for k, nome in enumerate(ROTULOS_ANGULO):
        fig.patches.append(plt.Rectangle((x, 0.115), 0.012, 0.022, transform=fig.transFigure, color=tons[k]))
        fig.text(x + 0.017, 0.126, nome, fontsize=13, color=cor["text"], va="center")
        x += 0.15
    ag.nota(fig, brand, "Cada observação é um jogador da equipa sem bola num instante da posse adversária (5 por segundo). "
            "Faixas de distância ao portador; barras a 100 %.", y=0.05)
    fig.savefig(destino, dpi=ag.DPI, facecolor=cor["background"])
    plt.close(fig)
    return t.to_dict("records")


def figura_bola_vs_frente(d: pd.DataFrame, brand, destino, idioma: str = "pt") -> list:
    """Por faixa de separação bola–frente: mediana do ângulo tronco–portador e do tronco–frente, lado a lado, e a % de
    observações em que o tronco está mais perto da direção do portador do que da frente."""
    cor = brand.colors
    pt, pct, mil = ag.formatadores(idioma)
    X = TEXTOS[idioma]
    t = bola_vs_frente(d)
    fig = ag.nova_figura(brand, X["titulo"], X["sub"])
    ax = fig.add_axes((0.17, 0.20, 0.50, 0.60))
    ag.estilo_eixo(ax, brand, grelha="x")
    ax.spines["left"].set_visible(False)
    altura = 0.34
    for i, r in t.iterrows():
        y = len(t) - 1 - i
        for deslocamento, valor, cor_barra in ((altura / 2, r["mediana_portador"], cor["secondary"]),
                                               (-altura / 2, r["mediana_frente"], cor["text_muted"])):
            ax.barh(y + deslocamento, valor, height=altura - 0.03, color=cor_barra)
            ax.text(valor + 1, y + deslocamento, f"{pt(valor, 0)}°", va="center", fontsize=14, fontweight="bold", color=cor["text"])
        ax.text(184, y, X["perto"].format(pct=pct(r["pct_mais_perto_da_bola"], 0), n=mil(r["n"])),
                va="center", fontsize=12.5, color=cor["text_muted"])
    ax.set_yticks(range(len(t)), list(reversed([X["separacao"].format(s=s) for s in t["separacao"]])))
    ax.tick_params(axis="y", labelsize=14, colors=cor["text"])
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 45))
    ax.set_xlabel(X["graus"], fontsize=12, color=cor["text_muted"])
    for x, nome, cor_barra in ((0.17, X["tronco_portador"], cor["secondary"]), (0.39, X["tronco_frente"], cor["text_muted"])):
        fig.patches.append(plt.Rectangle((x, 0.115), 0.012, 0.022, transform=fig.transFigure, color=cor_barra))
        fig.text(x + 0.017, 0.126, nome, fontsize=13, color=cor["text"], va="center")
    ag.nota(fig, brand, X["nota"], y=0.05, largura=200)
    fig.savefig(destino, dpi=ag.DPI, facecolor=cor["background"])
    plt.close(fig)
    return t.to_dict("records")


# ---------------------------------------------------------------- legenda

def legenda_md(n: dict) -> str:
    """Secção de `legendas.md` com os números da figura e das tabelas (sem conclusões)."""
    pt = ag.pt
    D = n["distribuicao"]
    L = [f"## {SECCAO}", "", "### orientacao_angulo_por_distancia.png", ""]
    f = n["faixas"]
    faixas = "; ".join(f"{r['faixa']}: mediana {pt(r['mediana'], 0)}°, até 45° {pt(r['pct_ate_45'])} %, até 90° {pt(r['pct_ate_90'])} %, "
                       f"acima de 135° {pt(r['pct_acima_135'])} % (n = {r['n']})" for r in f)
    L += [(f"Ângulo entre a frente do tronco e a direção do portador, por faixa de distância à bola (dois jogos, v3, jogadores da equipa sem bola "
          f"com pose, sem guarda-redes, durante as posses, 5 observações por segundo). Todos: mediana {pt(D['todos']['mediana'], 0)}°, até 45° {pt(D['todos']['pct_ate_45'])} %, "
          f"até 90° {pt(D['todos']['pct_ate_90'])} %, acima de 135° {pt(D['todos']['pct_acima_135'])} % (n = {D['todos']['n']}). "
          f"A marcar: mediana {pt(D['a_marcar']['mediana'], 0)}° (n = {D['a_marcar']['n']}); sem marcar: mediana "
          f"{pt(D['sem_marcar']['mediana'], 0)}° (n = {D['sem_marcar']['n']}). {faixas}."), "",
          "Números: `orientacao_defesas.py` (`defesas_com_angulo`, `por_faixa`); `notebook 05`, secção 4.1.", ""]
    ga = "; ".join(f"{r['grupo']}: n = {r['n']}, mediana {pt(r['mediana'], 0)}°, até 45° {pt(r['pct_ate_45'])} %, "
                   f"acima de 135° {pt(r['pct_acima_135'])} %" for r in n["grupos_agregado"])
    L += ["### Por grupo de posição (todas as faixas e marcação juntas)", "", f"{ga}.", "",
          "Números: `orientacao_defesas.por_grupo_agregado`; `notebook 05`, secção 4.1.", "",
          "### orientacao_bola_vs_frente.png", ""]
    bf = n["bola_frente"]
    separacoes = "; ".join(f"separação {r['separacao']}: n = {r['n']}, mediana tronco–portador {pt(r['mediana_portador'], 0)}°, "
                           f"tronco–frente {pt(r['mediana_frente'], 0)}°, tronco mais perto da bola em {pt(r['pct_mais_perto_da_bola'])} %"
                           for r in bf)
    L += [(f"Ângulo do tronco face ao portador e face à frente (direção perpendicular à linha da própria baliza, para longe dela), por "
          f"faixa de separação entre as duas direções (dois jogos, v3, jogadores da equipa sem bola com pose, sem guarda-redes, durante as "
          f"posses, 5 observações por segundo). {separacoes}. A separação é só geometria (não depende da pose)."), "",
          ("Números: `orientacao_defesas.bola_vs_frente` (também por faixa de distância e por grupo, em `numeros_orientacao.json`); "
          "`notebook 05`, secção 4.1."), ""]
    return "\n".join(L) + "\n"


def acrescentar_a_legendas(pasta) -> None:
    """Põe a secção da orientação no fim de `legendas.md` (substitui a anterior, se existir)."""
    ficheiro = pasta / "legendas.md"
    n = json.loads((pasta / "numeros_orientacao.json").read_text(encoding="utf-8"))
    texto = exigir(ficheiro, "figuras_9d.py (ou figuras_9e.py), que escreve legendas.md").read_text(encoding="utf-8")
    for titulo in ("## Orientação dos defesas", f"## {SECCAO}"):  # o título antigo (6d) também se substitui
        texto = texto.split(titulo)[0]
    texto = texto.rstrip() + "\n\n"
    ficheiro.write_text(texto + legenda_md(n), encoding="utf-8")


# ---------------------------------------------------------------- execução

def calcular(JG: dict) -> dict:
    d = defesas_com_angulo(JG)
    m = mudancas_de_defesa(JG, d)
    return {"defesas": d, "distribuicao": distribuicao(d), "faixas": por_faixa(d), "grupos": por_grupo(d),
            "grupos_agregado": por_grupo_agregado(d), "bola_frente": bola_vs_frente(d),
            "bola_frente_por_faixa": bola_vs_frente(d, "faixa"), "bola_frente_por_grupo": bola_vs_frente(d, "grupo"),
            "mudancas": m, "resumo_mudancas": resumo_das_mudancas(m)}


def main() -> None:
    brand = load_brand()
    apply_theme(brand)
    pasta = output_dir("cover_shadow") / "artigo"
    pasta.mkdir(parents=True, exist_ok=True)
    JG = ag.carregar_jogos()
    r = calcular(JG)
    figura_angulo_por_distancia(r["defesas"], brand, pasta / "orientacao_angulo_por_distancia.png")
    figura_bola_vs_frente(r["defesas"], brand, pasta / "orientacao_bola_vs_frente.png")
    numeros = {"distribuicao": r["distribuicao"], "faixas": r["faixas"].to_dict("records"),
               "grupos": r["grupos"].to_dict("records"), "grupos_agregado": r["grupos_agregado"].to_dict("records"),
               "bola_frente": r["bola_frente"].to_dict("records"),
               "bola_frente_por_faixa": r["bola_frente_por_faixa"].to_dict("records"),
               "bola_frente_por_grupo": r["bola_frente_por_grupo"].to_dict("records"),
               "mudancas": r["resumo_mudancas"].to_dict("records")}
    (pasta / "numeros_orientacao.json").write_text(json.dumps(numeros, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    acrescentar_a_legendas(pasta)
    print(json.dumps(numeros["distribuicao"], ensure_ascii=False, indent=1))
    print(r["grupos_agregado"].round(1).to_string())
    print(r["bola_frente"].round(1).to_string())
    print(r["bola_frente_por_faixa"].round(1).to_string())
    print(r["bola_frente_por_grupo"].round(1).to_string())
    print(r["faixas"].round(1).to_string())
    print(r["resumo_mudancas"].round(1).to_string())
    print(f"Figura e numeros_orientacao.json em {pasta}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: para cada defesa em cada instante da posse adversária, vê-se se o peito aponta
# para quem tem a bola (0°) ou para o lado oposto (180°). Mostra-se como esse ângulo se reparte consoante a
# distância à bola, a posição e se o defesa está a marcar. Depois olha-se para as linhas de passe em que o
# cone virado para a bola e o cone pelo tronco discordam: a que distância da bola estão, e com que ângulo,
# os defesas que só fazem sombra com um dos dois cones.
