"""Fase 9e · volta a gerar só as 9 figuras do artigo corrigidas (texto, eixos e paginação; números e cálculos iguais), mais `legendas.md`.

Da raiz do repositório:
    uv run python projects/cover_shadow/figuras_9e.py
As restantes figuras de `outputs/cover_shadow/artigo/` não se tocam. A animação também não.
"""

from __future__ import annotations

import json
import shutil

import figuras_9d as f9
import figuras_agregados as ag
import figuras_artigo as fa
import figuras_lance as fl
import gerar_candidatos as gc
from orientacao_defesas import acrescentar_a_legendas

from footy.paths import PROJECTS
from footy.viz.theme import apply_theme, load_brand

PASTA, C06, C03 = f9.PASTA, fa.C06, fa.C03


def main() -> None:
    brand = load_brand()
    apply_theme(brand)
    numeros = f9.ler_numeros()
    n = json.loads((PASTA / "numeros_das_figuras.json").read_text(encoding="utf-8"))
    JG = ag.carregar_jogos()
    ag.JG_TITULOS.update({j: JG[j]["titulo"] for j in JG})
    J6, J3, JP = gc.carregar(C06["jogo"]), gc.carregar(C03["jogo"]), gc.carregar(f9.PRESSAO_ALTA["jogo"])
    # 5 e 8 do passo a passo (os outros passos não mudam); os números dos restantes passos ficam os do json
    passos = fl.c06_passo_a_passo(J6, C06["frame"], C06["rotulo"], brand, PASTA, moeda="risco", passos=(5, 8))
    for k, v in passos.items():
        n["c06_passos"][str(k)] = v
    n["c06_corpo_bola"] = fl.c06_corpo_vs_bola(J6, C06["frame"], C06["rotulo"], brand, PASTA, moeda="risco")
    n["c03"] = fl.c03_lance(J3, C03["frame"], C03["rotulo"], brand, PASTA, "Klimala", "Isaias", moeda="risco")
    n["fase_terco"] = f9.figura_fase_e_terco_risco(numeros, brand, PASTA / "agregado_fase_e_terco.png")
    n["c4"] = f9.figura_c4_risco(JG, numeros, brand, PASTA / "agregado_c4_estabilidade.png")
    n["c3"] = ag.figura_c3(JG, brand, PASTA / "agregado_c3_sombra_escolhida.png")
    n["versoes"] = ag.figura_versoes(brand, PASTA / "agregado_versoes.png",
                                     "\nA v3.1 oficial usa a moeda risco; esta figura mantém o xT, a moeda das versões históricas.")
    n["pressao_alta"] = f9.figura_pressao_alta(JP, numeros, brand, PASTA / "novo_pressao_alta_goodwin.png")
    (PASTA / "numeros_das_figuras.json").write_text(json.dumps(n, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (PASTA / "legendas.md").write_text(fa.legendas({**n, "c06_passos": {int(k): v for k, v in n["c06_passos"].items()}}), encoding="utf-8")
    if (PASTA / "numeros_orientacao.json").exists():
        acrescentar_a_legendas(PASTA)
    copia = PROJECTS / "cover_shadow" / "artigo"
    copia.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PASTA / "legendas.md", copia / "legendas.md")
    print("limite do eixo da C4:", ag.limite_inferior_c4(min(r["rho_jogador"] for r in n["c4"])))


if __name__ == "__main__":
    main()


# Em linguagem corrente: refaz só as nove figuras do artigo em que se corrigiram textos, eixos ou paginação, e atualiza as legendas.
