"""Fase 9b · C6: recalcula a v3.1 a partir dos dados originais e confirma que as tabelas coincidem; corre a suíte de testes.

Guarda em memória as tabelas atuais (fase 3, base da v3.1 e adversários ultrapassados), volta a correr cada passo a partir de `data/raw`
(`preparar_dados.py` e a métrica v3.1), compara cada tabela nova com a guardada (sem as sobrescrever, exceto a da fase 3 que
`preparar_dados.py` reescreve), corre a suíte de testes e escreve o resultado em `outputs/cover_shadow/reprodutibilidade_v31.json`.
Da raiz do projeto:
    uv run python projects/cover_shadow/reprodutibilidade_v31.py
"""

from __future__ import annotations

import json
import time

import avaliacao as av
import avaliacao_v31 as a31
import pandas as pd
from metrica.pipeline import calcular_jogo
from preparar_dados import caminho_tabela
from reprodutibilidade_v4 import comparar_tabelas, correr, correr_testes

from footy.ingest import skillcorner as sc
from footy.paths import output_dir

DESTINO = output_dir("cover_shadow") / "reprodutibilidade_v31.json"


def main() -> None:
    jogos = list(sc.JOGOS_COM_POSE)
    pasta = av.PASTAS["v3_1"]
    antes = {("tabela", j): pd.read_parquet(caminho_tabela(j)) for j in jogos}
    antes |= {(t, j): pd.read_parquet(av.caminho_variante(j, "base", t, pasta)) for j in jogos for t in ("linhas", "atribuicoes")}
    antes |= {("ultrapassados", j): a31.ultrapassados_guardados(j) for j in jogos}
    tempos, resultado = {}, {"tabelas": {}}
    for j in jogos:
        tempos[f"preparar_dados {j}"] = correr("preparar_dados.py", "--jogo", str(j))
    for j in jogos:
        inicio = time.time()
        nova = calcular_jogo(j)  # a v3.1: `PADRAO` de `metrica/parametros.py`
        tempos[f"v3.1 base {j}"] = time.time() - inicio
        resultado["tabelas"][f"v3.1 linhas {j}"] = comparar_tabelas(antes[("linhas", j)], nova.linhas)
        resultado["tabelas"][f"v3.1 atribuicoes {j}"] = comparar_tabelas(antes[("atribuicoes", j)], nova.atribuicoes)
        resultado["tabelas"][f"posses_pose {j}"] = comparar_tabelas(antes[("tabela", j)], pd.read_parquet(caminho_tabela(j)))
        resultado["tabelas"][f"ultrapassados {j}"] = comparar_tabelas(antes[("ultrapassados", j)], a31.ultrapassados_do_jogo(j, progresso=False))
    resultado["tempos_s"] = tempos
    resultado["todas_iguais"] = all(t["iguais"] for t in resultado["tabelas"].values())
    resultado["testes"] = correr_testes()
    DESTINO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{DESTINO}\ntodas iguais: {resultado['todas_iguais']}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: para ter a certeza de que o resultado se reproduz, este script guarda em memória as tabelas da v3.1,
# refaz o caminho todo a partir dos dados originais (tabela do jogo, métrica, adversários ultrapassados) e confronta cada tabela nova
# com a antiga, coluna a coluna. No fim corre os testes. Se tudo for igual e os testes passarem, o processo é reprodutível.
