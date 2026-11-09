"""Fase 8d · C6: recalcula a v4 a partir dos dados originais e confirma que as tabelas coincidem; corre a suíte de testes.

Guarda em memória as tabelas atuais (fase 3, escolhas, resultados do modelo de escolha e v4), volta a correr cada passo a
partir de `data/raw` (`preparar_dados.py`, `escolha.py`, `modelo_escolha.py`, `calcular_v4.py`, nesta ordem), compara cada
tabela nova com a guardada, corre a suíte de testes e escreve o resultado em `outputs/cover_shadow/reprodutibilidade_v4.json`. Demora ~20 min.
Da raiz do projeto:
    uv run python projects/cover_shadow/reprodutibilidade_v4.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from escolha import PASTA_SAIDA
from preparar_dados import caminho_tabela

from footy.ingest import skillcorner as sc
from footy.paths import PROJECTS, output_dir

PASTA_PROJETO = PROJECTS / "cover_shadow"
DESTINO = output_dir("cover_shadow") / "reprodutibilidade_v4.json"
TOLERANCIA_RELATIVA, TOLERANCIA_ABSOLUTA = 1e-9, 1e-12


def comparar_tabelas(antes: pd.DataFrame, depois: pd.DataFrame) -> dict:
    """Compara duas tabelas pelas colunas comuns: mesmo nº de linhas, valores iguais (numéricos até à tolerância),
    maior diferença absoluta. `so_antes` / `so_depois`: colunas que só existem de um lado."""
    comuns = [c for c in antes.columns if c in depois.columns]
    r = {"linhas_antes": len(antes), "linhas_depois": len(depois), "colunas_comuns": len(comuns),
         "so_antes": [c for c in antes.columns if c not in depois.columns],
         "so_depois": [c for c in depois.columns if c not in antes.columns]}
    if len(antes) != len(depois):
        return {**r, "iguais": False, "maior_diferenca_absoluta": None, "colunas_diferentes": comuns}
    maior, diferentes = 0.0, []
    for c in comuns:
        a, b = antes[c].reset_index(drop=True), depois[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            ok = bool(np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=TOLERANCIA_RELATIVA, atol=TOLERANCIA_ABSOLUTA,
                                  equal_nan=True))
            dif = np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float))) if len(a) else 0.0
            maior = max(maior, float(dif) if np.isfinite(dif) else 0.0)
        else:
            ok = bool(((a == b) | (a.isna() & b.isna())).all())
        if not ok:
            diferentes.append(c)
    return {**r, "iguais": not diferentes, "maior_diferenca_absoluta": maior, "colunas_diferentes": diferentes}


def comparar_json(antes: dict, depois: dict, caminho: str = "") -> list[str]:
    """Caminhos dos valores que diferem entre dois JSON (números com a tolerância das tabelas)."""
    if isinstance(antes, dict) and isinstance(depois, dict):
        return [d for k in antes.keys() | depois.keys()
                for d in comparar_json(antes.get(k), depois.get(k), f"{caminho}/{k}")]
    if isinstance(antes, (int, float)) and isinstance(depois, (int, float)) and not isinstance(antes, bool):
        return [] if np.isclose(antes, depois, rtol=1e-7, atol=1e-10) else [caminho]
    return [] if antes == depois else [caminho]


def correr(script: str, *args: str) -> float:
    inicio = time.time()
    subprocess.run([sys.executable, str(PASTA_PROJETO / script), *args], check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return time.time() - inicio


def correr_testes() -> dict:
    """Corre a suíte completa (`pytest -q` na raiz) e devolve o resumo da última linha (p. ex. "812 passed in 95.02s")."""
    inicio = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=PASTA_PROJETO.parents[1],
                       capture_output=True, text=True, check=False)
    ultima = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    contagens = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|error|errors|skipped|xfailed|warnings?)", ultima)}
    return {"resumo": ultima, "codigo_de_saida": r.returncode, "contagens": contagens, "segundos": time.time() - inicio}


def main() -> None:
    jogos = list(sc.JOGOS_COM_POSE)
    antes = {("tabela", j): pd.read_parquet(caminho_tabela(j)) for j in jogos}
    antes |= {(n, 0): pd.read_parquet(PASTA_SAIDA / f"{n}.parquet") for n in ("escolhas_v3", "passes_escolha_v3")}
    json_antes = json.loads((output_dir("cover_shadow") / "modelo_escolha.json").read_text(encoding="utf-8"))
    for j in jogos:
        antes[("v4", j)] = {t: pd.read_parquet(PASTA_SAIDA / "v4" / f"{j}_v4_{t}.parquet") for t in ("linhas", "momentos", "atribuicoes")}
    tempos = {}
    for j in jogos:
        tempos[f"preparar_dados {j}"] = correr("preparar_dados.py", "--jogo", str(j))
    tempos["escolha"] = correr("escolha.py")
    tempos["modelo_escolha"] = correr("modelo_escolha.py")
    for j in jogos:
        tempos[f"calcular_v4 {j}"] = correr("calcular_v4.py", "--jogo", str(j))

    resultado = {"tempos_s": tempos, "tabelas": {}}
    for j in jogos:
        resultado["tabelas"][f"posses_pose {j}"] = comparar_tabelas(antes[("tabela", j)], pd.read_parquet(caminho_tabela(j)))
    for n in ("escolhas_v3", "passes_escolha_v3"):
        resultado["tabelas"][n] = comparar_tabelas(antes[(n, 0)], pd.read_parquet(PASTA_SAIDA / f"{n}.parquet"))
    json_depois = json.loads((output_dir("cover_shadow") / "modelo_escolha.json").read_text(encoding="utf-8"))
    resultado["modelo_escolha_json_valores_diferentes"] = comparar_json(json_antes, json_depois)
    for j in jogos:
        for t, a in antes[("v4", j)].items():
            resultado["tabelas"][f"v4 {t} {j}"] = comparar_tabelas(a, pd.read_parquet(PASTA_SAIDA / "v4" / f"{j}_v4_{t}.parquet"))
    resultado["todas_iguais"] = all(t["iguais"] for t in resultado["tabelas"].values()) and not resultado["modelo_escolha_json_valores_diferentes"]
    resultado["testes"] = correr_testes()
    DESTINO.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{DESTINO}\ntodas iguais: {resultado['todas_iguais']}")


if __name__ == "__main__":
    main()


# Em linguagem corrente: para ter a certeza de que o resultado se reproduz, este script guarda em memória as tabelas que
# já existem, apaga-as mentalmente ao voltar a fazer todo o caminho desde os dados originais (a tabela do jogo, as
# escolhas de passe, o modelo e a v4) e confronta cada tabela nova com a antiga, coluna a coluna. Se for tudo igual, o
# processo é reprodutível.
