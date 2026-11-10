"""Mensagem clara quando falta o ficheiro de entrada de um passo anterior. Só a mensagem: não calcula nem cria nada."""

from __future__ import annotations

from pathlib import Path


def exigir(caminho, corre_primeiro: str) -> Path:
    """Devolve `caminho` se existir; senão levanta `FileNotFoundError` a dizer que passo correr primeiro (ordem no README)."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"falta {caminho}\nCorre primeiro: {corre_primeiro} (a ordem dos passos está no README)")
    return caminho
