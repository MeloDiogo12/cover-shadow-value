"""Caminhos centrais do projeto, para nenhum script depender de onde é corrido."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"          # dados como chegaram da fonte; nunca editar à mão
DATA_INTERIM = ROOT / "data" / "interim"  # dados limpos/normalizados
DATA_PROCESSED = ROOT / "data" / "processed"  # tabelas finais prontas para gráficos/modelos
OUTPUTS = ROOT / "outputs"
PROJECTS = ROOT / "projects"


def output_dir(project: str) -> Path:
    """Pasta de saída de um projeto (criada se não existir)."""
    path = OUTPUTS / project
    path.mkdir(parents=True, exist_ok=True)
    return path
