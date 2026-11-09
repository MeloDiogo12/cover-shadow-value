"""Figura 5 (revisão de lances): contagens por ronda, legenda inglesa e geração só com o CSV de contagens."""

import re
import shutil

import matplotlib

matplotlib.use("Agg")  # sem janela: o backend por omissão (Tk) falha em terminais sem ecrã

import figuras_agregados as ag
import legendas_en
import pytest

from footy.viz.theme import apply_theme, load_brand

# sim / em parte / não por ronda e pergunta
ESPERADO = {
    "1.ª ronda · v1": {"sombras": (7, 8, 0), "marcacao": (5, 9, 1)},
    "2.ª ronda · v2": {"sombras": (15, 0, 0), "marcacao": (10, 4, 1)},
    "3.ª ronda · v3": {"sombras": (8, 0, 0), "marcacao": (7, 1, 0)},
}


def trio(c: dict) -> tuple:
    return (c["sim"], c["em parte"], c["não"])


def test_csv_tem_so_contagens():
    cabecalho = ag.CSV_REVISOES.read_text(encoding="utf-8").splitlines()[0]
    assert cabecalho == "ronda,versao,revisor,lances,pergunta,sim,em_parte,nao"


def test_contagens_do_csv():
    contagens = ag.contagens_do_csv()
    for ronda, perguntas in ESPERADO.items():
        for pergunta, valores in perguntas.items():
            assert trio(contagens[ronda][pergunta]) == valores


def test_folhas_e_csv_coincidem():
    if ag.contagens_das_folhas() is None:
        pytest.skip("folhas preenchidas não publicadas")
    assert ag.contagens_das_folhas() == ag.contagens_do_csv()


def test_legenda_en_tem_as_contagens_da_ronda_certa():
    """A legenda EN emparelhava as rondas pela ordem do dicionário (3.ª, 2.ª, 1.ª): as contagens ficavam com a ronda trocada."""
    numeros = {}  # a ordem de inserção das figuras: da 3.ª à 1.ª
    for nome in reversed(list(ESPERADO)):
        numeros[nome] = ag.contagens_do_csv()[nome]
    texto = legendas_en.linhas_revisoes(numeros)
    for (nome, _), rotulo_en in zip(ag.RONDAS, ag.TEXTOS["en"]["rv_rondas"]):
        trecho = next(t for t in texto.split("; ") if t.startswith(rotulo_en + ":"))
        s, m = re.findall(r"(\d+)/(\d+)/(\d+)", trecho)
        assert tuple(map(int, s)) == ESPERADO[nome]["sombras"]
        assert tuple(map(int, m)) == ESPERADO[nome]["marcacao"]


@pytest.mark.parametrize("idioma", ["pt", "en"])
def test_figura_gera_se_so_com_o_csv(tmp_path, idioma):
    sem_folhas = tmp_path / "sem_folhas"
    sem_folhas.mkdir()
    csv = tmp_path / "revisao_contagens.csv"
    shutil.copy(ag.CSV_REVISOES, csv)
    assert ag.contagens_das_folhas(sem_folhas) is None
    contagens = ag.contagens_revisoes(pasta=sem_folhas, csv=csv)
    brand = load_brand(variante="clara")
    apply_theme(brand)
    numeros = ag.figura_revisoes(brand, tmp_path / "f.png", idioma=idioma, contagens=contagens)
    assert (tmp_path / "f.png").stat().st_size > 0
    for ronda, perguntas in ESPERADO.items():
        for pergunta, valores in perguntas.items():
            assert trio(numeros[ronda][pergunta]) == valores
