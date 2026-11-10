"""A mensagem de ficheiro em falta diz que passo correr primeiro."""

import pytest
from entradas import exigir


def test_ficheiro_em_falta_diz_o_passo_a_correr(tmp_path):
    with pytest.raises(FileNotFoundError) as erro:
        exigir(tmp_path / "xpass_dados.npz", "xpass.py preparar")
    assert "xpass_dados.npz" in str(erro.value)
    assert "Corre primeiro: xpass.py preparar" in str(erro.value)


def test_ficheiro_existente_passa_sem_alterar_o_caminho(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"")
    assert exigir(f, "qualquer.py") == f


def test_a_leitura_de_uma_variante_em_falta_aponta_para_avaliar_variantes(tmp_path):
    import avaliacao as av
    with pytest.raises(FileNotFoundError, match="avaliar_variantes.py"):
        av.ler_variante(1925299, "base", tmp_path)
