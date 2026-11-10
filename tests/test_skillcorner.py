"""Testes do leitor SkillCorner com frames sintéticos (sem rede, sem ficheiros grandes)."""

import gzip
import json
import zipfile

import pytest

from footy.ingest.skillcorner import (
    descarregar_pose,
    frames,
    jogadores_por_frame,
    qualidade_da_pose,
)


def _frame(numero, jogadores):
    return {"frame": numero, "period": 1, "player_data": jogadores}


def _jogador(pid, detetado, com_pose):
    return {
        "player_id": pid, "x": 1.0, "y": -2.0, "is_detected": detetado,
        "joints": {"nose": {"xyz": [0, 0, 0], "p90_mae_cm": 9.0}} if com_pose else None,
    }


def test_frames_le_gzip_linha_a_linha(tmp_path):
    ficheiro = tmp_path / "amostra.jsonl.gz"
    with gzip.open(ficheiro, "wt", encoding="utf-8") as f:
        f.write(json.dumps(_frame(1, [])) + "\n" + json.dumps(_frame(2, [])) + "\n")
    assert [fr["frame"] for fr in frames(ficheiro)] == [1, 2]


def test_joints_null_conta_como_sem_pose_mesmo_com_posicao():
    tabela = jogadores_por_frame(iter([
        _frame(1, [_jogador(10, True, True), _jogador(11, True, False), _jogador(12, False, False)]),
    ]))
    assert tabela["tem_pose"].tolist() == [True, False, False]
    qualidade = qualidade_da_pose(tabela)
    assert qualidade["jogador_frames"] == 3
    assert qualidade["com_pose"] == 1 / 3
    assert qualidade["detetados"] == 2 / 3


def test_frames_le_o_jsonl_de_dentro_do_zip_e_ignora_macosx(tmp_path):
    ficheiro = tmp_path / "1925299.jsonl.zip"
    with zipfile.ZipFile(ficheiro, "w") as z:
        z.writestr("__MACOSX/._1925299.jsonl", "lixo que não é JSON")
        z.writestr("1925299.jsonl", json.dumps(_frame(5, [])) + "\n" + json.dumps(_frame(10, [])) + "\n")
    assert [fr["frame"] for fr in frames(ficheiro)] == [5, 10]


def test_pose_completa_so_existe_para_dois_jogos():
    with pytest.raises(ValueError):
        descarregar_pose(1886347)


def test_dados_por_descarregar_dizem_que_comando_correr(tmp_path, monkeypatch):
    from footy.ingest import skillcorner as sc
    monkeypatch.setattr(sc, "PASTA", tmp_path)
    with pytest.raises(FileNotFoundError, match="footy.ingest.skillcorner --jogo 1925299 --tracking --pose"):
        sc.ler_match(1925299)
    with pytest.raises(FileNotFoundError, match="--jogo 1874553 --tracking"):
        sc.ler_eventos(1874553)
