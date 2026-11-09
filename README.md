# Cover Shadow Value

A defender who is not touching the ball can still take passes away. Cover Shadow Value measures it: the
cone behind a defender's trunk (taken from body pose, not from the ball) closes the passing lanes that cross it,
and closing a lane has value even without a tackle or an interception. This repository computes the metric on
the SkillCorner open data (A-League 2024/25: tracking, events and body pose) and reproduces the numbers and
figures of the article.

![Cover Shadow Value on one play: who closed what](figures/en/c06_passo8_resumo.png)

Article: (link to be added)

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- About 3 GB of disk for the two games with body pose, plus the tracking of eighteen more games if you
  want the pass-model calibration (see below)
- Windows, macOS or Linux (developed on Windows)

## Install

```
uv sync
uv run pytest -q          # unit tests; no data needed
```

## Data (not included)

All data come from SkillCorner Open Data (MIT licence) and are downloaded on demand into `data/raw/`, which is
git-ignored:

- tracking (10 fps), dynamic events and phases of play: <https://github.com/SkillCorner/opendata>
- body pose (25 fps), only two games (1925299 and 1996435): <https://huggingface.co/datasets/SkillCorner/opendata-bodypose>

```
uv run python -m footy.ingest.skillcorner --jogo 1925299 --tracking --pose     # about 700 MB
uv run python -m footy.ingest.skillcorner --jogo 1996435 --tracking --pose     # about 700 MB
```

The two games with pose are enough for the metric and for most numbers. The calibration of the pass model
(`xpass.py`, `calibracao_fundo.py`) also uses the other eighteen open games, tracking only:

```
for j in 1874553 1886347 1899585 1927964 1953632 1959846 1986691 1996436 2006229 \
         2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461; do
  uv run python -m footy.ingest.skillcorner --jogo $j --tracking
done
```

## Reproduce

Run from the repository root. The scripts are idempotent: what is already computed is not repeated.

### Quick path: the C06 play (about 15 minutes after the download)

```
cd projects/cover_shadow
uv run python preparar_dados.py --jogo 1925299                    # about 2 min per game: aligns tracking and pose
uv run python preparar_dados.py --jogo 1996435
uv run python avaliar_variantes.py --versao v3 --so base bola     # about 2 min per game and variant, in parallel
uv run python avaliar_variantes.py --versao v3_1 --so base bola
uv run python numeros_9c.py                                       # about 3.5 min: loss costs and phase index
uv run python numeros_9d.py --so c06 c03 caso_pressao_alta        # under 1 min
```

This recomputes the C06 play of the article: denied threat 0.0109 (risk currency) out of 0.114 available,
Jelacic at phase index 157. `outputs/cover_shadow/artigo_v31.json` comes out identical to the tracked copy.

### Full path

Order of the whole pipeline (each script's docstring says what it needs and writes). Steps marked * need the
eighteen extra tracking games and take the longest; the pass-model calibration steps (6 and 7) were not rerun
from scratch when this repository was prepared.

1. `preparar_dados.py --jogo <id>` for both games.
2. `avaliar_variantes.py` for `--versao v2`, `--versao v3` and `--versao v3_1` (all variants; about 100 s per
   game and variant, `--trabalhadores 6` runs them in parallel).
3. `escolha.py`, then `modelo_escolha.py` (who the passer picks, and the conditional-logit model of that choice).
4. `calcular_v4.py --jogo <id>` and `avaliar_variantes_v4.py` (and `--fundo`).
5. `calibracao_passe.py`, then `calcular_modelo_passe.py --jogo <id>`.
6. `*calibracao_fundo.py preparar`, then `calibracao_fundo.py calibrar`.
7. `*xpass.py preparar`, `calcular_xpass.py --jogo <id>`, then `avaliacao_xpass.py` (writes `xpass_v4.json`).
8. `reprodutibilidade_v31.py` (recomputes v3.1 from the raw data and checks that the tables match).
9. `numeros_v31.py`, `numeros_9c.py`, `numeros_9d.py`, `orientacao_defesas.py` (all article numbers into
   `outputs/cover_shadow/artigo_v31.json`).
10. Figures: `figuras_9d.py` (Portuguese, writes `legendas.md`), then `figuras_10a.py --idioma en` and
    `--idioma pt --variante escura` / `--idioma en --variante escura` for the other language and the dark
    theme (`--variante clara` is the default). `animar_c06.py` writes the C06 animation (MP4 and one PNG per step).

The notebooks in `projects/cover_shadow/notebooks/` tell the same story step by step, with their outputs.

## Where things end up

| What | Where |
|---|---|
| Article numbers (all of them, one file) | `outputs/cover_shadow/artigo_v31.json` (a copy is tracked) |
| Figures of the article, light theme | `figures/pt/`, `figures/en/` |
| Same figures, dark theme (social media) | `figures/social/pt/`, `figures/social/en/` |
| Intermediate tables | `data/interim/`, `data/processed/` (git-ignored) |
| Generated figures and animation | `outputs/cover_shadow/` (git-ignored) |

Figure numbers are those of the article: 1 `agregado_versoes`, 2 `c06_passo4_cones_pelo_tronco`,
3 `c06_passo5_probabilidade_de_o_passe_chegar`, 4 `c06_passo8_resumo`, 5 `agregado_revisoes`,
6 `agregado_fase_e_terco`, 7 `agregado_c3_sombra_escolhida`, 8 `agregado_c5_tronco_vs_bola`,
9 `c06_corpo_vs_bola`, 10 `agregado_c4_estabilidade`, 11 `novo_c8_escolha_do_passe`, 12 `novo_moedas_por_fase`,
13 `novo_pressao_alta_goodwin`, 14 `novo_xpass_calibracao_e_gamma`, 15 `c03_lance`,
16 `orientacao_bola_vs_frente`. Figure 5 summarises the three rounds of play review, the first by the author and
the following two by coaches external to the work; it is generated from `projects/cover_shadow/referencias/revisao_contagens.csv`
(the sheets with the reviewers' comments are not published).

## Structure

```
config/brand.yaml              colours and fonts (light and dark variants)
src/footy/                     the small library the project needs
  ingest/skillcorner.py        download and read the SkillCorner open data
  transform/skillcorner_pose.py  align tracking and body pose in time
  metrics/orientacao.py        trunk orientation from the pose keypoints
  viz/theme.py, viz/goal.py    theme and drawing helpers; video/compose.py writes MP4s
projects/cover_shadow/
  metrica/                     the metric: cones, arrival index, marking, xT, Shapley split
  *.py                         pipeline, evaluation, numbers, figures (PT and EN) and the C06 animation
  notebooks/                   01 to 12, one per CRISP-DM step, with outputs
  tests/                       unit tests
  referencias/open_xt_12x8_v1.json   open 12x8 xT grid used for lane value
  DECISOES.md                  every method decision and why (in Portuguese)
```

Code comments, docstrings and `DECISOES.md` are in Portuguese; figures and captions exist in Portuguese and English.

## Tests

```
uv run pytest -q
```

## Licence and citation

Code: MIT (see `LICENSE`), Diogo Melo, 2026. The data are SkillCorner's (MIT) and are not included; please credit
SkillCorner wherever you use them. To cite this work use `CITATION.cff` (GitHub shows a "Cite this repository" button).

---

## Resumo em português

O Cover Shadow Value mede o valor do posicionamento defensivo sem bola: o cone que sai das costas do tronco de um
defesa (dado pela *body pose*, não pela bola) fecha as linhas de passe que o atravessam, e isso tem valor mesmo sem
desarme nem interceção. Este repositório calcula a métrica com os dados abertos da SkillCorner (A-League 2024/25) e
reproduz os números e as figuras do artigo.

- **Instalar:** `uv sync` (Python 3.12 e uv). Testes: `uv run pytest -q`.
- **Dados:** não vêm no repositório. Descarregam-se com `uv run python -m footy.ingest.skillcorner --jogo <id> --tracking --pose`
  (jogos 1925299 e 1996435, cerca de 700 MB cada); os outros 18 jogos abertos só com tracking servem para calibrar o modelo do passe.
- **Reproduzir:** a ordem dos comandos está na secção *Reproduce*; os números do artigo ficam em `outputs/cover_shadow/artigo_v31.json`.
- **Figuras:** `figures/pt/` e `figures/en/` (tema claro, artigo) e `figures/social/` (tema escuro, redes sociais).
- **Decisões de método:** `projects/cover_shadow/DECISOES.md`. O código e os comentários estão em português.
