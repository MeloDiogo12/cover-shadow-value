# Decisões de método — Cover Shadow Value (CSV)

Registo de cada escolha de definição, com a razão. Quando uma decisão muda, não se apaga:
acrescenta-se a nova entrada com a data e o motivo da mudança.

Estado: **definição v2 fechada no conceito (D1–D34); parâmetros numéricos em aberto** (a testar na
fase de avaliação, por sensibilidade).

---

## Decisões fechadas

### D1 · Objetivo (21/10)
Demonstrar como a métrica pode ser medida. Não tirar conclusões sobre jogadores ou equipas
(o body pose só existe em 2 jogos).

### D2 · O que é uma sombra de cobertura (21–26/10)
A métrica completa exige body pose; nos dados abertos só existe em 2 jogos, o que chega para o objetivo (D1).
Cone por trás do defesa, cuja direção é dada **sobretudo pela orientação corporal**
(vetor peito–costas, a partir de ombros e ancas).
- *Porquê:* é a ideia original e o diferencial face ao trabalho de Cascioli et al. (2025),
  que define a sombra só pela linha de passe e pelo alcance físico, sem o corpo.
- Teste previsto: cone orientado pelo corpo contra cone virado para a bola, nos 2 jogos com body pose.

### D3 · O alcance limita a sombra (26/10)
A sombra conta só onde o cone e o alcance do defesa se sobrepõem:
- partes do cone onde o defesa não chegaria à bola antes de ela passar não contam;
- junto ao defesa, a sombra tem a largura do corpo e do passo (não começa num ponto).
- *Porquê:* o cone sozinho dá valor a linhas que passam longe do defesa (ponta e bordas do
  cone) e ignora linhas que passam mesmo ao lado dele.
- Regra simples e explicável (tempo de reação, velocidade do defesa, velocidade da bola),
  não um modelo físico pesado.

### D4 · Comprimento da sombra = limite do passe por cima (26/10)
O comprimento do cone representa a distância atrás do defesa até onde **não é possível passar
a bola por cima** dele. Para lá dessa distância a bola pode ir pelo ar e a sombra só fechou o
passe rasteiro.
- *Porquê:* dá um significado físico e tático ao parâmetro (os 10 m originais eram uma estimativa).
- A métrica refere-se a **passes rasteiros**.

### D5 · Marcação individual fica de fora (26/10)
Um defesa que está a marcar um adversário **não conta** como sombra nesse momento (excluído
por inteiro, não só a linha para o jogador marcado).
- *Porquê:* a ação voluntária é a marcação; se tapa outra linha é efeito secundário.
- Coerente com Cascioli et al. (2025), o que torna os trabalhos comparáveis.
- A regra exata do que é "estar a marcar" está em aberto (A1).

### D6 · Quando se mede (26/10)
Só nos momentos em que o adversário pode passar: bola em jogo e um jogador com a bola
controlada (eventos `player_possession` da SkillCorner, do início ao fim de cada posse).
Dentro de cada posse, o valor é a **média** ao longo da posse.
- *Porquê:* uma linha fechada durante toda a posse vale mais do que fechada meio segundo,
  e as posses longas não inflacionam o total.

### D7 · Que linhas contam (22/10)
Todas as linhas do portador para cada colega, **incluindo as totalmente fechadas**: o defesa
não precisa de tirar a bola, intercetar nem estar junto ao recetor para ter valor.
- Atenção: a SkillCorner só regista opções de passe com `passing_option_score` > 0,6, por isso
  as linhas fechadas não aparecem nas opções deles.

### D8 · Valor de uma linha cortada (22/10)
*Ameaça negada* = perigo da linha × descida na probabilidade de o passe chegar por causa da sombra.
- *Nota:* a lógica "comparar com o cenário sem a sombra" é a mesma de Cascioli et al. (2025).
  O diferencial está em quem faz sombra e como (D2–D5), não no valor.

### D9 · Repartição entre defesas: valor de Shapley (26/10)
Quando vários defesas fecham a mesma linha, o valor reparte-se pelo valor de Shapley: os
defesas entram um a um, vê-se quanto cada um acrescenta, e faz-se a média sobre todas as ordens.
- *Porquê:* com um só defesa é igual à comparação "com e sem ele"; com vários, é justo e resolve
  o caso que Cascioli et al. deixam em aberto (tirar um defesa de cada vez subvaloriza ambos).

### D10 · O cone é orientado pelo tronco (27/10, era A10)
Orientação = ombros e ancas com o mesmo peso. A cabeça fica como informação extra, sem definir o cone.
- *Porquê (notebook 03, jogo 1925299):* é a versão mais estável (variação mediana de 6,6° entre
  medições a 0,2 s, contra 11,1° nos ombros e 8,2° na cabeça) e concorda com os ombros em 99,2%
  dos casos (até 20°); a cabeça afasta-se do tronco ~18° (mediana), o defesa a observar o jogo.

### D11 · Frequência: 5 medições por segundo (27/10, era A7)
Só nos frames em que pose e tracking coincidem (frame da pose = 2,5 × frame do tracking).
Nas posses sem frame alinhado (toques de primeira: 130 de 1 054 no jogo 1925299) usa-se o frame
alinhado mais próximo (a 0,1 s).

### D12 · Posição de cada jogador: a do corpo quando há pose (27/10, era A11)
Com pose, usa-se a posição do ficheiro da pose (coincide com o corpo: 0,08 m de mediana face ao
centro das ancas); sem pose, a do tracking.
- *Porquê:* a posição do tracking pode afastar-se do corpo quando o jogador não é bem visto
  (em 49% dos casos "não detetado no tracking mas com pose", fica a mais de 3 m do corpo).

### D13 · Defesas sem pose: decisão linha a linha (27/10, era A9)
Uma linha de passe só é avaliada se todos os defesas que a podem afetar têm pose; caso contrário
fica de fora e é contada.
- *Porquê:* exigir pose a todos os defesas deixa só 9,4% dos frames; os defesas sem pose são
  sobretudo os longe da bola, que não teriam impacto de sombra nessa linha.
- Por definir na fase 4: o que é "poder afetar" uma linha (distância à linha face ao alcance e ao
  comprimento do cone).

### D14 · Marcação: distância + lado da baliza + tempo (27/10, ponto de partida de A1)
Está a marcar o defesa que está perto do atacante mais próximo, do lado da baliza, durante pelo
menos um tempo mínimo. Não se usa o corpo.
- *Porquê:* a persistência sozinha quase não varia com a distância (2,0–2,6 s); o corpo raramente
  aponta para o atacante (31% até 45°) porque quem marca se posiciona para ver bola e homem.
- Limiares a testar na avaliação: distância 2 ou 3 m; tempo mínimo ~1 s. Validar nos exemplos (C2).

### D15 · Perigo de cada linha: ganho de xT numa grelha pública (27/10, era A5)
Valor de uma linha = xT do destino − xT da posição do portador, só a parte positiva (grelha
pública de Karun Singh). O `xthreat` da SkillCorner fica como comparação na avaliação, onde existe.
- *Porquê:* transparente e citável; não depende dos defesas (o perigo não muda quando se retira
  o defesa da sombra); fechar um passe sem ganho não vale nada. Limitação: ignora o contexto e
  vem da Premier League.

### D16 · Probabilidade de o passe chegar: regra física (27/10, era A6)
Para cada defesa cuja sombra cobre a linha: probabilidade de chegar a um ponto da linha antes da
bola (tempo da bola vs. tempo de reação + deslocação do defesa, com o alcance de D3). Sem sombra,
a mesma conta sem esse defesa. Verificação com passes reais na avaliação (C3).
- *Porquê:* transparente, sem treino e coerente com D3; os passes bloqueados não são observados,
  por isso um modelo treinado quase não vê o efeito, e 2 jogos dão poucos passes.
- Não se estima a probabilidade de o portador escolher cada passe (limitação a declarar).

### D17 · Apresentação: por lance e por jogador (27/10, era A8)
Por lance: ameaça negada por cada defesa no momento. Por jogador: soma das médias por posse (xT)
e % da ameaça disponível negada; comparação sempre dentro da posição. Com 2 jogos, as tabelas por
jogador são ilustrativas, não rankings.

### D18 · Quem pode afetar uma linha (27/10, completa D13)
Um defesa pode afetar uma linha se estiver a uma distância dela menor do que o comprimento do
cone mais o alcance. Regra independente da orientação (é a pose que pode faltar): mais longe do que
isso, rode o corpo como rodar, a sombra não chega à linha.

### D19 · Revisão de D3: o alcance passa a ser gradual (27/10)
O cone fica só com a parte geométrica (orientação do tronco, ângulo, comprimento e largura junto ao
corpo). O alcance deixa de cortar a sombra e passa a ser apenas a probabilidade de D16: um defesa
cuja sombra toca a linha mas que chegaria tarde recebe uma probabilidade baixa.
- *Porquê:* com o corte de D3 e a probabilidade de D16 juntos, o alcance contava duas vezes e a
  probabilidade ficava sempre entre 0,5 e 1 dentro da sombra (no jogo 1925299, 50,5% da ameaça
  disponível aparecia como negada). A ideia de D3 ("o alcance limita a sombra") mantém-se, de forma
  gradual. É também como Cascioli et al. (2025) tratam o alcance.

### D20 · O portador não entra na regra da marcação (27/10, completa D14)
Na regra de marcação, o "atacante mais próximo" exclui o portador da bola. Quem pressiona o
portador é avaliado pela sombra que faz.
- *Porquê:* pressionar não é marcar; a pressão com corrida curva para tapar uma linha de passe é
  um caso clássico de sombra de cobertura.

### D21 · Valores de partida dos parâmetros de tempo (27/10, A4)
Reação do defesa 0,7 s e velocidade 5 m/s (Spearman, 2018); velocidade média da bola 15 m/s e
incerteza do tempo de interceção 0,45 s (implementação de pitch control de Shaw, Friends of
Tracking, que segue Spearman). Testados por sensibilidade na fase 5.
- *Porquê:* valores de referência citáveis, em vez de escolhas próprias.

### D22 · Interpretações aceites na implementação (27/10)
- Grelha de xT: a pública de 12 × 8 de Karun Singh (`referencias/open_xt_12x8_v1.json`).
- D18: "alcance" = comprimento do cone + metade da largura junto ao corpo (limite geométrico).
- Vários defesas na mesma linha: probabilidades combinadas como independentes, prod(1 − p).
- Média por posse (D6): sobre os frames em que o defesa está em campo, com zero quando não nega
  nada ou está a marcar.

### D23 · Tempo de interceção segue o modelo de Shaw (27/10, completa D16 e D21)
Valores de D21 confirmados no código de Shaw (`Metrica_PitchControl.py`, LaurieOnTracking):
5 m/s, 0,7 s, `tti_sigma` = 0,45 s, bola a 15 m/s; probabilidade
1/(1 + exp(−π/√3 · margem/σ)). Durante o tempo de reação o defesa continua à velocidade que
trazia (posição + velocidade × 0,7 s) e só depois corre em linha reta a 5 m/s.
- Velocidade calculada a partir das posições (~0,4 s). Como em Shaw (`Metrica_Velocities.py`,
  `maxspeed = 12`), velocidades acima de 12 m/s são tratadas como erro do tracking.
- *Porquê:* com 0,7 s de reação, tratar o defesa como parado atrasava quem já corria para a linha.

### D24 · Revisão por treinador, primeira ronda (28/10)
15 lances revistos pelo Diogo (C1, C2): sombras 7 sim / 8 em parte / 0 não; marcação 5 sim /
9 em parte / 1 não. Os 5 lances com mais ameaça negada eram quase todos cruzamentos, onde a métrica
faz menos sentido. As decisões D25–D30 resultam desta revisão (iteração CRISP-DM: volta à modelação).

### D25 · Guarda-redes excluído (28/10)
O guarda-redes não projeta sombra nem é classificado como a marcar.

### D26 · Linhas para jogadores em fora de jogo não contam (28/10)
Um colega em posição de fora de jogo no momento não é uma linha de passe válida (não se avalia).

### D27 · Cruzamentos (28/10)
Zona de cruzamento: portador num corredor lateral do terço ofensivo (etiquetas da SkillCorner nos
eventos de posse). Nessa zona, uma linha para dentro da área só conta se for um passe rasteiro
plausível: passe atrasado (para trás ou na horizontal) ou curto (até ~12 m). As restantes contam
como cruzamento: não se avaliam e ficam contadas como excluídas. Linhas de apoio fora da área contam
normalmente. Bloquear o próprio cruzamento é outra ação, fora da CSV (limitação e trabalho futuro).
- Limiares a testar na fase 5: comprimento (~12 m) e o que conta como "atrasado".

### D28 · Pressão não é marcação (28/10, completa D20)
Um defesa perto do portador (distância de pressão, a definir como parâmetro) está a pressionar e
nunca é classificado como a marcar; é avaliado pela sombra que faz.

### D29 · Continuidade da marcação (28/10, completa D14)
Quem começa a marcar (≤ 3 m, lado da baliza, tempo mínimo) continua a marcar até se afastar mais
do que uma distância maior (~5 m). Evita que a marcação ligue e desligue a cada passo.

### D30 · Bloqueio pelo corpo (28/10, completa D19)
Se a linha passa a menos de ~1 m de um defesa dentro do seu cone, está tapada por ele, sem contar o
tempo de reação. Quanto mais perto do portador, maior o ângulo tapado.
- *Porquê:* com 0,7 s de reação, um defesa colado ao portador nunca "chegava a tempo", embora a bola
  lhe bata no corpo (comentário do L10).

### D31 · Interpretações aceites na implementação de D25–D30 (28/10)
- D25: o guarda-redes da equipa que defende sai da tabela de defesas (não projeta sombra nem marca). O guarda-redes
  da equipa com a bola continua a ser um destino possível (passe atrás).
- D26: fora de jogo medido "se o passe saísse agora": colega mais perto da baliza adversária do que o portador
  (a fazer de bola), do que o penúltimo adversário (guarda-redes incluído) e do que o meio-campo.
- D27: corredor e terço são os do **início** da posse (etiquetas `channel_start` e `third_start`); zona =
  `wide_left` ou `wide_right` no `attacking_third`. "Dentro da área" = grande área (16,5 m desde a linha de golo,
  ±20,16 m). "Atrasado" = avanço de no máximo 1 m em direção à baliza adversária; "curto" = até 12 m.
  Ordem dos motivos de exclusão quando coincidem: fora de jogo, cruzamento, sem pose.
- D28: distância de pressão de partida = 4 m ao portador.
- D29: a continuação não exige o mesmo atacante (basta o mais próximo estar a ≤ 5 m, do lado da baliza) e atravessa
  a mudança de portador; cai se o defesa passar a pressionar, perder o lado da baliza ou faltar um frame alinhado.
- D30: a bola bate no corpo: probabilidade do defesa = 1 nessa linha (sem passar pela curva logística).
- Todos estes valores estão em `parametros.py` e entram na sensibilidade da fase 5 (v2), incluindo valores que
  desligam cada regra.

### D32 · Cruzamentos: manter D27 e mostrar o valor por zona e fase (28/10)
A regra D27 fica como está: nos lances de cruzamento revistos (V01, V07) as sombras fazem sentido;
o que pesa é o valor alto das linhas perto da área (xT). Os resultados passam a ser apresentados
também por zona do campo e por fase defensiva (fases de jogo da SkillCorner: bloco alto, médio,
baixo…). O valor mais interpretável é o da pressão à 1.ª e 2.ª fase de construção adversária.
- Ideia para o artigo / trabalho futuro: dividir a CSV em métricas mais específicas por fase.
- *Porquê (verificação de 28/10):* 96 dos 100 frames com mais ameaça negada são no terço ofensivo,
  42 com o portador no meio-espaço esquerdo; a D27 (corredor lateral + terço ofensivo) exclui ~1% das linhas.

### D33 · Limiares da marcação ajustados (28/10, A1)
Entrada a ≤ 4 m (era 3 m); "lado da baliza" com tolerância de 1 m (lado a lado também conta);
saída acima de 5 m (D29). Testados na sensibilidade e confirmados na revisão C2.
- *Porquê:* na 2.ª revisão, 4 marcações não detetadas: duas a 3,1–3,9 m (entrada a 3 m), duas
  ligeiramente ao lado do atacante (−0,10 m e −0,68 m no lado da baliza).
- Cautela: ajuste feito a partir de 4 casos.
- Verificação (29/10, notebook 05, secção 1.3): V06 a 3,36 m; V12 e V15 a −0,10 m e −0,68 m no lado da baliza; no V11 o
  defesa já era classificado como a marcar no frame do lance (2,99 m), mas só em 5 de 20 frames da janela de ±3 s, com a
  distância a oscilar entre 2,7 e 3,1 m. Com a v3, os quatro são classificados como a marcar no frame do lance.

### D34 · Bloqueio pelo corpo gradual (28/10, substitui o raio fixo de D30)
A probabilidade de bloqueio pelo corpo é ~1 com a linha a 0,3 m do defesa e desce até 0 a 1,2 m
(dentro do cone), sem contar o tempo de reação (`bloqueio_gradual` em `parametros.py`).
- *Porquê (sensibilidade v2, 28/10):* o raio é a regra que mais pesa (ameaça negada de 28,6% sem
  bloqueio a 62,4% com 1,5 m, jogo 1925299). O gradual dá praticamente o mesmo que o raio de 0,7 m
  de Cascioli et al. (50,7% / 43,5% contra 51,4% / 43,7%), sem degrau artificial. Com qualquer
  bloqueio, nas linhas com probabilidade de chegar < 0,5 os passes tentados chegam em ~58–65%.
- Limitação a declarar: só se observam passes tentados (o passador só arrisca uma linha "tapada"
  quando vê que dá), por isso esta verificação não basta para calibrar o bloqueio.
- Consequência para a apresentação: a % negada mostra-se com o intervalo da sensibilidade e as
  leituras assentam em comparações relativas (jogadores da mesma posição, fases).

### D35 · Correção da versão "cone virado para a bola" (C5) (3/11, interno)
`pipeline.virar_para_a_bola` devolvia a Série de `angulo_para` com índice 0..n-1, mas as defesas já vêm
filtradas (sem guarda-redes) com índice não contíguo: `.where` e a atribuição alinhavam por índice e cada defesa
recebia a orientação de outra linha (e muitas ficavam NaN). Corrigido com `.set_axis(defesas.index)`; teste com
índice não contíguo em `tests/test_pipeline.py`. A métrica (cone pelo tronco) não é afetada.
- Recalculadas as variantes `bola` da v2 e da v3. A v1 não se recalcula (definição antiga, D25–D30 não aplicados),
  por isso os números da v1 do C5 ficam com o erro e não devem ser usados.
- Os ficheiros antigos estão em `data/processed/cover_shadow/_antes_fix_bola/`; figuras antigas em
  `outputs/cover_shadow/artigo_antes_fix_bola/`.

### D36 · Volta à modelação: métrica v4 com a escolha do passe (4/11, em curso)
A v3 trata todas as linhas como igualmente prováveis e sobrevaloriza linhas longas que quase nunca se jogam pelo chão (no C06, a
linha para o Faisal tem 44,3 m; nos dois jogos, linhas ≥ 30 m dão ~28 % da ameaça negada e < 9 % dos passes reais, a maioria pelo ar).
Nova volta do CRISP-DM para uma versão 4:
- **8a** montar e descrever a tabela de escolhas (um passe × opção por linha), sem modelo (`escolha.py`, `06_escolha_dados.ipynb`);
- **8b** logit condicional ("se passar, para quem?") com variáveis da linha e a sombra; comparar sem sombra / sombra virada para a bola /
  sombra pelo tronco, com validação entre jogos (C8);
- **8c** métrica v4 = ameaça negada por interceção (pesada pela probabilidade de escolha) + por dissuasão (mudança da escolha), com
  Shapley; regra para passes longos;
- **8d** reavaliação.
- *Estado:* em curso. Nada na definição v3 muda até a 8c ser validada.
- *8a, opções de leitura ainda por decidir:* limite do intervalo entre o frame da escolha e o passe (proposta ≤ 0,4 s; observado ≤ 0,1 s);
  definição de "rasteiro" (`high_pass` só existe nos passes completos); passes com opções sem pose (só conjuntos completos ou manter e marcar);
  limite de comprimento dos passes rasteiros.

### D37 · Definição dos passes e das linhas para o modelo de escolha; critério C8 (4/11, fase 8b, em curso)
Decisões do Diogo:
- **Passe rasteiro** = `high_pass` diferente de verdadeiro **e** comprimento ≤ 35 m. Os passes falhados (sem `high_pass`) entram se tiverem até 35 m.
- **Limite de 35 m** aplica-se aos passes do modelo, às linhas do conjunto de escolhas e, na fase 8c, à métrica v4. A métrica v3 não se altera.
- **Intervalo** entre o frame da escolha e o passe ≤ 0,4 s (observado: ≤ 0,1 s, por isso não corta passes).
- **Linhas sem pose:** a opção A (só conjuntos completos) deixa 628 passes (242 + 386), menos que os ~800 de referência, por isso fica a **opção B**:
  as linhas com algum defesa sem pose ficam, com a sombra calculada só com os defesas que têm pose (valor mínimo) e `linha_sem_pose` = 1.
  Passes do modelo: 1 414 (1925299: 728; 1996435: 686).
- **Critério C8** (segunda volta do CRISP-DM): a sombra pelo tronco ajuda a prever para quem vai o passe, fora da amostra, melhor do que sem sombra e do que
  a sombra virada para a bola. Medida: log-loss por passe em cada jogo de teste (ajuste no outro jogo), com IC 95 % por bootstrap sobre os passes.
  O notebook 01 e o artigo atualizam-se depois.
- *Implementação:* logit condicional com máxima verosimilhança (`modelo_escolha.py`, sem `statsmodels`); M0 sem sombra, M1 com a sombra "bola", M2 com a sombra pelo tronco,
  sobre os mesmos passes e linhas. A sombra com os defesas com pose está em `escolhas_v3.parquet` (`sombra_tronco_pose`, `sombra_bola_pose`).

### D38 · Métrica v4: ameaça negada pesada pela escolha, com interceção e dissuasão (4/11, fase 8c, em curso)
A v3 fica intacta e reproduzível; a v4 é uma versão nova (`metrica/v4.py`, `calcular_v4.py`, tabelas em `data/processed/cover_shadow/v4/`).
Para cada momento alinhado da posse (5/s), com C = defesas que projetam sombra (não marcam, têm pose, podem afetar alguma linha):
- **Linhas:** as da v3, sem fora de jogo (D26), sem as da D27 e sem linhas com mais de 35 m (D37). As linhas com algum defesa sem pose
  entram com a sombra só dos defesas com pose e `linha_sem_pose` = 1 (opção B da D37).
- `p_l(S)`: probabilidade de o passe chegar com as sombras da coligação S (como na v3); `p_l(∅)` = 1.
  `s_l(S)`: probabilidade de escolha do M2 com `sombra_tronco` = 1 − `p_l(S)`; coeficientes ajustados no outro jogo (cross-fitting). `x_l` = `valor_xt`.
- Perigo esperado `E(S) = Σ s_l(S) p_l(S) x_l`; ameaça negada `v(S) = E(∅) − E(S)`.
- Interceção `I(S) = Σ s_l(∅) (p_l(∅) − p_l(S)) x_l`; dissuasão `D(S) = Σ (s_l(∅) − s_l(S)) p_l(S) x_l` (pode ser negativa; não se corta). `I + D = v`.
- **Shapley** sobre os defesas de C, com enumeração exata das coligações (no máximo 10 defesas, 2¹⁰ coligações), separadamente para I e para D.
- Agregação igual à v3 (média por posse, soma por jogador); ameaça disponível = `E(∅)`. Unidade: xT esperado (já não é uma soma de alternativas).
- **Ordem da decomposição (8c2, 4/11):** `v = Σ x (s₀ p₀ − s p)` é a variação de um produto, por isso a divisão em I e D depende de qual
  fator se muda primeiro. A definição acima é a **ordem 1** (`I₁ = Σ s₀ (1 − p) x`, `D₁ = Σ (s₀ − s) p x`): na v4 inicial, D < 0 em 52 % dos
  momentos e total de D negativo nos dois jogos, em parte por esta ordem (a mudança de escolha é avaliada com os passes já cortados). Há duas alternativas:
  **ordem 2** (`D₂ = Σ (s₀ − s) x`, `I₂ = Σ s (1 − p) x`) e a **simétrica** (`I = (I₁ + I₂)/2`, `D = (D₁ + D₂)/2`; Shapley entre os dois fatores,
  coerente com o Shapley entre defesas). Em todas `I + D = v`; o Shapley entre defesas é linear, por isso o da simétrica é a média.
  Implementadas as três em `metrica/v4.py` (`ordem1`, `ordem2`, `simetrica`); a **simétrica é a omissão no código**, mas **a escolha final é do Diogo** (por decidir).
  Números nas três em `08_metrica_v4.ipynb`, secções 3 e 7.
- *Estado:* em curso até a 8c ser validada. Números e testes no notebook `08_metrica_v4.ipynb`.
- **C3 não se recalcula na v4 (8d, 5/11):** as escolhas reais de passe entram no modelo de escolha que pesa as linhas, por isso comparar a v4 com essas
  mesmas escolhas não seria uma validação independente. O C3 fica o da v3 (`05_evaluation.ipynb`). A reavaliação da v4 (C1/C2, C4, C5, C6, C8, descrição
  para o artigo) está em `09_avaliacao_v4.ipynb` e `outputs/cover_shadow/avaliacao_v4.json`, só com números.

### D39 · Defesas de fundo: probabilidade de chegar "de fundo" na v4 (5/11, fase 8e)
Problema: na v4 (D38), `p_l` só tem em conta as sombras; uma linha sem sombra chega com `p = 1`, mesmo com o recetor marcado ou um defesa sem pose em cima
da linha. Na v3 isto não importava (só contavam as linhas tapadas), mas a v4 avalia as **alternativas** para onde o portador vai, e com `p = 1` ficam irrealistas
(exemplos: D01 = 1996435 frame 25338, com E(C) > E(∅) e v = −0,025; e W04).
- **Fundo** F de cada momento = todos os da equipa que defende que não estão em C: os que marcam (D5), os que não têm pose e o guarda-redes (D25). Não recebem
  crédito (D5, D25 mantêm-se), mas podem cortar o passe.
- `p_fundo_l` = probabilidade de o passe da linha l chegar considerando só F, com o mesmo modelo de tempo de interceção (Spearman/Shaw, mesmos parâmetros de
  `metrica/parametros.py`: reação, velocidade, incerteza), com a velocidade de cada defesa (como as sombras) e **sem cone nem bloqueio pelo corpo** (sem orientação,
  qualquer um de F pode chegar a qualquer ponto da linha). Cada defesa chega com a maior probabilidade ao longo dos 120 pontos da linha; independentes entre si, como na v3:
  `p_fundo_l = Π_d (1 − p_{l,d})`. Código em `metrica/fundo.py`.
- `p_l(C) = p_fundo_l × p_sombras_l(C)` e `p_l(∅) = p_fundo_l`. O fundo é igual nos dois cenários, por isso só as sombras de C geram valor. O resto da v4 não muda:
  escolha M2 com cross-fitting, 35 m, ordem simétrica, Shapley. **O modelo de escolha não muda**: `s_l` continua a ser função de `1 − p_sombras` (a variável com que
  o M2 foi ajustado), não de `1 − p_l`. As três ordens de I e D usam `p_0 = p_fundo` (`I₁ = Σ s₀ (p₀ − p) x`, `D₁ = Σ (s₀ − s) p x`; `I₂ = Σ s (p₀ − p) x`, `D₂ = Σ (s₀ − s) p₀ x`); `I + D = v` mantém-se.
- Variante testada: **sem o guarda-redes em F** (`--fundo sem_guarda_redes`). A diferença nos totais está em `08_metrica_v4.ipynb`, secção 8.2.
- Leitura literal de "F = os que não estão em C": C é por momento, por isso um defesa de C que não projeta sombra numa dada linha não conta para o fundo dessa linha
  (mesmo que esteja perto dela). Não se alterou; fica anotado como ponto a rever.
- **Tabelas:** a v4 anterior (D38) mantém-se em `data/processed/cover_shadow/v4_sem_fundo/` (e `avaliacao_v4/`, `outputs/cover_shadow/avaliacao_v4_sem_fundo.json`, `revisao_v4_sem_fundo/`,
  `revisao_lances_v4_sem_fundo.xlsx`); a v4 com fundo está em `v4/`, as variantes essenciais em `avaliacao_v4_fundo/` e a variante sem o guarda-redes em `v4_sem_guarda_redes/`.
  As tabelas de linhas têm a coluna nova `p_fundo` (= `p_l(∅)`). Com F vazio (ou `p_fundo = 1`) a v4 com fundo é igual à anterior (teste e verificação nos dados dos dois jogos: diferença 0).
- **4.ª ronda do treinador** refeita com o fundo (ainda não tinha sido feita): D01–D03 voltam a escolher-se com o fundo, W01–W08 mantêm-se; legenda para treinadores nas imagens e no topo da folha.
- Números (só tabelas, sem leitura) em `08_metrica_v4.ipynb` (secção 8) e `09_avaliacao_v4.ipynb` (secções 1 e 8). O artigo, as figuras do artigo e as animações não se tocaram.
- *Estado:* implementado; a decisão de adotar o fundo como definição da v4 é do Diogo.

### D40 · Modelo completo do passe (calibrado) e moedas (5/11, fase 8f, em curso)
Problema: `p` só contava os defesas com sombra sobre a linha; os que estão à frente da linha, os que mais a cortam, não contavam (C3 da v3: 61,6 % dos passes com p < 0,5 chegaram; na v4 as alternativas pareciam livres).
- **Modelo (`metrica/passe.py`):** para cada linha l e defesa d (todos, guarda-redes incluído) os pontos dividem-se em dentro do cone de d (só se d projeta sombra: tem pose e não marca; `p_cone` da v3 com bloqueio pelo corpo) e fora do cone (ou todos, se d não projeta sombra; `p_fora`, tempo de interceção de Spearman/Shaw, sem orientação nem bloqueio).
  `p_fundo_l = Π_d (1 − p_fora)` sobre todos; `p_sombras_l(C) = Π_{d∈C} (1 − p_cone)`; `p_l(C) = p_fundo · p_sombras`, `p_l(∅) = p_fundo`. Só o cone gera crédito. Difere da D39: o fundo passa a incluir também a parte de fora do cone dos defesas de C.
- **Calibração (`calibracao_passe.py`):** passes do modelo da D37 (1 414; 1 408 com resultado `successful`/`unsuccessful`; os 6 `offside` ficam de fora), linha do recetor visado no último frame alinhado. `unsuccessful` = não chegou (inclui bola fora e passe mal executado: a SkillCorner não os separa).
  Máxima verosimilhança de Bernoulli (p limitado a [1e-4, 1 − 1e-4]) para tempo de reação, velocidade do defesa, σ e fim do bloqueio pelo corpo (`bloqueio_nulo`; `bloqueio_pleno` fica em 0,3 m); variante com a velocidade da bola. Validação entre jogos (ajusta num, testa no outro). Ficheiros: `outputs/cover_shadow/calibracao_passe.json`, `data/processed/cover_shadow/calibracao_passe_previsoes.parquet`.
- **Moedas** (`ev(p) = p·a − (1 − p)·b`): `xt` (a = ganho de xT, b = 0); `linhas` (a = nº de adversários de campo, sem GR, com coordenada longitudinal estritamente entre a do portador e a do recetor; 0 para passes para trás; b = 0); `risco` (a = ganho de xT; b = xT do adversário, grelha invertida, no **ponto da linha onde o defesa com maior probabilidade de corte a corta com mais probabilidade**, contando todos os defesas). I/D simétricas generalizam-se trocando `p·x` por `ev(p)` (I + D = v testado); na v3.1 `negada = Σ (a + b)·p_fundo·(1 − p_sombras)`.
- **Escolha (M2) não muda:** coeficientes do outro jogo e sombras da literatura (a variável com que foi ajustado); só o `p` usa os parâmetros calibrados.
- **v3.1** = v3 (sem fora de jogo, cruzamentos e linhas com algum defesa sem pose, D13) + 35 m; com `p_fora = 0` coincide com a v3 (teste). Nas moedas `linhas` e `risco` a v3.1 mantém as mesmas linhas.
- **Tabelas:** `v3_1/` (três moedas nos nomes dos ficheiros) e `v4_completo_{xt,linhas,risco}/`. `calcular_modelo_passe.py`, `avaliacao_modelo_passe.py`, notebook `10_modelo_passe.ipynb`. A v3, a v4 (D39) e as suas pastas não se alteram.
- *Estado:* implementado e calculado; a escolha da moeda e dos parâmetros a adotar é do Diogo.

### D41 · Calibração do fundo com os 20 jogos abertos e critérios de sucesso fixos (5/11, fase 8g)
Motivo: na 8f o modelo completo, calibrado com 2 jogos (~1 400 passes), não bateu a taxa base fora da amostra num dos jogos e os parâmetros não ficaram identificados
(ex.: velocidade 4,2 vs 9,0 m/s). A parte de **fundo** (interceção sem orientação, sem cone) não precisa de pose: calibra-se com os 20 jogos abertos (só tracking + eventos).
Teste limitado; os critérios fixam-se **antes** de calcular e não se mudam depois de ver resultados. Decisão do Diogo.
- **Validação:** deixando um jogo de fora de cada vez (20 dobras).
  1. *Bate a taxa base:* log-loss do modelo < log-loss de prever sempre a taxa média do treino em ≥ 16 dos 20 jogos, e a melhoria média (por jogo, sem pesos) é positiva.
  2. *Parâmetros estáveis:* em todas as dobras, nenhum parâmetro no limite da busca e cada um dentro de ±20 % do ajuste com todos os jogos.
  3. *Calibração:* ECE (10 grupos por decil da probabilidade prevista) ≤ 0,03 fora da amostra, juntando as previsões das 20 dobras; curva no notebook.
  Os três passam → «calibrável»; caso contrário → «não calibrável com estes dados». Sem procurar outras combinações de parâmetros ou de filtros para fazer passar.
- **Passes:** `player_possession` com `end_type == "pass"`, rasteiros pela D37 (`high_pass` não verdadeiro, ≤ 35 m), `pass_outcome` ∈ {successful, unsuccessful} (sem fora de jogo), no frame de
  tracking do passe (`frame_end`, 10 fps), linha do passador ao recetor visado nesse frame; sem pose. Comprimento = distância passador–recetor no tracking.
- **Resultado principal:** `pass_outcome` (`successful` = chegou). **Secundário (descritivo):** «cortado» = falhou **e** a posse seguinte é da equipa que defendia e começa a < 5 m da linha
  (distância ao segmento passador–recetor, projeção antes do recetor); os falhados que não cumprem ficam de fora; os `successful` contam como não cortados. Os critérios aplicam-se ao principal.
- **Modelo:** `p_fundo` da linha, todos os defesas incluindo o guarda-redes, tempo de interceção de Spearman/Shaw sem orientação nem bloqueio pelo corpo, combinação independente (`Π_d (1 − p_fora)`, 120 pontos
  por linha). Ajustam-se tempo de reação [0,05; 2,0] s, velocidade do defesa [2; 9] m/s e σ [0,05; 2,0] s (limites da 8f); velocidade da bola fixa em 15 m/s. «No limite» = a menos de 0,01 do limite.
  Velocidade dos defesas: a da `metrica.velocidade` (diferença central sobre 0,4 s, tracking a 10 fps; passos > 12 m/s anulados).
- **Referências:** taxa base (média do treino), literatura (0,7 s; 5 m/s; 0,45 s) e logístico simples (comprimento + distância do defesa mais próximo à linha), todos com as mesmas 20 dobras.
- **Fora de âmbito:** v3.1 e v4 não se recalculam com os parâmetros novos; depende do resultado.
- *Estado:* critérios fixados antes de calcular; implementação em `calibracao_fundo.py`, notebook `11_calibracao_20_jogos.ipynb`, `outputs/cover_shadow/calibracao_20_jogos.json`.

### D42 · xPass calibrado + peso da sombra; v4 com as três moedas; critérios K1–K5 fixos (6/11, fase 9a)
Motivo: 8f/8g — o modelo físico de "o passe chegar" não é calibrável (a maioria dos passes falhados não são interceções); na 8g um logístico simples
(comprimento + distância do defesa mais próximo à linha) calibrou bem com os 20 jogos (ECE 0,017). Ideia: `p` passa a ser um **xPass** aprendido com os 20 jogos
(sem pose) e a sombra entra por cima com um peso γ aprendido nos 2 jogos com pose. **Ponto de paragem final:** seja qual for o resultado, o artigo fecha depois desta
fase (com a v4 se passar os critérios; com a v3.1 se não). Critérios fixados **antes** de calcular; não se mudam depois de ver resultados; sem procurar variantes para fazer passar.

**Critérios de aceitação** (decisão do Diogo; veredicto por aplicação mecânica)
- **K1 · xPass base calibrado** (20 jogos, um de fora de cada vez): ECE ≤ 0,03 (10 decis da probabilidade prevista, previsões das 20 dobras juntas) **e** log-loss melhor que a
  taxa base (média do treino) em ≥ 16 dos 20 jogos.
- **K2 · com sombra, nos 2 jogos com pose** (γ ajustado num jogo, testado no outro, e ao contrário): ECE ≤ 0,04 em cada jogo de teste. γ com IC 95 % reportado (não é condição).
- **K3 · dissuasão coerente** (v4, na moeda escolhida): momentos com v < 0 ≤ 10 % e com D < 0 ≤ 25 % (entre os momentos com defesas em C), e D total ≥ 0 nos dois jogos.
- **K4 · a construção conta:** ameaça negada média por momento em bloco alto + médio ≥ 25 % da do bloco baixo (na moeda escolhida).
- **K5 · coerência com o treinador:** no C06 o Jelacic tem o maior valor total; no C03 o Klimala.
- **Regra:** a v4 é aceite se K1, K2, K3, K4 e K5 passarem em pelo menos uma moeda; se passarem em mais do que uma, a escolha da moeda é do Diogo. Senão, o artigo segue com a v3.1.

**Leitura operacional dos critérios (fixada agora)**
- K3 e K4 aplicam-se à v4 xPass de cada moeda (`v4_xpass_{xt,linhas,risco}`), com os dois jogos juntos nas percentagens. «Entre os momentos com defesas em C» = momentos com
  `n_defesas ≥ 1`, para as duas percentagens de K3; v < 0 e D < 0 contam-se com tolerância −1e-12 (como em `avaliacao_modelo_passe.totais`). D total = soma de D, por jogo.
- K4: média de `v` por momento (momentos com alguma linha avaliada, como em `agregacao.por_zona_ou_fase`) nas fases `team_out_of_possession_phase_type` de bloco alto e médio juntos,
  dividida pela do bloco baixo, dois jogos juntos. Os rótulos exatos das fases lêem-se nos dados (só os de bloco alto, médio e baixo contam).
- K5: «valor total» = Shapley total (I + D) de cada defesa no próprio lance (C06 = 1925299 frame 12660; C03 = 1996435 frame 39646), o maior entre os defesas do lance.
- K2: `logit p = base + γ·sombra` com a base (xPass) como *offset* e **sem** termo independente; base do jogo de teste e do jogo de ajuste = xPass ajustado nos outros 19 jogos
  (deixando esse jogo de fora). ECE em 10 decis do jogo de teste (≈ 700 passes). Compara-se também sem sombra (γ = 0) e com a sombra virada para a bola.

**xPass base** (`xpass.py`; passes rasteiros D37 dos 20 jogos, os da 8g, N = 14 584; resultado `pass_outcome`, 1 = chegou)
- Logístico sem penalização, 6 variáveis + termo independente, padronizadas: comprimento; avanço em direção à baliza (`sentido × (x_recetor − x_portador)`); distância mínima de um
  defesa à linha (perpendicular, só defesas cuja projeção cai **dentro** do segmento; sem nenhum, vale o teto); nº de defesas a < 3 m da linha (mesma regra de projeção);
  distância do recetor ao adversário mais próximo; distância do portador ao adversário mais próximo. «Adversários» = todos os da equipa que defende, guarda-redes incluído.
  Distâncias com **teto de 15 m** (escolha feita agora, não testada). Passador e recetor no frame de tracking do passe (10 fps), como na 8g.
- A mesma função geométrica calcula as variáveis para qualquer linha de qualquer momento a partir do tracking (v3.1/v4) e para os passes dos 2 jogos com pose.

**Peso da sombra** (2 jogos com pose; passes do modelo da D37 = os 1 408 da 8f, frame da escolha)
- `sombra` = 1 − `p_sombras` da v3 (tronco, parâmetros da literatura, linhas ≤ 35 m) da linha do recetor visado; versão com a sombra virada para a bola (`sombra_bola_pose`).
  γ por máxima verosimilhança de Bernoulli (offset), IC 95 % por Hessiano numérico; diferença de log-loss tronco − bola com IC 95 % por bootstrap sobre os passes; matriz de correlação
  sombra × variáveis da base.

**Versões** (sem apagar nada; pastas `v3_1_xpass/`, `v4_xpass_{xt,linhas,risco}/`)
- `p_l(C) = σ(base_l + γ·sombra_l(C))` com `sombra_l(C) = 1 − Π_{d∈C}(1 − p_cone_{l,d})` (cone da v3, parâmetros da literatura, bloqueio pelo corpo); `p_l(∅) = σ(base_l)`.
  γ = ajuste com os 2 jogos juntos; base dos 2 jogos = xPass ajustado nos outros 19 (cross-fitting). **Contrafactual «sem sombra» = sombra a 0**; a proximidade dos defesas fica na base.
- v3.1 calibrada: `negada = Σ_l (a_l + b_l)·(p_l(∅) − p_l(C))`, com Shapley exato entre os defesas de C sobre o valor de coligação `Σ_l (a_l + b_l)(p_l(∅) − p_l(S))`
  (já não é linear em S). Mantém as exclusões da v3 (D13: linhas com algum defesa sem pose ficam de fora), sem fora de jogo nem cruzamentos, ≤ 35 m.
- v4: escolha M2 (cross-fitting, `s_l` função de `1 − p_sombras` da literatura, como na D40), I/D simétrica, Shapley exato; moedas `xt`, `linhas`, `risco` (D40). Linhas como na D40 (v4 inclui as sem pose,
  com `linha_sem_pose`). Na moeda `risco`, o ponto de corte que dá o custo `b` vem do modelo físico com os parâmetros **da literatura** (a calibração física não se adota: 8g).
- Nada na v3, v3.1 (D40), v4 (D39) ou v4_completo se altera.

*Estado:* critérios fixados antes de calcular. Código em `xpass.py`, `calcular_xpass.py`, `avaliacao_xpass.py`; notebook `12_xpass_v4.ipynb`; `outputs/cover_shadow/xpass_v4.json`.
- **Resultado (aplicação mecânica, 6/11):** K1 cumpre (ECE 0,0052; xPass bate a taxa base nos 20 jogos). K2 **não cumpre** (ECE 0,0406 no 1996435, limite 0,04; 0,0269 no 1925299); γ (2 jogos, tronco) = −0,374, IC 95 % [−0,955; 0,208];
  a sombra não melhora o log-loss fora da amostra nem pelo tronco nem pela bola. K3 não cumpre em nenhuma moeda (v < 0: 31,5 % xt, 30,7 % linhas, 70,6 % risco; D < 0: 34,8 %, 33,2 %, 79,5 %; D total < 0 nos dois jogos só no risco). K4 cumpre só em `linhas` (razão 0,87; xt 0,07; risco −0,31).
  K5 cumpre nas três, mas o Jelacic (C06) e o Klimala (C03) são os únicos defesas em C nesses lances. **Nenhuma moeda cumpre K1–K5 → a v4 não é aceite; o artigo segue com a v3.1.** Ponto de paragem final cumprido; nada mais se testa.
- Nota: com γ = 0 a interceção da v4 é zero mas a dissuasão não (a escolha M2 continua a depender das sombras); na v3.1 a ameaça negada é zero.

### D43 · Definição oficial: v3.1 (6/11, fase 9b)
Decisão do Diogo, a seguir ao ponto de paragem da D42 (a v4 não foi aceite).
- **v3.1** = v3 (parâmetros da literatura, cone pelo tronco, D19–D34) **+ exclusão das linhas com mais de 35 m** (D37). É a versão do artigo. A v4 (D38–D42) fica como trabalho futuro.
  Em código: `Parametros.comprimento_maximo_linha = 35.0` em `metrica/parametros.py`; `PADRAO` é a v3.1, `PADRAO_V3` (sem limite) e `PADRAO_V2` continuam a reproduzir a v3 e a v2. Uma linha com mais de 35 m
  fica excluída com o motivo `mais_de_35m` (só se não tiver já outro motivo, por esta ordem: fora de jogo, cruzamento, sem pose); nos lances desenhados (figuras e animação) essas linhas saem em vez de aparecerem a tracejado.
- **Nas linhas até 35 m a v3.1 é exatamente a v3** (diferença 0,0 em `p_chegar` e em `ameaca_negada`, nos dois jogos; teste `test_nos_dados_reais_a_v3_1_e_a_v3_nas_linhas_ate_35_m`). Tabelas: `data/processed/cover_shadow/avaliacao_v3_1/` (base, cone virado para a bola e as variantes da C4, mais os limites de 30 e 40 m).
- **O valor da sombra apresenta-se como índice de fecho da linha**, não como probabilidade calibrada de o passe chegar (a calibração não foi possível: D41, D42). A «probabilidade de o passe chegar» da v3.1 é a do modelo de interceção (Spearman/Shaw) com os parâmetros da literatura, só para os defesas que projetam sombra sobre a linha.
- **`v3_1_literatura/` não é a v3.1.** Essa pasta (criada na 9a como referência) guarda a v3.1 do modelo físico completo da D40, em que `p_l(∅) = p_fundo < 1` (todos os defesas, também fora do cone) e a ameaça negada é `Σ (a + b) · p_fundo · (1 − p_sombras)`. A v3.1 oficial tem `p_l(∅) = 1` (como a v3). Nos totais do jogo: ver `artigo_v31.json`, secção `v3_1_literatura_vs_definicao`. A pasta mantém-se (referência da 9a) e não entra no artigo.
- **Segunda leitura: moeda «linhas quebradas»** (`avaliacao_v31.py`): valor de cada linha = nº de adversários de campo (sem GR) ultrapassados pelo passe (a definição da 8f, `metrica.passe.adversarios_ultrapassados`);
  `negada_linhas = Σ linhas_l · (1 − p_l(C))` (com `p_l(∅) = 1`), mesmo Shapley (linear no valor da linha) e mesma agregação (média por posse, soma por jogador). Moeda principal do artigo: xT.
- **Momento da decisão** (verificação, sem alterar a definição): agregação alternativa que só conta os momentos alinhados dos últimos 1 s e 2 s antes do fim da posse individual (`player_possession.frame_end`; frames de tracking a 10 fps).
- **Números do artigo:** `outputs/cover_shadow/artigo_v31.json` (`numeros_v31.py`); figuras em `outputs/cover_shadow/artigo/` (as da v3 estão em `artigo_v3/`); animação `animacoes/c06_pt_16x9.mp4` (a anterior em `animacoes/v3/`).
- *Estado:* implementado e calculado; os números estão em `artigo_v31.json` (sem leitura). O artigo (documento) e a 4.ª ronda não se tocaram.

### D44 · Moeda oficial da v3.1: risco; leitura por índice de fase (6/11, fase 9d)
Decisão do Diogo, a seguir à 9c. Substitui «Moeda principal do artigo: xT» da D43; as sombras e as probabilidades da v3.1 não mudam.
- **Moeda oficial = risco.** Valor negado numa linha `l` = `(p_l(∅) − p_l(C)) · (g_l + c_l)`: `g_l` = ganho de xT da linha (nunca negativo) e `c_l` = xT do adversário no ponto onde a linha seria cortada
  (`metrica.passe.custo_da_perda`). Parâmetros da literatura, linhas ≤ 35 m, `p_l(∅) = 1` (v3.1 sem fundo). Mesmo Shapley (escalado por `g + c`) e mesma agregação. A «ameaça disponível» de uma linha é `g + c`.
- **Leitura oficial = índice por fase defensiva.** 100 = média por momento dos jogadores nessa fase; percentil dentro da fase; só jogadores com ≥ 100 momentos (um momento = 1 frame alinhado de 0,2 s) na fase entram nas distribuições e nos percentis.
- **xT e linhas quebradas ficam como análise de sensibilidade** (mesmas tabelas, mesma agregação).
- **Frase para treinadores:** «fechar uma linha vale o perigo que o adversário deixa de ganhar, mais o perigo que passa a correr se arriscar o passe».
- **Ressalvas:** (1) o custo da perda vem da grelha de xT (12×8, grosseira); (2) o valor é potencial, o que valeria se o passe fosse tentado, e somam-se linhas alternativas entre si (não são golos);
  (3) as etiquetas de fase da SkillCorner são por fase, não por momento: um frame herda a etiqueta da fase que o contém; (4) nas variantes da sensibilidade (C4, C5) o custo `c` é o da base (parâmetros da literatura): só as sombras variam.
- **Código:** `metrica/parametros.py` tem `MOEDA_PADRAO = "risco"` e `MOEDAS_DISPONIVEIS`; `PADRAO`, `PADRAO_V3` e `PADRAO_V2` não mudam (versões anteriores reproduzíveis e a moeda xT continua a sair das tabelas guardadas).
  A moeda calcula-se sobre as tabelas da v3.1 (`avaliacao_risco.moeda_risco`; para um só lance `moeda_risco_do_lance`).
- **Números e figuras:** `outputs/cover_shadow/artigo_v31.json` (`"moeda_oficial": "risco"`, secção `risco`; as secções em xT ficam como sensibilidade) por `numeros_9d.py`; figuras em `outputs/cover_shadow/artigo/` (as em xT em `artigo_v31_xt/`);
  animação do C06 em risco (a anterior em `animacoes/v31_xt/`). O artigo (documento) não se tocou.
- *Estado:* implementado e calculado; números sem leitura.

### D45 · Cone pelo tronco: escolha de definição; veredicto do C8 (7/11)
Decisão do Diogo, na revisão do artigo.
- **A sombra pelo tronco é uma escolha de definição, tomada pela leitura do jogo** (é o corpo de quem defende que fecha a linha), e não depende de um teste estatístico.
  O C8 (D37) não escolhe a definição: verifica se a medição se comporta como essa leitura prevê.
- **Veredicto do C8 = «Cumpre parcialmente».** A sombra ajuda a prever para onde vai o passe, fora da amostra, melhor do que sem sombra; no modelo com as duas sombras (M3) a informação
  vai para o tronco; a vantagem fora da amostra face à sombra virada para a bola só é significativa num dos sentidos (números em `outputs/cover_shadow/modelo_escolha.json`).
- **Leitura no artigo:** o corpo de quem defende pesa sobretudo na decisão do portador (C8), não no sucesso do passe (γ ≈ 0, xPass). Frase acrescentada no fim da secção 7.5.
- **Regra:** se um teste futuro contrariar a leitura (por exemplo, a sombra pelo tronco sem relação com as escolhas), não se troca a definição; revê-se a medição (pose, cone, parâmetros).
- *Estado:* registado; sem alterações no código.

### D46 · Veredicto do C7 (explicabilidade) e leitura do índice por percentil (7/11)
Decisões do Diogo, na revisão final do artigo.
- **C7 = «Cumpre».** Três treinadores que não fizeram a revisão de lances viram a animação do C06 sem explicação prévia e perceberam a intenção da métrica em menos de um minuto. O funcionamento em detalhe e a leitura dos valores só ficaram claros depois de uma explicação: a animação transmite a ideia, mas os números pedem acompanhamento (a ter em conta no Reel e no vídeo longo, vistos sem ninguém a explicar).
- **Índice por fase lê-se primeiro pelo percentil e pelo multiplicador da média** («à frente de 73% dos jogadores em bloco médio, 1,6× a média (índice 157)»), porque um índice sozinho parece exagerado (distribuição assimétrica: em bloco médio, mediana 102 e p90 245; em bloco alto, 61 e 279). Aplicado no artigo (6.6, secção 8), na Figura 13 e na animação (fase 9g, `figuras_lance.leitura_do_indice`).
- **Origem do cone na tabela de parâmetros:** abertura (60°) e comprimento (10 m) passam a «decisão de método, testada» (45°/90° e 8/12 m não alteram a ordenação; C4). Não há valor publicado para adotar.

### D47 · Paleta nova, duas variantes e cor da marcação (9/11)
Decisão do Diogo: nova base de cores, mantendo a ideia azul + coral da marca. Só cores e os textos que as nomeiam; cálculos e números iguais.
- **Paleta:** fundo `#02070A` (era `#0B1426`), azul `#0297DE` (era `#3FA7FF`), coral `#FC734D` (era `#FF8A00`). A paleta antiga ficou em comentário em `config/brand.yaml`.
- **Duas variantes** (`load_brand(variante="escura"|"clara")`, escura por omissão): *escura* para redes sociais e vídeo; *clara* para o artigo (fundo branco, campo `#F5F6F8`). Na clara, áreas, barras e pontos usam o azul e o coral; texto e linhas finas usam as «tintas» `secondary_ink #0277B6` e `primary_ink #D9532E` (≥ 4:1 sobre branco). Texto principal `#02070A`. Os defesas são brancos na escura e quase pretos na clara; as legendas geram-se da variante (`idioma.cor_defesa`).
- **Cor da marcação: hipótese (b), anel e linha tracejados no tom do texto** (`save` = `#EAF0F8` na escura, `#02070A` na clara). O vermelho-rosa anterior (`#FF4D6D`) ficava a ΔE ≈ 31 do coral novo (≈ 21 em deuteranopia). ΔE (CIELAB, CIE76; matrizes de Machado 2009, severidade 1) mínimo face ao coral e ao azul: (b) escura 54 normal / 58 deuteranopia / 49 protanopia / 45 tritanopia; (b) clara 61 / 61 / 59 / 57. (a) amarelo falha: `#FFD23F` na escura fica a 29 (deuteranopia) e 31 (tritanopia, face ao branco) do coral/branco; o mostarda `#B8860B` na clara fica a 10 em deuteranopia (24 em protanopia). Ressalva: no tracejado o tom é o dos defesas (ΔE 0), por isso o que o separa deles é a forma (anel maior, vazado, tracejado), não a cor.
- **Figura 14:** «Log-loss, n = 1 408» (vírgula recuperada) e nota «Sombra = 1 − índice de chegada» / «Shadow = 1 − arrival index».

---

## Em aberto (decidir antes ou durante a modelação)

| # | Questão | Opções / valores a testar |
|---|---|---|
| A1 | Limiares da marcação (D14) | distância 2 ou 3 m; tempo mínimo ~1 s; validar nos exemplos (C2) |
| A2 | Ângulo do cone | 60° (original) e alternativas |
| A3 | Comprimento do cone (D4) | 8, 10, 12 m, ou valor fundamentado na física do passe por cima |
| A4 | Parâmetros do alcance (D3) | tempo de reação, velocidade do defesa, velocidade da bola, largura junto ao corpo |
| A11 | Novos limiares (D27–D30) | valores de partida: passe curto 12 m (testar 8, 15); atrasado 1 m de avanço (0, 2); pressão 4 m (0, 3, 5); saída da marcação 5 m (3, 4, 6); raio do bloqueio 1 m (0, 0,5, 1,5). Em teste na fase 5 (v2) |
