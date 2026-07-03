# Plano de Estudo: Feature Engineering Clássico vs. Projected Quantum Feature Maps (PQFMs)

Documento de planejamento para execução em servidor externo. Referência: Carmo, R. S.
(2026), *"Mapas de Características Quânticas Projetadas Baseados em Hamiltonianos
Generalizados de Dois Qubits"* (dissertação, Estudos Especiais II, PPGCC-UNESP Bauru,
orientador F. F. Fanchini), e o preprint homônimo Carmo, Amaro Junior & Fanchini,
arXiv:2606.13641 (11 jun. 2026) — mesmo trabalho, versão condensada em inglês com mais
detalhes de bases de dados e referências completas.

---

## 1. Pergunta de pesquisa

O trabalho de referência mostra que características extraídas de circuitos quânticos
(PQFMs — CD-Ising-glass, Heisenberg map, e a família Hamiltoniana XYZ generalizada
proposta pelo autor) concatenadas ou usadas isoladamente com um classificador clássico
(Gradient Boosting) melhoram métricas (principalmente AUC) em relação à base clássica
original, em 4 datasets biomédicos.

**Hipótese a testar:** o ganho observado não depende de um circuito quântico real —
transformações **clássicas não-lineares** de complexidade e custo computacional
comparáveis (produtos, razões, kernels, projeções aleatórias, etc.), aplicadas sobre os
mesmos pares/variáveis selecionados por informação mútua, produzem ganho igual, maior,
ou nulo em relação ao mesmo baseline clássico, com o mesmo protocolo estatístico.

**Resultado do estudo:** para cada dataset, uma tabela comparando (a) baseline clássico,
(b) cada PQFM do trabalho original, (c) ~10 técnicas clássicas de feature engineering,
todas avaliadas com o **mesmo pipeline de aprendizado, mesmo protocolo de validação
cruzada aninhada e mesmos testes estatísticos pareados** usados na dissertação.

### 1.1 Um ponto conceitual central: "dequantização" analítica

Um detalhe importante da arquitetura dos PQFMs usados no trabalho original é que os
circuitos são **rasos e não emaranhados além de pares/blocos vizinhos** (1 passo de
Trotter, blocos locais de 1–2 qubits). Isso significa que, para cada observável medido,
existe uma **fórmula analítica fechada** em termos de senos/cossenos dos ângulos de
codificação (que são, por sua vez, funções lineares/tangente-hiperbólica dos atributos
originais). Ou seja: **não é necessário simular um circuito quântico para obter esses
valores — eles podem ser calculados diretamente em NumPy**, dado o mesmo mapeamento
atributo→par de qubits (a etapa de seleção por informação mútua/algoritmo genético do
`pqfmlib`).

Isso dá origem à técnica clássica **#0 — "dequantização exata"** (Seção 7), que deve ser
tratada como o experimento mais informativo do estudo: ela isola completamente a
pergunta "o ganho vem de computação quântica?" da pergunta "o ganho vem da escolha
específica de transformação não-linear + seleção de pares por informação mútua?".

---

## 2. Bases de dados: identificação exata e fontes de aquisição

A dissertação usa 4 bases (Tabela 3.1 / Table I do arXiv). Identifiquei as fontes
primárias lendo as referências completas do preprint (arXiv:2606.13641, referências
33–37). **Etapa 0 do plano deve validar cada fonte antes de prosseguir.**

| # | Nome no trabalho | Amostras | Atributos (orig. → usados) | Fonte primária | Como obter |
|---|---|---|---|---|---|
| 1 | Cardiac Disease – Cleveland | 303 | 13 → 13 | Detrano et al., *Am. J. Cardiol.* 64, 304 (1989) — UCI Heart Disease | UCI ML Repository, dataset id 45 (`ucimlrepo`) ou `processed.cleveland.data` |
| 2 | Breast Cancer Pathology – Wisconsin Diagnostic (WDBC) | 569 | 30 → 30 | Street, Wolberg & Mangasarian, SPIE 1905 (1993) | `sklearn.datasets.load_breast_cancer()` (idêntico) ou UCI id 17 |
| 3 | Breast Cancer Radiology – I-SPY1 | 161 | 370 → 370 | Radiômica tabular via **radMLBench** (Demircioğlu, *Comput. Biol. Med.* 182, 109140, 2024), dados originais do TCIA I-SPY1 (Newitt et al., 2016, I-SPY1/ACRIN 6657) | pacote Python `radMLBench` (`pip install radMLBench`; repo `aydindemircioglu/radMLBench`) |
| 4 | Molecular Toxicity – CRY1 | 171 | 1203 → 156 (top-MI) | Gul et al., *Sci. Rep.* 11, 18510 (2021) — moléculas alvo do domínio funcional da proteína circadiana CRY1, 1203 descritores moleculares | **Ver risco R1 abaixo — fonte não trivial** |

Ordem de execução recomendada: **13 → 30 → 370 → 156**, da mais simples/rápida para a
mais cara, permitindo depurar todo o pipeline nas duas primeiras antes de gastar tempo
de simulação nas maiores.

### 2.1 Risco R1 — Molecular Toxicity (CRY1)

Diferente das outras 3 bases, não há um pacote Python público e direto para essa base.
A dissertação (Cap. 3) e o preprint (ref. [33]) apontam a origem para Gul et al. (2021),
mas o `pqfmlib` do próprio autor (`github.com/rscarmo/pqfmlib`) apenas referencia o nome
de arquivo `Toxicity_preprocessed_shuffled` nos exemplos, sem incluir o CSV no repositório
público — ou seja, o dataset pré-processado (1203 → seleção dos 156 por informação
mútua) não está publicamente hospedado no repo da lib.

Ações a fazer no servidor, **em ordem de preferência**, antes de investir tempo na Etapa
5 (Seção 9.4):
1. Verificar o material suplementar do artigo Gul et al., *Scientific Reports* 11, 18510
   (2021) — é open access (Nature/Springer), deve ter os 1203 descritores moleculares
   (provavelmente calculados via PaDEL ou RDKit) em planilha suplementar.
2. Verificar se os preprints da Kipu Quantum citados no trabalho (Simen et al.,
   arXiv:2508.20975 "Quenched quantum feature maps" e arXiv:2510.13807 "Digitized
   counterdiabatic quantum feature extraction") — que são a fonte original do uso dessa
   base em PQFMs — têm repositório de reprodutibilidade com o CSV pré-processado.
3. Abrir uma issue/contato no repositório `rscarmo/pqfmlib` perguntando pela localização
   do arquivo `Toxicity_preprocessed_shuffled.csv` usado nos exemplos.
4. **Fallback**: se nenhuma fonte for localizada, substituir por um dataset público
   equivalente de toxicidade molecular com featurização de alta dimensão (ex.: subconjunto
   do Tox21 ou ClinTox via `deepchem`/MoleculeNet, com descritores RDKit/Mordred
   gerados localmente até ~1000+ colunas). Isso deve ser **documentado explicitamente**
   como substituição, não como reprodução exata — o objetivo do estudo (comparar FE
   clássico vs. PQFM) continua válido mesmo com um dataset substituto, só perde a
   comparabilidade direta com os números da Tabela A.1–A.3 da dissertação.

Não bloquear as Etapas 1–3 (as outras 3 bases) esperando resolução deste risco.

---

## 3. Ambiente e ferramentas

```
Python 3.10 ou 3.11
qiskit, qiskit-aer          # simulação por vetor de estado (AerSimulator)
pqfmlib                      # git clone https://github.com/rscarmo/pqfmlib && pip install -e .
scikit-learn                 # GradientBoostingClassifier, PolynomialFeatures, Nystroem,
                              # KernelPCA, KBinsDiscretizer, GridSearchCV, PredefinedSplit
scipy                         # wilcoxon
shap                          # seleção de atributos (Breast Cancer Pathology)
ucimlrepo                     # Cardiac Disease
radMLBench                    # Breast Cancer Radiology (I-SPY1)
xgboost                       # opcional, ver Seção 6.1
pandas, numpy, matplotlib
```

Fixar `random_state=42` em todos os pontos estocásticos (igual à dissertação, Seção
3.5/3.6), e registrar versões exatas de cada biblioteca em `requirements.txt` /
`environment.yml` para reprodutibilidade entre execuções.

Se o servidor tiver GPU, instalar também `requirements-gpu.txt` (por cima do
`requirements.txt` base) — ver Seção 11.1 para quais partes do pipeline realmente se
beneficiam disso.

### 3.1 Estrutura de diretórios sugerida

```
feature_engineering_vs_pqfm/
├── data/                      # CSVs brutos e pré-processados por dataset
├── src/
│   ├── datasets.py            # loaders + pré-processamento padronizado (Seção 5)
│   ├── quantum_features.py    # wrappers em torno do pqfmlib (baseline "a bater")
│   ├── classical_features.py  # as ~10 técnicas de FE clássico (Seção 7)
│   ├── pipeline.py            # GradientBoosting + GridSearch + nested CV (Seção 6)
│   └── stats.py                # delta pareado, IC 95%, Wilcoxon, rótulos (Seção 8)
├── configs/                   # um YAML/JSON por dataset com hiperparâmetros de cada PQFM
├── results/                   # tabelas .csv e figuras por etapa
└── notebooks/                 # exploração ad hoc (opcional)
```

---

## 4. Pré-processamento (idêntico à dissertação, Seção 3.1)

Para todas as bases:
- Padronização (`StandardScaler`, média 0 / variância 1) ajustada **apenas no
  conjunto de treino de cada fold externo** (sem vazamento).
- Valores ausentes: imputação pela mediana do treino correspondente.
- Amostras embaralhadas antes da validação cruzada (a dissertação discute em detalhe,
  Seção 4.2, como a ordem de submissão a hardware quântico pode contaminar resultados
  com efeito de deriva de calibração — como usaremos simulação noiseless, esse efeito
  específico não se aplica, mas embaralhar continua sendo boa prática de CV).

---

## 5. Geração das características quânticas (baseline "a bater")

Para cada dataset, gerar as mesmas 3 famílias de PQFM da dissertação, usando o
`pqfmlib` (classes `CDIsingProjectiveQFM`, `HeisenbergProjectiveQFM`, `XYZProjectiveQFM`):

1. **CD-Ising-glass** — observáveis `⟨Z⟩` (1-corpo) e `⟨ZZ⟩` (2-corpo).
2. **Heisenberg map** — observáveis `⟨X⟩`, `⟨Y⟩`, `⟨Z⟩` (1-corpo), opcionalmente
   `⟨XX⟩`/`⟨YY⟩`/`⟨ZZ⟩` (2-corpo).
3. **Família XYZ generalizada** — variantes diagonal, cross-only e full/shared-feature
   (Seção 2.4 da dissertação), com codificação multi-eixo quando aplicável.

**Decisão de design importante:** a dissertação usa hardware real (IBM QPU) para
Molecular Toxicity e Breast Cancer Radiology, e simulação por vetor de estado
(`AerSimulator`, `statevector`) para Breast Cancer Pathology e Cardiac Disease. Para
este estudo comparativo, **usar simulação por vetor de estado (noiseless) para as 4
bases**, não hardware real. Motivo: (a) reduz custo/tempo de fila de QPU; (b) remove o
efeito de ruído/deriva de calibração que a própria dissertação identifica como fator de
confusão (Seção 4.2); (c) o objetivo aqui é isolar "estrutura matemática do mapa" vs.
"engenharia de atributos clássica", não efeitos de hardware NISQ. Isso deve ser
declarado explicitamente no relatório final como um desvio metodológico consciente em
relação à dissertação original.

Usar a mesma atribuição de atributos a qubits (busca por algoritmo genético,
maximizando informação mútua entre pares mapeados em arestas — Eqs. 2.16/2.30 da
dissertação) já implementada no `pqfmlib`, para manter comparabilidade com o trabalho
original quanto à *escolha de pares*.

Para cada dataset, escolher as configurações de PQFM (nº de qubits, observáveis,
variante XYZ) que tiveram a melhor AUC média reportada na dissertação para aquela base
(ver Tabelas 4.1/4.2/A.1–A.3 e Tabelas III/IV/V–X do arXiv) — isso define um alvo fixo
e documentado para cada dataset, evitando repetir toda a busca de configurações do
trabalho original.

---

## 6. Pipeline clássico de aprendizado (reutilizar sem alterações)

Reproduzir exatamente a Seção 3.5–3.7 da dissertação:

- Classificador: `sklearn.ensemble.GradientBoostingClassifier`, `random_state=42`.
- Grade de hiperparâmetros (Tabela 3.2):
  - `n_estimators`: {80, 100, 120, 140, 160}
  - `max_depth`: {2, 5, 7, 9, 11}
  - `criterion`: {friedman_mse, squared_error}
- Seleção de hiperparâmetros por `GridSearchCV`, métrica AUC, com holdout interno
  estratificado 80/20 via `PredefinedSplit`.
- Validação cruzada aninhada: `StratifiedKFold` 5 folds × 10 repetições = 50 avaliações
  externas (loop externo). Refit no treino externo completo, avaliação única no teste
  externo (Figura 3.1 da dissertação).
- Métricas: AUC (primária), F1, precisão, recall, acurácia.

### 6.1 Nota sobre "XGBoost" vs. Gradient Boosting

O usuário mencionou XGBoost, mas a dissertação usa `GradientBoostingClassifier` do
scikit-learn (não o pacote `xgboost`). **Para comparabilidade direta com os números da
dissertação, manter `GradientBoostingClassifier` como algoritmo principal em todas as
etapas.** Como checagem de robustez opcional (Seção 10, extensão), repetir a Etapa 4
final (síntese) com `xgboost.XGBClassifier` no mesmo espaço de busca adaptado, apenas
para as 2 ou 3 técnicas clássicas que melhor performarem — não repetir toda a grade de
~13 configurações (3 PQFMs + 10 técnicas clássicas) × 4 bases com dois algoritmos, pois
o custo computacional dobraria sem agregar muito à pergunta central.

---

## 7. Técnicas clássicas de feature engineering (10 candidatas)

Todas devem ser aplicadas produzindo **aproximadamente o mesmo número de atributos
novos** que a configuração de PQFM escolhida para aquele dataset (para comparação justa
de capacidade representacional), e avaliadas em 3 modos — igual ao protocolo da
dissertação: **(i) atributos clássicos originais** (já é o baseline), **(ii) atributos
da técnica isolados**, **(iii) híbrido** (concatenação clássico+técnica, com
seleção/compressão via SHAP ou PCA quando aplicável, replicando Seção 3.4/4.4).

| # | Técnica | Descrição | Biblioteca |
|---|---|---|---|
| 0 | **Dequantização analítica** | Recalcular em NumPy, sem simulação de circuito, as fórmulas fechadas de `⟨Z_i⟩`, `⟨Z_iZ_j⟩` (CD-Ising), `⟨X/Y/Z⟩` (Heisenberg) e dos canais XYZ, usando os mesmos pares/eixos escolhidos pelo `pqfmlib`. Ver Seção 1.1. | NumPy puro (fórmulas das Eqs. 2.15, 2.18–2.19, 2.21–2.22, 2.28 da dissertação) |
| 1 | Aritmética par a par nos pares de maior informação mútua | soma, diferença, produto, razão (`x_i·x_j`, `x_i/x_j`, etc.) exatamente nos mesmos pares selecionados pelo algoritmo genético do PQFM | pandas/NumPy |
| 2 | Polinômio completo grau 2 | todos os pares e quadrados, sem seleção prévia | `sklearn.preprocessing.PolynomialFeatures(degree=2)` |
| 3 | Random Fourier Features (RBF) | aproximação clássica de kernel RBF, mesma dimensionalidade de saída do PQFM | `sklearn.kernel_approximation.RBFSampler` |
| 4 | Nystroem (RBF) | aproximação de kernel via amostragem de landmarks | `sklearn.kernel_approximation.Nystroem(kernel="rbf")` |
| 5 | Nystroem (polinomial) | idem, kernel polinomial grau 2–3 | `Nystroem(kernel="poly", degree=2 ou 3)` |
| 6 | Kernel PCA (RBF) | projeção não-linear via autovalores do kernel | `sklearn.decomposition.KernelPCA(kernel="rbf")` |
| 7 | Projeções aleatórias não-lineares (estilo ELM) | `tanh`/`sin` de combinação linear aleatória dos atributos — análogo clássico mais próximo de "observáveis de um Hamiltoniano aleatório" | NumPy (pesos aleatórios fixos por seed) |
| 8 | Autoencoder raso (bottleneck) | MLP autoencoder pequeno, dimensão do gargalo = nº de atributos quânticos, treinado sem vazamento dentro de cada fold externo | `sklearn.neural_network.MLPRegressor` (autoencoder manual) ou `keras`/`torch` leve |
| 9 | Discretização + cruzamento (binning) | `KBinsDiscretizer` seguido de cruzamento dos bins nos pares de maior IM | `sklearn.preprocessing.KBinsDiscretizer` |

Todas as técnicas de 1–9 devem, quando fizer sentido, ser restritas aos **mesmos pares
de atributos** que o PQFM equivalente usou (via a mesma matriz de informação mútua já
calculada para a Etapa 5), de forma a isolar "escolha de pares" de "forma da
transformação não-linear" como duas variáveis independentes do experimento.

---

## 8. Protocolo estatístico (reutilizar sem alterações — Seção 3.7 da dissertação)

Para cada dataset, cada técnica (quântica ou clássica) vs. baseline clássico:

```
Δ_i = M_i^(técnica) − M_i^(clássico)     para cada uma das 50 dobras externas
```

- Média de Δ, IC 95% (distribuição t de Student), teste de Wilcoxon unilateral
  (`alternative="greater"`).
- Rótulos idênticos aos da dissertação:
  - **Consistente**: Δ̄ > 0, IC 95% totalmente > 0, p < 0.05
  - **Promissor**: Δ̄ > 0, p < 0.05, IC cruza zero
  - **Sugestivo**: Δ̄ > 0, IC cruza zero, 0.05 ≤ p < 0.10
  - **Tendência positiva**: Δ̄ > 0, sem evidência suficiente nos critérios acima

Manter exatamente esses rótulos permite comparar diretamente "quantas configurações
clássicas atingem *Consistente*" vs. "quantas configurações de PQFM atingem
*Consistente*" na síntese final (Seção 9.5).

---

## 9. Execução em etapas (uma base por vez)

### 9.1 Etapa 0 — Setup e checkpoint de reprodutibilidade
1. Montar ambiente (Seção 3), clonar `pqfmlib`.
2. Baixar as 4 bases (Seção 2) e validar shapes contra a Tabela 3.1/Table I
   (13, 30, 370, 156 atributos; 303, 569, 161, 171 amostras).
3. Rodar apenas o **baseline clássico** (sem nenhum PQFM) nas 4 bases com o pipeline da
   Seção 6, e comparar a AUC obtida com a coluna "Clássico" das tabelas da dissertação
   (ex.: WDBC → AUC 0.9847; Cleveland → AUC 0.8784; I-SPY1 → AUC 0.5315–0.5389).
   **Não prosseguir até esses números baterem dentro de variação razoável de seed/CV.**
   Isso garante que o pipeline de ML está corretamente implementado antes de comparar
   features.

### 9.2 Etapa 1 — Cardiac Disease (Cleveland), 303×13
Base mais simples/rápida, ideal para depurar o pipeline completo (quântico + 10
técnicas clássicas + estatística) antes de escalar.
1. Gerar PQFMs de referência (config. do artigo: CD-Ising, Heisenberg, XYZ-cross/full/
   diagonal com shared-feature, 13–14 qubits — Tabela 4.2/Table IV) via statevector.
2. Gerar as 10 técnicas clássicas (Seção 7), com PCA a 13 componentes no modo híbrido
   (replicando a Seção 3.4/4.4: "PCA sobre atributos quânticos antes da concatenação").
3. Rodar pipeline completo (baseline, cada PQFM, cada técnica clássica; isolado e
   híbrido) → ~1 + 3 + 10×2 = 24 configurações × 50 dobras.
4. Estatística pareada (Seção 8) para cada configuração vs. baseline.
5. Tabela de síntese da Etapa 1 + interpretação curta.

### 9.3 Etapa 2 — Breast Cancer Pathology (WDBC), 569×30
1. Gerar PQFMs de referência (Tabela 4.1/Table III — CD-Ising, Heisenberg, XYZ-full/
   diagonal/cross, 15–20 qubits), com seleção SHAP dos 30 melhores atributos
   (clássico, quântico e híbrido), replicando Seção 3.4/4.4.
2. Repetir passos 2–5 da Etapa 1 adaptados a esta base.
3. Atenção: o baseline clássico já é muito forte aqui (AUC 0.9847) — a dissertação nota
   pouca margem de ganho; documentar se as técnicas clássicas também mostram teto
   semelhante ou se alguma consegue romper esse teto.

### 9.4 Etapa 3 — Breast Cancer Radiology (I-SPY1), 161×370
1. Gerar PQFMs de referência via **statevector** (não QPU real, ver Seção 5) nas
   configurações que tiveram melhor AUC na dissertação (Table VII-arXiv / Fig. 4.3:
   XYZ-diagonal Z-local, XYZ-full XX, XY-cross-only YY, CD-Ising ZZ, Heisenberg
   Z-local — todas usando 370 atributos de entrada mapeados em 124–156 qubits
   simulados).
2. Gerar as 10 técnicas clássicas com a mesma dimensionalidade de saída.
3. Atenção a custo computacional: simular 124–156 qubits por *statevector* é caro
   (2^124+ amplitudes é inviável); a dissertação consegue isso porque os circuitos são
   produtos de blocos locais rasos — usar preferencialmente simulação por blocos/MPS
   do próprio `pqfmlib` (ele já implementa isso para o Heisenberg map) em vez de
   `AerSimulator` genérico ingênuo, ou aproveitar a dequantização analítica (técnica #0)
   também como atalho de implementação para o próprio PQFM de referência (mesma
   matemática, sem overhead de simulação de circuito).
4. Repetir estatística e síntese.

### 9.5 Etapa 4 — Molecular Toxicity (CRY1), 171×156 (de 1203)
Só iniciar após resolver ou contornar o Risco R1 (Seção 2.1).
1. Selecionar os 156 atributos de maior informação mútua com o alvo, entre os 1203
   descritores moleculares originais (replicando a Seção 3.1 da dissertação).
2. Repetir passos 1–4 da Etapa 3.
3. Se usando dataset substituto (fallback do Risco R1), rotular claramente os
   resultados desta etapa como "estudo exploratório em dataset análogo", não como
   reprodução direta da Tabela A.1–A.3.

### 9.6 Etapa 5 — Síntese comparativa final
Consolidar as 4 tabelas de síntese em uma única visão:
- Para cada dataset, ranquear todas as configurações (PQFMs + técnicas clássicas) pela
  AUC média e pelo rótulo estatístico (Consistente > Promissor > Sugestivo > Tendência
  > sem ganho).
- Responder diretamente à pergunta do estudo: **em quantos dos 4 datasets pelo menos
  uma técnica clássica iguala ou supera o melhor PQFM da dissertação, com o mesmo nível
  de suporte estatístico?**
- Separar o resultado da técnica #0 (dequantização) das demais — se a dequantização
  igualar o PQFM simulado (esperado, matematicamente devem ser idênticas a menos de
  ruído de shots/estatística de medição), isso confirma que o "ganho quântico" reportado
  é, na prática, um ganho de **engenharia de atributos com estrutura Hamiltoniana
  específica**, reproduzível deterministicamente sem qualquer recurso quântico —
  reforçando ainda mais a pergunta de pesquisa sobre se as demais técnicas clássicas
  (#1–#9) também alcançam o mesmo patamar.
- Redigir um mini-relatório (Markdown) com tabelas + 3–5 parágrafos de discussão,
  espelhando a estrutura da Seção 4.5/Tabela 4.3 da dissertação ("Síntese comparativa").

---

## 10. Entregáveis esperados

1. Código-fonte organizado conforme Seção 3.1.
2. `results/etapa{1..4}_sintese.csv` — uma linha por (dataset, configuração, métrica).
3. `results/etapa{1..4}_estatistica_pareada.csv` — replicando o formato das Tabelas
   A.4/VIII–X da dissertação (Δ médio, IC95%, p de Wilcoxon, rótulo).
4. Figuras de barras (uma por dataset, estilo Fig. 4.2/4.3/8) comparando baseline,
   melhores PQFMs e melhores técnicas clássicas lado a lado.
5. `RELATORIO_FINAL.md` com a síntese da Seção 9.6.

---

## 11. Riscos e cuidados adicionais

- **R1** (Molecular Toxicity): ver Seção 2.1.
- **R2 — custo de simulação**: statevector para 100+ qubits só é viável porque os
  PQFMs do trabalho são produtos de blocos locais rasos; nunca tentar simular o
  registrador inteiro como um estado denso genérico. Usar a estrutura por blocos do
  `pqfmlib` (ou a técnica #0) sempre que o número de qubits simulado ultrapassar ~20–24.
- **R3 — vazamento de dados**: replicar rigorosamente a regra da dissertação de que
  SHAP/PCA/autoencoder (técnica #8) sejam ajustados **apenas no treino de cada fold
  externo**, nunca no dataset completo antes do split.
- **R4 — comparabilidade de dimensionalidade**: ao comparar técnica clássica vs. PQFM,
  manter o número de atributos gerados aproximadamente igual; caso contrário, qualquer
  ganho/perda pode refletir diferença de capacidade em vez de diferença de "qualidade"
  da transformação.
- **R5 — custo computacional total**: 4 bases × (~3 PQFMs + 10 técnicas clássicas) ×
  (isolado + híbrido) × 50 dobras × grid search (5×5×2=50 combinações) é substancial.
  Considerar paralelizar por dataset/técnica (embaraçosamente paralelo) e, se necessário,
  reduzir repetições de CV externas de 10 para 3–5 nas etapas exploratórias iniciais,
  voltando a 10 apenas na rodada final de cada dataset.

### 11.1 Aceleração por GPU

Nem todo o pipeline se beneficia de GPU — a maior parte do custo (GridSearchCV com
GradientBoostingClassifier, 50 dobras × 50 combinações de hiperparâmetros) é
CPU-bound e paraleliza melhor com `n_jobs=-1`/múltiplos processos do que com GPU. Os
dois pontos do plano que **de fato** valem a pena rodar em GPU, se o servidor tiver
uma disponível:

1. **Simulação dos PQFMs em regimes de muitos qubits** (Etapas 3 e 4 — I-SPY1 e
   Molecular Toxicity, 78–156 qubits simulados). Se optar por `AerSimulator` genérico
   em vez da rota de simulação por blocos/MPS já embutida no `pqfmlib` ou da técnica
   de dequantização analítica (#0, que é a mais barata de todas por não simular
   circuito nenhum), usar o backend `qiskit-aer-gpu` (`AerSimulator(method="statevector",
   device="GPU")`). Ver `requirements-gpu.txt`. **Recomendação prática:** tentar
   primeiro a rota MPS/dequantização em CPU antes de recorrer a GPU aqui — só vale a
   pena instalar/depurar `qiskit-aer-gpu` se essa rota mais barata não for viável.
2. **Treino do autoencoder raso (técnica #8)**, rodado uma vez por fold externo × por
   dataset × por modo (isolado/híbrido) — com `torch` + CUDA (`device="cuda"` em
   `src/classical_features.shallow_autoencoder`), o treino de cada rede pequena cai de
   segundos para frações de segundo, o que importa multiplicado por 50 dobras × 4
   datasets.

As demais 9 técnicas clássicas (polinômio, RFF, Nystroem, kernel PCA, projeções
aleatórias, binning) e o próprio `GradientBoostingClassifier` são implementadas em
`scikit-learn`/NumPy puro e **não usam GPU** — não faz sentido investir tempo tentando
acelerá-las por esse caminho; o ganho de velocidade ali vem de paralelismo em CPU
(`n_jobs`) e/ou redução de repetições de CV nas rodadas exploratórias (ver R5).

---

## 12. Extensões opcionais (após o estudo principal)

- Repetir a síntese final (Seção 9.6) com `XGBClassifier` real, apenas nas 2–3 melhores
  técnicas clássicas + melhor PQFM por dataset (ver Seção 6.1).
- Testar interações de 3 corpos nas técnicas clássicas (ex.: `x_i·x_j·x_k`), espelhando
  a extensão de "canais adicionais de correlação" que a própria dissertação lista como
  trabalho futuro (Cap. 5, terceira direção).
- Avaliar se um *classical surrogate* (modelo clássico treinado para prever as
  características quânticas a partir dos atributos originais, ideia citada na
  Conclusão da dissertação — Flores-Garrigós et al. 2026) consegue aproximar tanto o
  PQFM quanto as técnicas clássicas propostas aqui, unificando as duas comparações.
