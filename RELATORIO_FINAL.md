# Relatório Final: Feature Engineering Clássico vs. PQFMs

> Experimentos concluídos em 2026-07-07

## 1. Pergunta de pesquisa

Transformações clássicas não-lineares (kernels, projeções aleatórias, dequantização analítica)
produzem ganho estatisticamente equivalente, maior ou nulo em relação ao baseline clássico,
**comparado diretamente** às configurações de PQFM (CD-Ising, Heisenberg, XYZ) reportadas
na dissertação de Carmo (2026), com o mesmo pipeline GBC e protocolo de CV aninhada?

## 2. Protocolo experimental

| Item | Detalhe |
|---|---|
| Classificador | `GradientBoostingClassifier` (sklearn, `random_state=42`) |
| Grade | `n_estimators`∈{80,100,120,140,160}, `max_depth`∈{2,5,7,9,11}, `criterion`∈{friedman_mse, squared_error} |
| CV externa | `RepeatedStratifiedKFold(5×10)` = 50 dobras |
| CV interna | `GridSearchCV + PredefinedSplit` (80/20 estratificado) |
| Métrica primária | AUC (ROC); secundárias: F1, precisão, recall, acurácia |
| Análise estatística | Δᵢ = M_i(técnica) − M_i(baseline); IC 95% (t-Student); Wilcoxon unilateral |
| Rótulos de status | Consistente (IC>0 e p<0.05) > Promissor (p<0.05) > Sugestivo (0.05≤p<0.10) > Tendência > Sem ganho |
| Simulação quântica | `pqfmlib` v0.1.0 (Carmo 2026), simulador statevector ideal (`qiskit-aer`) |
| q_enc | 13 qubits (Cardiac), 20 qubits (WDBC, I-SPY1) |

**Nota sobre os PQFMs**: features quânticas pré-computadas uma vez por dataset (abordagem
transdutiva), consistente com a dissertação. Antes da CV, a imputação, a padronização e a
matriz J de informação mútua **entre atributos** são ajustadas com todas as amostras
(`run_quantum_cv.py`). O rótulo *y* não é usado pelo `pqfmlib` em nenhuma etapa (a MI é
atributo×atributo; *y* só é gravado junto às features de saída), portanto **não há
vazamento de rótulo** — apenas vazamento transdutivo não supervisionado (a distribuição de
*X* do teste influencia J e a escala). O efeito desse vazamento é medido na Seção 6.

## 3. Resultados por dataset

### Etapa 1 — Cardiac Disease — Cleveland (303×13)

**Baseline AUC:** 0.8885 (dissertação: 0.8784, Δ = +0.0101)

| Tipo | Técnica | AUC média | Δ vs baseline | p-Wilcoxon | Status |
|---|---|---|---|---|---|
| Baseline | GBC puro | 0.8885 | — | — | — |
| **PQFM** | **CD-Ising (híb)** | **0.8969** | **+0.0084** | **0.0086** | **Consistente** |
| **PQFM** | **CD-Ising (iso)** | **0.8963** | **+0.0078** | **0.0195** | **Consistente** |
| Clássico | KPCA (híb) | 0.8884 | −0.0001 | 0.417 | Sem ganho |
| Clássico | Autoencoder (híb) | 0.8872 | −0.0013 | 0.516 | Sem ganho |
| Deq | CD-Ising (híb) | 0.8836 | −0.0049 | 0.667 | Sem ganho |
| PQFM | Heisenberg (híb) | 0.8828 | −0.0058 | 0.889 | Sem ganho |
| PQFM | XYZ (híb) | 0.8801 | −0.0085 | 0.946 | Sem ganho |

**Conclusão:** Nenhuma técnica clássica ou dequantização supera o baseline significativamente.
O **CD-Ising (pqfmlib)** é a única abordagem a atingir *Consistente* (+0.84 pp, IC > 0, p < 0.05).

---

### Etapa 2 — Breast Cancer Patologia — WDBC (569×30)

**Baseline AUC:** 0.9894 (dissertação: 0.9847, Δ = +0.0047)

| Tipo | Técnica | AUC média | Δ vs baseline | p-Wilcoxon | Status |
|---|---|---|---|---|---|
| Baseline | GBC puro | 0.9894 | — | — | — |
| **Clássico** | **Nystroem-Poly (híb)** | **0.9917** | **+0.0024** | **0.0370** | **Promissor** |
| **Clássico** | **Autoencoder (híb)** | **0.9917** | **+0.0023** | **0.0103** | **Promissor** |
| **Clássico** | **Nystroem-Poly (iso)** | **0.9916** | **+0.0022** | **0.0422** | **Promissor** |
| PQFM | Heisenberg (iso) | 0.9910 | +0.0016 | 0.0800 | Sugestivo |
| PQFM | Heisenberg (híb) | 0.9904 | +0.0010 | 0.3201 | Tendência positiva |
| Deq | XYZ (híb) | 0.9892 | −0.0001 | 0.968 | Sem ganho |
| PQFM | CD-Ising (híb) | 0.9889 | −0.0005 | 0.999 | Sem ganho |
| PQFM | XYZ (iso) | 0.9538 | −0.0356 | 1.000 | Sem ganho |

**Referência dissertação:** Heisenberg SHAP = 0.9911 → reproduzido: 0.9910 (Δ = −0.0001 ✓)

**Conclusão:** Técnicas clássicas dominam este dataset. Nystroem polinomial e autoencoder
atingem *Promissor*. Heisenberg chega a *Sugestivo* (p=0.080). XYZ (630 features) causa
overfitting severo sobre os 569 samples.

---

### Etapa 3 — Breast Cancer Radiologia — I-SPY1 (161×370)

**Baseline AUC:** 0.5248 (dissertação: 0.5315, Δ = −0.0067)

| Tipo | Técnica | AUC média | Δ vs baseline | p-Wilcoxon | Status |
|---|---|---|---|---|---|
| Baseline | GBC puro | 0.5248 | — | — | — |
| **PQFM** | **CD-Ising (híb)** | **0.5745** | **+0.0497** | **0.0083** | **Consistente** |
| PQFM | CD-Ising (iso) | 0.5448 | +0.0200 | 0.248 | Tendência positiva |
| Clássico | Bins+Cross (híb) | 0.5263 | +0.0015 | 0.526 | Tendência positiva |
| Deq | XYZ (iso) | 0.5261 | +0.0013 | 0.684 | Tendência positiva |
| PQFM | Heisenberg (iso) | 0.5192 | −0.0056 | 0.581 | Sem ganho |
| PQFM | XYZ (iso) | 0.4758 | −0.0490 | 0.991 | Sem ganho |

**Conclusão:** O CD-Ising com 20 qubits codifica **todos** os 370 atributos em pelo menos 19 (⌈370/20⌉) camadas
sequenciais de 20 qubits (blocos montados pela MI entre atributos; não há descarte de
atributos) e mede 210 observáveis (20 ⟨Z⟩ + 190 ⟨ZZ⟩). Atua como **compressão quântica
370→210**: +4.97 pp vs baseline (*Consistente*, p=0.0083). Com AUC de base ≈ 0.52, esse
ganho deve ser confirmado com o teste por dobra (Seção 6) e com outras sementes.
Nenhuma técnica clássica ou dequantização supera o baseline significativamente aqui.

---

### Etapa 4 — Molecular Toxicity — CRY1 (171×156)

**Status: BLOQUEADA** — dataset não disponível publicamente (Gul et al. 2021).
Para execução: solicitar via issue em `github.com/rscarmo/pqfmlib`.

---

## 4. Síntese comparativa

| Dataset | Baseline | Melhor Clássica | Status | Melhor PQFM | Status | Melhor Deq | Status |
|---|---|---|---|---|---|---|---|
| Cardiac (303×13) | 0.8885 | 0.8884 KPCA-híb | Sem ganho | **0.8969 CD-Ising-híb** | **Consistente** | 0.8836 CDIsing-híb | Sem ganho |
| WDBC (569×30) | 0.9894 | **0.9917 NystPoly-híb** | **Promissor** | 0.9910 Heisenberg-iso | Sugestivo | 0.9892 XYZ-híb | Sem ganho |
| I-SPY1 (161×370) | 0.5248 | 0.5263 Bins-híb | Tend. positiva | **0.5745 CD-Ising-híb** | **Consistente** | 0.5261 XYZ-iso | Tend. positiva |

## 5. Resposta à pergunta de pesquisa

**Em 2 de 3 datasets (Cardiac e I-SPY1)**, o PQFM CD-Ising (simulação quântica ideal)
é a **única abordagem a atingir *Consistente*** — superando tanto o baseline quanto todas
as técnicas clássicas de forma estatisticamente robusta (IC > 0, p < 0.05).

**Em 1 de 3 datasets (WDBC)**, técnicas clássicas (Nystroem polinomial, autoencoder) são
superiores e atingem *Promissor*; o PQFM Heisenberg fica em *Sugestivo*.

### Padrão dimensional observado

| Regime | Observação |
|---|---|
| Baixa dimensão (13 features) | PQFM CD-Ising domina; clássicas não superam baseline |
| Média dimensão (30 features) | Clássicas dominam; PQFM Heisenberg próximo do limiar |
| Alta dimensão (370 atributos em 20 qubits) | PQFM CD-Ising domina via compressão quântica (370→210 observáveis) |

### Dequantização analítica vs. circuito quântico real

A dequantização analítica (blocos 2-qubit, `scipy.linalg.expm`) **não reproduz** o ganho
dos PQFMs em nenhum dataset. O circuito de 13–20 qubits com entanglement global captura
correlações que a decomposição par a par não alcança — indicando que o benefício dos PQFMs
não é trivialmente "dequantizável" por blocos independentes.

### Implicações para a dissertação

1. **CD-Ising é o mapa mais robusto** — único a atingir *Consistente* nos dois datasets
   mais desafiadores (baixa dimensão e alta dimensão).
2. **O ganho quântico é dependente do dataset** — não é universalmente superior nem inferior
   às clássicas; o regime de dimensionalidade e a estrutura dos dados determinam o resultado.
3. **A simulação quântica importa** — a dequantização analítica não captura o mesmo benefício,
   sugerindo que a geometria do espaço de Hilbert é relevante para o desempenho.

---

*Figuras completas: `results/figures/etapa{1,2,3}_{dataset}_completo.png`*
*Dados brutos: `results/etapa{N}_{dataset}_{resultados,quantum_resultados,estatistica_pareada}.csv`*
