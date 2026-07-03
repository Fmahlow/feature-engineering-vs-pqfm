# Configs

Um arquivo por dataset (`cardiac_disease.yaml`, `breast_cancer_pathology.yaml`,
`breast_cancer_radiology.yaml`, `molecular_toxicity.yaml`) com:

- a configuracao de cada PQFM de referencia escolhida na Secao 5 do plano
  (numero de qubits, observaveis medidos, variante XYZ), fixada a partir das
  tabelas da dissertacao para aquele dataset;
- os pares/eixos selecionados por informacao mutua (reaproveitados pelas
  tecnicas classicas #0 e #1, ver Secao 7 do plano);
- a dimensionalidade de saida alvo para as tecnicas #3-#7 e #9 (para casar
  com o numero de atributos gerados pelo PQFM correspondente -- Risco R4).

Ainda nao preenchidos -- criar ao iniciar cada etapa (Secao 9 do plano).
