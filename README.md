# Feature Engineering Clássico vs. Projected Quantum Feature Maps (PQFMs)

Estudo comparativo: as características extraídas por mapas quânticos Hamiltonianos
(PQFMs — CD-Ising-glass, Heisenberg map, família XYZ generalizada) realmente melhoram
um classificador Gradient Boosting mais do que técnicas clássicas de feature
engineering de custo comparável, aplicadas sobre as mesmas bases e com o mesmo
protocolo de validação?

Ponto de partida: dissertação de Rafael Simões do Carmo, *"Mapas de Características
Quânticas Projetadas Baseados em Hamiltonianos Generalizados de Dois Qubits"*
(`Estudos_Especiais_II_Rafael_Simoes_Carmo.pdf`), orientador F. F. Fanchini, e o
preprint homônimo arXiv:2606.13641.

## Comece aqui

Leia **`PLANO_ESTUDO_FEATURE_ENGINEERING_VS_PQFM.md`** — plano completo, em etapas,
com:
- identificação exata das 4 bases de dados e como obtê-las (Seção 2);
- as ~10 técnicas clássicas de feature engineering a testar, incluindo uma
  "dequantização analítica" que recalcula os observáveis do PQFM em NumPy puro, sem
  simular circuito nenhum (Seção 7);
- o pipeline de aprendizado e protocolo estatístico, idênticos aos da dissertação, para
  garantir comparação justa (Seções 6 e 8);
- execução em 4 etapas, uma base por vez, da mais simples (Cardiac Disease, 303×13)
  até a mais cara (Molecular Toxicity, 171×156) (Seção 9);
- riscos conhecidos — em especial a fonte não trivial da base Molecular Toxicity/CRY1
  (Seção 2.1) — e notas de aceleração por GPU (Seção 11.1).

## Estrutura do repositório

```
├── PLANO_ESTUDO_FEATURE_ENGINEERING_VS_PQFM.md   # plano de execução (ler primeiro)
├── Estudos_Especiais_II_Rafael_Simoes_Carmo.pdf  # dissertação de referência
├── requirements.txt                              # ambiente base (CPU)
├── requirements-gpu.txt                          # extras opcionais de GPU
├── src/
│   ├── datasets.py             # loaders das 4 bases (stubs, ver TODOs)
│   ├── quantum_features.py     # wrappers em torno do pqfmlib (stubs)
│   ├── classical_features.py   # as 10 técnicas clássicas (stubs)
│   ├── pipeline.py             # GradientBoosting + CV aninhada (stub)
│   └── stats.py                # delta pareado, IC95%, Wilcoxon (stub)
├── configs/                    # um YAML por dataset (a preencher por etapa)
├── data/{raw,processed}/       # datasets baixados/pré-processados (fora do git)
├── results/                    # tabelas e figuras de síntese por etapa
└── notebooks/                  # exploração ad hoc (opcional)
```

Os módulos em `src/` são **stubs** com `NotImplementedError` e comentários indicando
exatamente o que implementar e em qual seção do plano encontrar a especificação —
o objetivo é começar a preencher etapa por etapa (Seção 9 do plano), não rodar nada
ainda.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# se houver GPU no servidor, ver Seção 11.1 do plano antes de instalar:
# pip install -r requirements-gpu.txt
```

## Ordem de execução recomendada

Cardiac Disease (13 atributos) → Breast Cancer Pathology/WDBC (30) → Breast Cancer
Radiology/I-SPY1 (370) → Molecular Toxicity/CRY1 (156 de 1203) — da base mais rápida de
depurar para a mais cara, conforme Seção 9 do plano. Antes de tudo, rodar o
**checkpoint de reprodutibilidade** da Seção 9.1: bater a AUC do baseline clássico
puro contra os números já publicados na dissertação, para validar o pipeline antes de
comparar qualquer feature nova.
