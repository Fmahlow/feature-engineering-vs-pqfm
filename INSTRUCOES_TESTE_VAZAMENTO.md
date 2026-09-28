# Teste de vazamento transdutivo do CD-Ising: instruções para o servidor

## Checklist para o agente (Claude Code) — seguir em ordem

1. Confirme que está na branch `claude/stoic-hamilton-m085a3` e atualizada
   (`git pull origin claude/stoic-hamilton-m085a3`).
2. Localize o Python com `pqfmlib`: tente `.venv311/bin/python` primeiro e depois o
   `python` ativo. Verifique com `python -c "import pqfmlib, qiskit_aer, radMLBench, ucimlrepo; print(pqfmlib.__version__)"`.
   - Se faltar algo, instale no mesmo venv:
     `pip install git+https://github.com/rscarmo/pqfmlib.git qiskit-aer ucimlrepo radMLBench`.
   - Anote a versão do `pqfmlib`.
3. Verifique os recursos (`nproc`, `free -g`, `nvidia-smi`) e informe ao usuário a
   estimativa de tempo (ver "Execução" abaixo).
4. Rode o **Cardiac** em background com `nohup` (comando abaixo) e acompanhe o log até o fim.
5. Rode o **I-SPY1 rápido** (`--n_repeats 2`) em background e acompanhe até o fim.
6. **Pergunte ao usuário** antes de rodar o I-SPY1 completo, porque pode levar muitas
   horas. Mostre os resultados rápidos e o tempo medido por circuito.
7. Com os CSVs gerados, preencha a **Seção 6** do `RELATORIO_FINAL.md`:
   - uma linha por dataset, usando a linha `_hyb` de `results/vazamento_{ds}_cdising_estatistica_pareada.csv`;
   - AUC média de cada label a partir de `results/vazamento_{ds}_cdising_resultados.csv`;
   - a versão do `pqfmlib` e a máquina usadas;
   - troque "PENDENTE" no título por "concluído" (ou "parcial", se o I-SPY1 completo não rodou).
   - Se o status de alguma técnica mudar (ver "Como interpretar"), ajuste também as conclusões
     das Etapas 1 e 3 e a Seção 5.
8. Faça commit dos CSVs `results/vazamento_*` e do relatório, e dê push para a mesma branch.
   Não faça commit dos logs nem de `results/quantum_precomputed/`.
9. Resuma ao usuário, em poucas linhas, o que mudou na conclusão.

## Objetivo

Verificar se o ganho do PQFM CD-Ising (Cardiac: +0.84 pp; I-SPY1: +4.97 pp) se mantém
quando **nada** do conjunto de teste é usado para gerar as features quânticas.

No protocolo atual (`run_quantum_cv.py`), três coisas são ajustadas com **todas** as
amostras antes da validação cruzada:

- a imputação;
- a padronização;
- a matriz J de informação mútua entre atributos.

O rótulo *y* **não** é usado, então não há vazamento de rótulo. Mesmo assim, a
distribuição de *X* do teste influencia as features.

O script `run_quantum_cv_perfold.py` roda, no mesmo ambiente e com as mesmas 50 dobras:

| Label | O que é |
|---|---|
| `{ds}_baseline` | GBC puro, recalculado no mesmo ambiente |
| `{ds}_cdising_global_{iso,hyb}` | Protocolo atual: J e escala ajustados com todas as amostras |
| `{ds}_cdising_perfold_{iso,hyb}` | J, escala e imputação ajustados **só no treino** de cada dobra; o circuito é aplicado ao treino e ao teste com esses parâmetros |

## Pré-requisitos

- Use o mesmo ambiente dos experimentos (`.venv311`, Python ≥ 3.10) com `pqfmlib`,
  `qiskit-aer`, `ucimlrepo` e `radMLBench`.
- Faça o pull da branch:
  ```bash
  git fetch origin claude/stoic-hamilton-m085a3
  git checkout claude/stoic-hamilton-m085a3
  ```
- **Versão do `pqfmlib`:** confira com `python -c "import pqfmlib; print(pqfmlib.__version__)"`.
  - O relatório usou a v0.1.0. A v0.1.1 corrigiu a ordem dos rótulos dos observáveis,
    então as features podem mudar entre as versões.
  - Isso não invalida o teste: global e por dobra rodam na mesma versão, na mesma execução.
  - Anote a versão usada no relatório.

## Execução

Rode uma base por vez, a partir da raiz do repositório:

```bash
# 1) Cardiac: 13 qubits. Cada circuito levou ~25 s em 4 vCPUs; total estimado de 1 a 1.5 h
nohup .venv311/bin/python run_quantum_cv_perfold.py --dataset cardiac > vazamento_cardiac.log 2>&1 &

# 2) I-SPY1: 20 qubits, mais de 19 camadas. Cada circuito passou de 13 min em 4 vCPUs,
#    o que dá mais de 11 h para as 50 dobras nessa máquina. Comece pela checagem rápida:
nohup .venv311/bin/python run_quantum_cv_perfold.py --dataset ispy1 --n_repeats 2 > vazamento_ispy1_rapido.log 2>&1 &
#    e depois rode o protocolo completo (sem --n_repeats):
nohup .venv311/bin/python run_quantum_cv_perfold.py --dataset ispy1 > vazamento_ispy1.log 2>&1 &
```

Para acompanhar: `grep "^\[" vazamento_cardiac.log`. O log mostra o baseline, o tempo do
circuito global e a AUC de cada configuração.

## Saídas

- `results/vazamento_{ds}_cdising_resultados.csv`: AUC, F1 etc. por dobra (5 labels × 50 dobras).
- `results/vazamento_{ds}_cdising_estatistica_pareada.csv`: Δ médio, IC95%, Wilcoxon e status para:
  - `global` vs baseline;
  - `perfold` vs baseline;
  - `perfold` vs `global`, que é o efeito do vazamento.

## Como interpretar

| Resultado | Conclusão |
|---|---|
| `perfold` vs baseline continua **Consistente** e `perfold` vs `global` ≈ 0 (IC contém 0) | O ganho do CD-Ising é real, e o vazamento transdutivo é irrelevante |
| `perfold` vs baseline cai para Sugestivo/Tendência/Sem ganho, com `perfold` < `global` | Parte do ganho vinha do ajuste transdutivo. O protocolo por dobra passa a ser o resultado principal |
| `global` já não reproduz o relatório (ex.: por mudança de versão do `pqfmlib`) | Registrar. A comparação válida é a desta execução (global × por dobra × baseline) |

## Depois de rodar

1. Faça commit dos dois CSVs de cada base em `results/`.
2. Preencha a **Seção 6** do `RELATORIO_FINAL.md` com a tabela de resultados e ajuste as
   conclusões das Etapas 1 e 3 se o status mudar.
3. Opcional: se o ganho se mantiver, repita o I-SPY1 com outra semente do
   `RepeatedStratifiedKFold` para checar a estabilidade. Com AUC de base ≈ 0.52, a
   variância entre dobras é alta.
