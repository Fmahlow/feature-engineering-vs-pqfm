# Data

`raw/` -- datasets baixados sem modificacao das fontes listadas na Secao 2
do plano (UCI, sklearn, radMLBench, e a base Molecular Toxicity/CRY1 --
ver Risco R1). Nao versionar os arquivos brutos aqui (ver `.gitignore`);
manter so este README com instrucoes/scripts de download.

`processed/` -- versoes padronizadas/pre-processadas (Secao 4 do plano),
geradas por `src/datasets.py`. Tambem fora do git por padrao.
