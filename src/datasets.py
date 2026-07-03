"""
Loaders e pre-processamento padronizado para as 4 bases do estudo.

Ver PLANO_ESTUDO_FEATURE_ENGINEERING_VS_PQFM.md, Secoes 2 e 4, para a
identificacao exata de cada base e as regras de pre-processamento
(padronizacao ajustada so no treino de cada fold externo, imputacao pela
mediana, sem vazamento).

Cada loader abaixo deve devolver (X, y) como arrays/DataFrames brutos,
SEM padronizacao -- a padronizacao e feita dentro do pipeline de CV
aninhada (src/pipeline.py), fold a fold.
"""

from __future__ import annotations

import pandas as pd


def load_cardiac_disease() -> tuple[pd.DataFrame, pd.Series]:
    """Cleveland Heart Disease (UCI id 45). 303 amostras x 13 atributos.

    Alvo binarizado: presenca (1) vs ausencia (0) de doenca cardiaca,
    a partir do resultado original do angiograma (>0 -> 1).
    """
    raise NotImplementedError(
        "TODO: usar ucimlrepo.fetch_ucirepo(id=45) e binarizar o alvo."
    )


def load_breast_cancer_pathology() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Diagnostic Breast Cancer (WDBC). 569 amostras x 30 atributos."""
    raise NotImplementedError(
        "TODO: sklearn.datasets.load_breast_cancer(as_frame=True)."
    )


def load_breast_cancer_radiology() -> tuple[pd.DataFrame, pd.Series]:
    """I-SPY1 via radMLBench. 161 amostras x 370 atributos radiomicos."""
    raise NotImplementedError(
        "TODO: pip install radMLBench; localizar a chave exata do dataset "
        "(ex.: radMLBench.getDatasets()) antes de carregar."
    )


def load_molecular_toxicity() -> tuple[pd.DataFrame, pd.Series]:
    """Molecular Toxicity - CRY1 (Gul et al., Sci. Rep. 11, 18510, 2021).

    ATENCAO: ver Risco R1 no plano -- fonte nao trivial, pode exigir
    material suplementar do artigo original ou dataset substituto.
    171 amostras x 1203 descritores originais (usar top-156 por
    informacao mutua com o alvo).
    """
    raise NotImplementedError(
        "TODO: resolver Risco R1 (Secao 2.1 do plano) antes de implementar."
    )
