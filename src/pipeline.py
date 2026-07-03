"""
Pipeline classico de aprendizado + validacao cruzada aninhada, replicando
sem alteracoes a Secao 3.5-3.7 da dissertacao. Ver Secao 6 do plano.

GradientBoostingClassifier (sklearn), random_state=42, grade de
hiperparametros da Tabela 3.2, GridSearchCV com PredefinedSplit (holdout
interno 80/20 estratificado), loop externo StratifiedKFold 5x10 repeticoes
(50 avaliacoes), AUC como metrica primaria.
"""

from __future__ import annotations

import pandas as pd

HYPERPARAM_GRID = {
    "n_estimators": [80, 100, 120, 140, 160],
    "max_depth": [2, 5, 7, 9, 11],
    "criterion": ["friedman_mse", "squared_error"],
}

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10


def run_nested_cv(X: pd.DataFrame, y: pd.Series, label: str) -> pd.DataFrame:
    """Roda o protocolo de CV aninhada completo para uma configuracao de
    atributos (baseline, PQFM isolado/hibrido, ou tecnica classica
    isolada/hibrida) e devolve um DataFrame com uma linha por dobra externa
    (50 linhas), contendo AUC, F1, precisao, recall, acuracia.

    `label` identifica a configuracao (ex.: "cardiac_baseline",
    "cardiac_cd_ising_zz", "cardiac_dequantization_hybrid") para depois
    casar com o baseline correspondente em src/stats.py.
    """
    raise NotImplementedError(
        "TODO: RepeatedStratifiedKFold(n_splits=5, n_repeats=10, "
        "random_state=42) como loop externo; dentro de cada fold, "
        "GridSearchCV(GradientBoostingClassifier(random_state=42), "
        "HYPERPARAM_GRID, scoring='roc_auc', cv=PredefinedSplit(...)) "
        "sobre um holdout interno 80/20 estratificado do treino externo."
    )
