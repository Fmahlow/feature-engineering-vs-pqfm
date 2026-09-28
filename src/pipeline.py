"""
Pipeline classico de aprendizado + validacao cruzada aninhada, replicando
sem alteracoes a Secao 3.5-3.7 da dissertacao.

GradientBoostingClassifier (sklearn), random_state=42, grade da Tabela 3.2,
GridSearchCV com PredefinedSplit (holdout interno 80/20 estratificado),
loop externo RepeatedStratifiedKFold 5x10 = 50 avaliacoes, AUC primaria.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler

HYPERPARAM_GRID = {
    "n_estimators": [80, 100, 120, 140, 160],
    "max_depth": [2, 5, 7, 9, 11],
    "criterion": ["friedman_mse", "squared_error"],
}

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10


def _inner_split(X_train: np.ndarray, y_train: np.ndarray) -> PredefinedSplit:
    """Cria um PredefinedSplit 80/20 estratificado sobre o conjunto de treino externo."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    inner_train_idx, inner_val_idx = next(sss.split(X_train, y_train))
    test_fold = np.full(len(X_train), -1, dtype=int)
    test_fold[inner_val_idx] = 0
    return PredefinedSplit(test_fold)


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    feature_fn: Optional[Callable] = None,
    mode: str = "baseline",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Roda o protocolo de CV aninhada completo.

    Args:
        X: DataFrame de atributos (sem padronizacao).
        y: Series de rotulos binarios.
        label: Identificador da configuracao (ex.: "cardiac_baseline").
        feature_fn: Callable com assinatura
            feature_fn(X_train_scaled, X_test_scaled) -> (X_tr_feat, X_te_feat).
            Se None, usa mode="baseline" automaticamente.
        mode: "baseline" (usa X original), "isolated" (apenas features geradas),
              "hybrid" (concatena X original + features geradas).
        n_jobs: Paralelismo para GridSearchCV.

    Returns:
        DataFrame com 50 linhas (uma por dobra externa), colunas:
        [label, fold, auc, f1, precision, recall, accuracy].
    """
    if feature_fn is None:
        mode = "baseline"

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_arr, y_arr)):
        X_outer_train, X_outer_test = X_arr[train_idx], X_arr[test_idx]
        y_outer_train, y_outer_test = y_arr[train_idx], y_arr[test_idx]

        # imputacao pela mediana (fit apenas no treino)
        imputer = SimpleImputer(strategy="median")
        X_outer_train = imputer.fit_transform(X_outer_train)
        X_outer_test = imputer.transform(X_outer_test)

        # padronizacao (fit apenas no treino)
        scaler = StandardScaler()
        X_outer_train_sc = scaler.fit_transform(X_outer_train)
        X_outer_test_sc = scaler.transform(X_outer_test)

        if mode == "baseline" or feature_fn is None:
            X_tr = X_outer_train_sc
            X_te = X_outer_test_sc
        else:
            X_tr_feat, X_te_feat = feature_fn(X_outer_train_sc, X_outer_test_sc)
            if mode == "isolated":
                X_tr = X_tr_feat
                X_te = X_te_feat
            else:  # hybrid
                X_tr = np.hstack([X_outer_train_sc, X_tr_feat])
                X_te = np.hstack([X_outer_test_sc, X_te_feat])

        # selecao de hiperparametros via GridSearchCV + PredefinedSplit 80/20
        ps = _inner_split(X_tr, y_outer_train)
        clf_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
        gs = GridSearchCV(
            clf_base,
            HYPERPARAM_GRID,
            scoring="roc_auc",
            cv=ps,
            n_jobs=n_jobs,
            refit=False,
        )
        gs.fit(X_tr, y_outer_train)
        best_params = gs.best_params_

        # refit no treino externo completo com melhores hiperparametros
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
        clf.fit(X_tr, y_outer_train)

        # avaliacao no teste externo
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        records.append({
            "label": label,
            "fold": fold_idx,
            "auc": roc_auc_score(y_outer_test, y_prob),
            "f1": f1_score(y_outer_test, y_pred, zero_division=0),
            "precision": precision_score(y_outer_test, y_pred, zero_division=0),
            "recall": recall_score(y_outer_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_outer_test, y_pred),
        })

    return pd.DataFrame(records)


def run_nested_cv_precomputed(
    X: np.ndarray,
    y: np.ndarray,
    Xq_full: np.ndarray,
    label: str,
    mode: str = "isolated",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Nested CV com features quânticas pré-computadas (uma linha por amostra).

    Diferença de run_nested_cv: recebe Xq_full (n_samples, n_q_features) já
    calculado sobre o dataset completo. A cada fold, fatia por train_idx/test_idx
    sem re-computar features quânticas.

    Args:
        X: array de features originais (antes de imputação/escalonamento).
        y: array de rótulos.
        Xq_full: matriz de features quânticas para TODAS as amostras.
        label: identificador da configuração.
        mode: "isolated" (só Xq) ou "hybrid" (X_sc + Xq).
        n_jobs: paralelismo GridSearchCV.
    """
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )

    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X_arr, y_arr)):
        X_outer_train, X_outer_test = X_arr[train_idx], X_arr[test_idx]
        y_outer_train, y_outer_test = y_arr[train_idx], y_arr[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_outer_train = imputer.fit_transform(X_outer_train)
        X_outer_test = imputer.transform(X_outer_test)

        scaler = StandardScaler()
        X_outer_train_sc = scaler.fit_transform(X_outer_train)
        X_outer_test_sc = scaler.transform(X_outer_test)

        Xq_train = Xq_full[train_idx]
        Xq_test = Xq_full[test_idx]

        if mode == "isolated":
            X_tr = Xq_train
            X_te = Xq_test
        else:  # hybrid
            X_tr = np.hstack([X_outer_train_sc, Xq_train])
            X_te = np.hstack([X_outer_test_sc, Xq_test])

        ps = _inner_split(X_tr, y_outer_train)
        clf_base = GradientBoostingClassifier(random_state=RANDOM_STATE)
        gs = GridSearchCV(
            clf_base,
            HYPERPARAM_GRID,
            scoring="roc_auc",
            cv=ps,
            n_jobs=n_jobs,
            refit=False,
        )
        gs.fit(X_tr, y_outer_train)
        best_params = gs.best_params_

        clf = GradientBoostingClassifier(random_state=RANDOM_STATE, **best_params)
        clf.fit(X_tr, y_outer_train)

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        records.append({
            "label": label,
            "fold": fold_idx,
            "auc": roc_auc_score(y_outer_test, y_prob),
            "f1": f1_score(y_outer_test, y_pred, zero_division=0),
            "precision": precision_score(y_outer_test, y_pred, zero_division=0),
            "recall": recall_score(y_outer_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_outer_test, y_pred),
        })

    return pd.DataFrame(records)
