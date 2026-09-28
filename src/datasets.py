"""
Loaders e pre-processamento padronizado para as 4 bases do estudo.

Cada loader devolve (X, y) como DataFrame/Series brutos, SEM padronizacao.
A padronizacao e feita dentro do pipeline de CV aninhada (src/pipeline.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_cardiac_disease() -> tuple[pd.DataFrame, pd.Series]:
    """Cleveland Heart Disease (UCI id 45). 303 amostras x 13 atributos.

    Alvo binarizado: presenca (1) vs ausencia (0) de doenca cardiaca.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        X = ds.data.features.copy()
        y = ds.data.targets.iloc[:, 0].copy()
    except Exception:
        # fallback: download direto do UCI
        import io, urllib.request
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","num"]
        raw = urllib.request.urlopen(url).read().decode()
        df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
        y = df["num"]
        X = df.drop(columns="num")

    # binariza alvo (0 = sem doenca, 1 = com doenca)
    y = (y > 0).astype(int)
    y.name = "target"

    # substitui "?" / NaN que possam ter chegado como string
    X = X.replace("?", np.nan)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    assert X.shape == (303, 13), f"Shape inesperado: {X.shape}"
    return X, y


def load_breast_cancer_pathology() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Diagnostic Breast Cancer (WDBC). 569 amostras x 30 atributos."""
    from sklearn.datasets import load_breast_cancer
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.frame.drop(columns="target")
    y = bunch.target
    assert X.shape == (569, 30), f"Shape inesperado: {X.shape}"
    return X, y


def load_breast_cancer_radiology() -> tuple[pd.DataFrame, pd.Series]:
    """I-SPY1 via radMLBench. 161 amostras x 370 atributos radiomicos."""
    import radMLBench
    # lista os datasets disponiveis e encontra o I-SPY1
    datasets = radMLBench.listDatasets()
    ispy_key = None
    for k in datasets:
        if "ispy" in k.lower() or "spy" in k.lower():
            ispy_key = k
            break
    if ispy_key is None:
        # tenta chaves conhecidas
        for candidate in ["ISPY1", "I-SPY1", "ispy1", "Breast_MRI_ISPY1"]:
            if candidate in datasets:
                ispy_key = candidate
                break
    if ispy_key is None:
        raise RuntimeError(
            f"Dataset I-SPY1 nao encontrado em radMLBench. "
            f"Datasets disponíveis: {datasets}"
        )
    X, y = radMLBench.loadData(ispy_key, return_X_y=True)
    X = pd.DataFrame(X)
    y = pd.Series(y, name="target", dtype=int)
    assert X.shape[1] == 370, f"Numero de atributos inesperado: {X.shape[1]}"
    assert X.shape[0] == 161, f"Numero de amostras inesperado: {X.shape[0]}"
    return X, y


def load_molecular_toxicity() -> tuple[pd.DataFrame, pd.Series]:
    """Molecular Toxicity CRY1 — requer resolucao do Risco R1 (Secao 2.1).

    Tenta carregar de data/raw/Toxicity_preprocessed_shuffled.csv.
    Se nao encontrado, lanca FileNotFoundError com instrucoes.
    """
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    candidates = [
        os.path.join(data_dir, "Toxicity_preprocessed_shuffled.csv"),
        os.path.join(data_dir, "toxicity.csv"),
        os.path.join(data_dir, "molecular_toxicity.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # ultima coluna e o alvo (convencao do pqfmlib)
            y = df.iloc[:, -1].astype(int)
            X = df.iloc[:, :-1]
            y.name = "target"
            return X, y

    raise FileNotFoundError(
        "Dataset Molecular Toxicity nao encontrado. "
        "Ver Risco R1 (Secao 2.1 do plano): colocar o arquivo CSV em "
        "data/raw/Toxicity_preprocessed_shuffled.csv"
    )
