"""
Wrappers em torno do pqfmlib para gerar as caracteristicas quanticas de
referencia. Usa sempre simulacao por vetor de estado (AerSimulator /
statevector), nunca hardware IBM real (ver Secao 5 do plano).

API padrao: fn(X_train, X_test, **kwargs) -> (X_train_feat, X_test_feat)
onde X_train/X_test sao arrays numpy ja padronizados.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

RANDOM_STATE = 42


def _check_pqfmlib():
    try:
        import pqfmlib
        return pqfmlib
    except ImportError:
        raise ImportError(
            "pqfmlib nao instalado. Execute:\n"
            "  pip install git+https://github.com/rscarmo/pqfmlib.git\n"
            "ou\n"
            "  git clone https://github.com/rscarmo/pqfmlib && "
            "pip install -e pqfmlib"
        )


def _pqfm_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    qfm_cls,
    qfm_kwargs: dict,
    mi_pairs: Optional[list] = None,
    mi_weights: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Instancia o PQFM, ajusta no X_train (para calculo de MI e atribuicao),
    transforma X_train e X_test. Segue API do pqfmlib.
    """
    import pandas as pd

    qfm = qfm_cls(**qfm_kwargs)

    # pqfmlib espera DataFrame; se nao, converte
    X_tr_df = pd.DataFrame(X_train)
    X_te_df = pd.DataFrame(X_test)

    # fit: calcula informacao mutua e atribui atributos a qubits
    qfm.fit(X_tr_df)

    # transform: executa circuito e extrai observaveis
    X_tr_feat = np.array(qfm.transform(X_tr_df))
    X_te_feat = np.array(qfm.transform(X_te_df))

    return X_tr_feat, X_te_feat


def generate_cd_ising_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    observables: str = "ZZ",
    n_qubits: Optional[int] = None,
    backend: str = "statevector",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """CDIsingProjectiveQFM do pqfmlib, observaveis <Z> e/ou <ZZ>.

    Args:
        observables: "Z" (1-corpo), "ZZ" (2-corpo), ou "ZZZ" (ambos).
        n_qubits: numero de qubits (None = um por feature).
        backend: sempre 'statevector' (noiseless).
    """
    pqfmlib = _check_pqfmlib()
    qfm_cls = pqfmlib.CDIsingProjectiveQFM

    obs_map = {
        "Z": ["Z"],
        "ZZ": ["ZZ"],
        "ZZZ": ["Z", "ZZ"],
    }
    obs = obs_map.get(observables.upper(), ["ZZ"])

    qfm_kwargs = {
        "observables": obs,
        "backend": backend,
        "random_state": RANDOM_STATE,
    }
    if n_qubits is not None:
        qfm_kwargs["n_qubits"] = n_qubits
    qfm_kwargs.update(kwargs)

    return _pqfm_transform(X_train, X_test, qfm_cls, qfm_kwargs)


def generate_heisenberg_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    observables: str = "Z",
    n_qubits: Optional[int] = None,
    backend: str = "statevector",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """HeisenbergProjectiveQFM do pqfmlib, observaveis <X>, <Y>, <Z>."""
    pqfmlib = _check_pqfmlib()
    qfm_cls = pqfmlib.HeisenbergProjectiveQFM

    obs_map = {
        "Z": ["Z"],
        "X": ["X"],
        "Y": ["Y"],
        "XYZ": ["X", "Y", "Z"],
    }
    obs = obs_map.get(observables.upper(), ["Z"])

    qfm_kwargs = {
        "observables": obs,
        "backend": backend,
        "random_state": RANDOM_STATE,
    }
    if n_qubits is not None:
        qfm_kwargs["n_qubits"] = n_qubits
    qfm_kwargs.update(kwargs)

    return _pqfm_transform(X_train, X_test, qfm_cls, qfm_kwargs)


def generate_xyz_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    variant: str = "diagonal",
    observables: str = "Z",
    n_qubits: Optional[int] = None,
    backend: str = "statevector",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """XYZProjectiveQFM do pqfmlib.

    Args:
        variant: "diagonal", "cross", "full", ou "shared_feature".
        observables: observaveis a medir ("Z", "X", "Y", "XX", "YY", "ZZ", ...).
        n_qubits: numero de qubits (None = automatico).
        backend: sempre 'statevector'.
    """
    pqfmlib = _check_pqfmlib()
    qfm_cls = pqfmlib.XYZProjectiveQFM

    qfm_kwargs = {
        "variant": variant,
        "observables": [observables] if isinstance(observables, str) else observables,
        "backend": backend,
        "random_state": RANDOM_STATE,
    }
    if n_qubits is not None:
        qfm_kwargs["n_qubits"] = n_qubits
    qfm_kwargs.update(kwargs)

    return _pqfm_transform(X_train, X_test, qfm_cls, qfm_kwargs)
