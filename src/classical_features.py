"""
As 10 tecnicas classicas de feature engineering (Secao 7 do plano).

API padrao de cada funcao:
    fn(X_train, X_test, **params) -> (X_train_feat, X_test_feat)

onde X_train/X_test sao arrays numpy ja padronizados (media 0 / var 1).
Qualquer ajuste de transformador ocorre APENAS sobre X_train (sem vazamento).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Tecnica #0 — Dequantizacao analitica
# ---------------------------------------------------------------------------

def _ry_plus(theta: float) -> np.ndarray:
    """Estado R_Y(2*theta)|+>: amplitudes [alpha0, alpha1]."""
    # |+> = [1/sqrt(2), 1/sqrt(2)]
    # R_Y(2theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
    a0 = (np.cos(theta) - np.sin(theta)) / np.sqrt(2)
    a1 = (np.sin(theta) + np.cos(theta)) / np.sqrt(2)
    return np.array([a0, a1])


def _pauli_z() -> np.ndarray:
    return np.diag([1.0, -1.0])


def _pauli_y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]])


def _pauli_x() -> np.ndarray:
    return np.array([[0, 1.0], [1.0, 0]])


def _kron(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.kron(A, B)


def _expectation(state: np.ndarray, obs: np.ndarray) -> float:
    return np.real(state.conj() @ obs @ state)


def _cd_ising_pair_state(theta_i: float, theta_j: float, J_ij: float,
                          gamma: float = 1.0) -> np.ndarray:
    """Estado 2-qubit apos circuito CD-Ising de 1 passo de Trotter no par (i,j).

    Circuito (Eq. 2.14 da dissertacao):
      |ψ₀⟩ = |+⟩|+⟩
      → R_Y(2θᵢ) ⊗ R_Y(2θⱼ)
      → exp(-iγ J_ij (Y⊗Z + Z⊗Y))
    """
    # estado inicial apos rotacoes locais
    psi_i = _ry_plus(theta_i)
    psi_j = _ry_plus(theta_j)
    psi = np.kron(psi_i, psi_j).astype(complex)  # shape (4,)

    # bloco de dois qubits: U = exp(-i gamma J_ij (YZ + ZY))
    Y = _pauli_y()
    Z = _pauli_z()
    YZ = np.kron(Y, Z)
    ZY = np.kron(Z, Y)
    H_block = gamma * J_ij * (YZ + ZY)  # 4x4 hermitiana
    # exponencial matricial
    from scipy.linalg import expm
    U = expm(-1j * H_block)
    return U @ psi


def _random_su2(rng: np.random.RandomState) -> np.ndarray:
    """Unitario SU(2) aleatorio (distribuicao de Haar)."""
    theta = rng.uniform(0, np.pi)
    phi = rng.uniform(0, 2 * np.pi)
    lam = rng.uniform(0, 2 * np.pi)
    return np.array([
        [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2),
         np.exp(1j * (phi + lam)) * np.cos(theta / 2)],
    ], dtype=complex)


def _heisenberg_pair_state(eta_j: float, gamma: float = 1.0,
                            pair_idx: int = 0, seed: int = 42) -> np.ndarray:
    """Estado 2-qubit apos bloco Heisenberg (Eq. 2.19).

    Usa unitarios locais aleatorios (SU(2) com semente fixa por par, igual
    ao pqfmlib) como estado inicial, depois aplica exp(-i gamma eta_j (XX+YY+ZZ)).
    O estado resultante depende de eta_j (que carrega os dados via tanh).
    """
    rng = np.random.RandomState(seed + pair_idx * 1337)
    U1 = _random_su2(rng)
    U2 = _random_su2(rng)
    # estado inicial: U1|0⟩ ⊗ U2|0⟩
    psi0 = np.kron(U1[:, 0], U2[:, 0])

    X = _pauli_x()
    Y = _pauli_y()
    Z = _pauli_z()
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    ZZ = np.kron(Z, Z)
    H_block = gamma * eta_j * (XX + YY + ZZ)
    from scipy.linalg import expm
    U = expm(-1j * H_block)
    return U @ psi0


def _xyz_pair_state(
    theta_i_x: float, theta_i_y: float, theta_i_z: float,
    theta_j_x: float, theta_j_y: float, theta_j_z: float,
    J: dict, gamma: float = 1.0,
) -> np.ndarray:
    """Estado 2-qubit apos bloco XYZ diagonal (Eq. 2.28).

    H_diag = h^x X_i + h^y Y_i + h^z Z_i (termo local qubit i)
            + h^x X_j + h^y Y_j + h^z Z_j (termo local qubit j)
            + J^xx X_i X_j + J^yy Y_i Y_j + J^zz Z_i Z_j

    Campos locais: h^a_i = theta_i_a (feature padronizada / eixo a)
    Acoplamentos: J['xx'], J['yy'], J['zz'] = informacao mutua (par)
    """
    X = _pauli_x()
    Y = _pauli_y()
    Z = _pauli_z()
    I2 = np.eye(2)

    H = (theta_i_x * np.kron(X, I2) + theta_i_y * np.kron(Y, I2) +
         theta_i_z * np.kron(Z, I2) +
         theta_j_x * np.kron(I2, X) + theta_j_y * np.kron(I2, Y) +
         theta_j_z * np.kron(I2, Z) +
         J.get("xx", 0) * np.kron(X, X) +
         J.get("yy", 0) * np.kron(Y, Y) +
         J.get("zz", 0) * np.kron(Z, Z))

    from scipy.linalg import expm
    U = expm(-1j * gamma * H)
    psi0 = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)  # |++⟩
    return U @ psi0


def _z1_z2_expectations(psi: np.ndarray) -> tuple[float, float, float]:
    """Retorna <Z1>, <Z2>, <Z1 Z2> para estado de 2 qubits."""
    Z = _pauli_z()
    I2 = np.eye(2)
    z1 = _expectation(psi, np.kron(Z, I2))
    z2 = _expectation(psi, np.kron(I2, Z))
    zz = _expectation(psi, np.kron(Z, Z))
    return z1, z2, zz


def _xyz_local_expectations(psi: np.ndarray) -> dict[str, float]:
    """Retorna <X1>, <Y1>, <Z1>, <X2>, <Y2>, <Z2> para estado de 2 qubits."""
    X = _pauli_x()
    Y = _pauli_y()
    Z = _pauli_z()
    I2 = np.eye(2)
    out = {}
    for name, op in [("X", X), ("Y", Y), ("Z", Z)]:
        out[f"{name}1"] = _expectation(psi, np.kron(op, I2))
        out[f"{name}2"] = _expectation(psi, np.kron(I2, op))
    return out


def exact_dequantization(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_pairs: list[tuple[int, int]],
    mi_weights: Optional[list[float]] = None,
    map_type: str = "cd_ising",
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #0 — formulas fechadas equivalentes ao PQFM.

    Recalcula em NumPy puro os mesmos valores esperados que o PQFM
    correspondente mediria (sem simulacao de circuito quântico).

    Args:
        X_train, X_test: arrays (n_samples, n_features), ja padronizados.
        mi_pairs: lista de pares (i, j) de indices de colunas, selecionados
                  por informacao mutua (mesmos pares usados pelo pqfmlib).
        mi_weights: pesos J_ij para cada par (informacao mutua). Se None,
                    usa J_ij = 1.0 para todos os pares.
        map_type: "cd_ising" ou "heisenberg" ou "xyz_diagonal".
        gamma: parametro de escala do bloco de dois qubits.
    """
    if mi_weights is None:
        mi_weights = [1.0] * len(mi_pairs)

    def transform(X: np.ndarray) -> np.ndarray:
        rows = []
        for x in X:
            feats = []
            for k, ((i, j), J) in enumerate(zip(mi_pairs, mi_weights)):
                theta_i = float(x[i])
                theta_j = float(x[j])
                if map_type == "cd_ising":
                    psi = _cd_ising_pair_state(theta_i, theta_j, J, gamma)
                    z1, z2, zz = _z1_z2_expectations(psi)
                    feats.extend([z1, z2, zz])
                elif map_type == "heisenberg":
                    # η_j = 2π tanh(h_j / 3), Eq. 2.18 da dissertacao
                    h_j = (abs(theta_i) + abs(theta_j)) / 2.0
                    eta_j = 2 * np.pi * np.tanh(h_j / 3.0)
                    psi = _heisenberg_pair_state(eta_j * J, gamma, pair_idx=k)
                    exp_vals = _xyz_local_expectations(psi)
                    feats.extend(list(exp_vals.values()))
                elif map_type == "xyz_diagonal":
                    J_dict = {"xx": J, "yy": J, "zz": J}
                    psi = _xyz_pair_state(
                        theta_i, theta_i, theta_i,
                        theta_j, theta_j, theta_j,
                        J_dict, gamma,
                    )
                    exp_vals = _xyz_local_expectations(psi)
                    feats.extend(list(exp_vals.values()))
                else:
                    raise ValueError(f"map_type desconhecido: {map_type}")
            rows.append(feats)
        return np.array(rows, dtype=float)

    return transform(X_train), transform(X_test)


# ---------------------------------------------------------------------------
# Tecnica #1 — Aritmetica par a par
# ---------------------------------------------------------------------------

def pairwise_arithmetic(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #1 — soma, diferenca, produto, razao nos pares de maior IM."""
    def transform(X: np.ndarray) -> np.ndarray:
        cols = []
        for i, j in mi_pairs:
            a, b = X[:, i], X[:, j]
            cols.append(a + b)
            cols.append(a - b)
            cols.append(a * b)
            # razao: evitar divisao por zero
            ratio = np.where(np.abs(b) > 1e-8, a / b, 0.0)
            cols.append(ratio)
        return np.column_stack(cols) if cols else np.empty((X.shape[0], 0))

    return transform(X_train), transform(X_test)


# ---------------------------------------------------------------------------
# Tecnica #2 — Polinomio grau 2
# ---------------------------------------------------------------------------

def polynomial_degree2(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #2 — sklearn.preprocessing.PolynomialFeatures(degree=2)."""
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr = poly.fit_transform(X_train)
    X_te = poly.transform(X_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Tecnica #3 — Random Fourier Features (RBF)
# ---------------------------------------------------------------------------

def random_fourier_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #3 — sklearn.kernel_approximation.RBFSampler."""
    rff = RBFSampler(n_components=n_components, random_state=RANDOM_STATE)
    X_tr = rff.fit_transform(X_train)
    X_te = rff.transform(X_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Tecnica #4 — Nystroem RBF
# ---------------------------------------------------------------------------

def nystroem_rbf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #4 — Nystroem(kernel='rbf')."""
    ny = Nystroem(kernel="rbf", n_components=n_components, random_state=RANDOM_STATE)
    X_tr = ny.fit_transform(X_train)
    X_te = ny.transform(X_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Tecnica #5 — Nystroem polinomial
# ---------------------------------------------------------------------------

def nystroem_poly(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
    degree: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #5 — Nystroem(kernel='poly', degree=degree)."""
    ny = Nystroem(
        kernel="poly", degree=degree, n_components=n_components,
        random_state=RANDOM_STATE,
    )
    X_tr = ny.fit_transform(X_train)
    X_te = ny.transform(X_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Tecnica #6 — Kernel PCA (RBF)
# ---------------------------------------------------------------------------

def kernel_pca_rbf(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #6 — KernelPCA(kernel='rbf')."""
    kpca = KernelPCA(
        n_components=n_components, kernel="rbf",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    X_tr = kpca.fit_transform(X_train)
    X_te = kpca.transform(X_test)
    return X_tr, X_te


# ---------------------------------------------------------------------------
# Tecnica #7 — Projecoes aleatorias nao-lineares (ELM)
# ---------------------------------------------------------------------------

def random_nonlinear_projection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
    activation: str = "tanh",
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #7 — tanh/sin de combinacao linear aleatoria (ELM).

    Pesos W e bias b fixados por seed (sem fit em X_train — sem vazamento).
    """
    rng = np.random.RandomState(RANDOM_STATE)
    n_in = X_train.shape[1]
    W = rng.randn(n_in, n_components)
    b = rng.randn(n_components)

    def transform(X: np.ndarray) -> np.ndarray:
        Z = X @ W + b
        if activation == "tanh":
            return np.tanh(Z)
        elif activation == "sin":
            return np.sin(Z)
        else:
            return np.tanh(Z)

    return transform(X_train), transform(X_test)


# ---------------------------------------------------------------------------
# Tecnica #8 — Autoencoder raso
# ---------------------------------------------------------------------------

def shallow_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    latent_dim: int = 20,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #8 — autoencoder raso (bottleneck = latent_dim).

    Treina apenas no X_train (sem vazamento). Se torch+CUDA disponivel e
    device='cuda', usa GPU para acelerar o treino.
    """
    try:
        import torch
        import torch.nn as nn

        use_cuda = device == "cuda" and torch.cuda.is_available()
        dev = torch.device("cuda" if use_cuda else "cpu")

        n_in = X_train.shape[1]
        hidden = max(latent_dim * 2, n_in // 2)

        model = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, latent_dim), nn.Tanh(),
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, n_in),
        ).to(dev)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(model(X_t), X_t)
            loss.backward()
            optimizer.step()

        # extrai representacao do gargalo (saida da 3a camada)
        encoder = nn.Sequential(*list(model.children())[:4])
        model.eval()
        encoder.eval()
        with torch.no_grad():
            X_tr_feat = encoder(X_t).cpu().numpy()
            X_te_feat = encoder(
                torch.tensor(X_test, dtype=torch.float32).to(dev)
            ).cpu().numpy()
        return X_tr_feat, X_te_feat

    except ImportError:
        pass

    # fallback: MLPRegressor como autoencoder
    warnings.warn("PyTorch nao disponivel; usando MLPRegressor como autoencoder.")
    n_in = X_train.shape[1]
    hidden = max(latent_dim * 2, n_in // 2)
    ae = MLPRegressor(
        hidden_layer_sizes=(hidden, latent_dim, hidden),
        activation="tanh",
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    ae.fit(X_train, X_train)

    # extrai saida da camada do gargalo (nao ha acesso direto no sklearn;
    # usa o codigo forward ate a 2a camada)
    def _encode(ae, X):
        X_cur = X
        n_layers = len(ae.coefs_)
        # encoder: ate a camada do meio (latent_dim)
        mid = n_layers // 2
        for i in range(mid):
            X_cur = np.tanh(X_cur @ ae.coefs_[i] + ae.intercepts_[i])
        return X_cur

    return _encode(ae, X_train), _encode(ae, X_test)


# ---------------------------------------------------------------------------
# Tecnica #9 — Discretizacao + cruzamento (binning)
# ---------------------------------------------------------------------------

def binning_and_cross(
    X_train: np.ndarray,
    X_test: np.ndarray,
    mi_pairs: list[tuple[int, int]],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Tecnica #9 — KBinsDiscretizer + cruzamento dos bins nos pares de maior IM."""
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    X_tr_bins = kbd.fit_transform(X_train)
    X_te_bins = kbd.transform(X_test)

    def cross_features(X_bins: np.ndarray) -> np.ndarray:
        cols = []
        for i, j in mi_pairs:
            # produto dos indices de bin (interacao discreta)
            cols.append(X_bins[:, i] * n_bins + X_bins[:, j])
            # tambem inclui os bins individuais dos pares
            cols.append(X_bins[:, i])
            cols.append(X_bins[:, j])
        return np.column_stack(cols) if cols else X_bins

    return cross_features(X_tr_bins), cross_features(X_te_bins)


# ---------------------------------------------------------------------------
# Utilidades: calculo de pares por informacao mutua
# ---------------------------------------------------------------------------

def compute_mi_pairs(
    X: np.ndarray,
    y: np.ndarray,
    n_pairs: Optional[int] = None,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Calcula informacao mutua entre pares de atributos (features x features)
    e retorna os top-n_pairs pares ordenados por IM decrescente.

    Usada para replicar a selecao do pqfmlib quando ele nao esta disponivel.
    """
    from sklearn.feature_selection import mutual_info_classif
    from itertools import combinations

    n_feat = X.shape[1]
    if n_pairs is None:
        n_pairs = n_feat  # um par por feature (aproximadamente)

    # IM de cada feature com o alvo (para referencias)
    mi_with_y = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # IM entre pares de features (estimativa de Pearson normalizada como proxy)
    # Para datasets grandes, IM real par-a-par seria caro; usamos |correlacao|
    corr = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(corr, 0)  # ignora self-correlation

    pairs_scores = []
    for i, j in combinations(range(n_feat), 2):
        score = corr[i, j]
        pairs_scores.append(((i, j), score))

    pairs_scores.sort(key=lambda x: x[1], reverse=True)
    top = pairs_scores[:n_pairs]
    mi_pairs = [p for p, _ in top]
    mi_weights = [s for _, s in top]
    return mi_pairs, mi_weights
