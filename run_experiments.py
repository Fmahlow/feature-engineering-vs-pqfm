"""
Script principal de execucao dos experimentos, em etapas.

Uso:
    python run_experiments.py --step 0          # checkpoint de reprodutibilidade
    python run_experiments.py --step 1          # Cardiac Disease
    python run_experiments.py --step 2          # Breast Cancer Pathology (WDBC)
    python run_experiments.py --step 3          # Breast Cancer Radiology (I-SPY1)
    python run_experiments.py --step 4          # Molecular Toxicity (CRY1)
    python run_experiments.py --step all        # todas as etapas em sequencia

    python run_experiments.py --step 1 --quick  # 3 repeticoes externas (debug)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from src import datasets, pipeline, stats
from src.classical_features import (
    binning_and_cross,
    compute_mi_pairs,
    exact_dequantization,
    kernel_pca_rbf,
    nystroem_poly,
    nystroem_rbf,
    pairwise_arithmetic,
    polynomial_degree2,
    random_fourier_features,
    random_nonlinear_projection,
    shallow_autoencoder,
)

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame, name: str):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  -> salvo: {path}")


def timed(label: str, fn):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"  [{label}] {elapsed:.1f}s")
    return result


def make_feature_fn(technique_fn, **kwargs):
    """Encapsula uma tecnica classica como feature_fn(X_train, X_test)."""
    def fn(X_tr, X_te):
        return technique_fn(X_tr, X_te, **kwargs)
    return fn


def n_quantum_features(dataset_name: str) -> int:
    """Numero alvo de features geradas (aproximado), para parear dimensionalidade.
    Baseado nas configuracoes da dissertacao (Tabela 4.1 / 4.2).
    """
    return {
        "cardiac": 26,    # 13 qubits -> ~26 observaveis (Z + ZZ)
        "wdbc": 60,       # 20 qubits, SHAP 30 features
        "ispy1": 370,     # 370 atributos -> ~370 features
        "toxicity": 156,  # 156 atributos usados
    }.get(dataset_name, 50)


def run_all_techniques(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    n_repeats_outer: int = 10,
    include_quantum: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roda baseline + todas as tecnicas classicas (e quanticas se disponivel)
    para um dataset. Retorna (resultados_brutos, tabela_estatistica).
    """
    n_q = n_quantum_features(dataset_name)
    from sklearn.impute import SimpleImputer
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=int)

    # imputa NaN para calcular pares de IM (proxy global; a imputacao por fold
    # ocorre dentro do run_nested_cv, sem vazamento)
    X_imp = SimpleImputer(strategy="median").fit_transform(X_arr)

    # calcula pares de maior IM (usados pelas tecnicas #0, #1, #9)
    n_pairs = min(n_q // 3, X_arr.shape[1])
    mi_pairs, mi_weights = compute_mi_pairs(X_imp, y_arr, n_pairs=n_pairs)
    print(f"  Pares IM calculados: {len(mi_pairs)} pares")

    # passa n_repeats diretamente para run_nested_cv via monkey-patch temporario
    import src.pipeline as _pipeline_mod
    original_repeats = _pipeline_mod.N_REPEATS
    _pipeline_mod.N_REPEATS = n_repeats_outer

    baseline_label = f"{dataset_name}_baseline"

    configurations = [
        # (label, feature_fn, mode)
        (baseline_label, None, "baseline"),
        # Tecnica #0 - dequantizacao
        (f"{dataset_name}_deq_cdising_iso",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="cd_ising"),
         "isolated"),
        (f"{dataset_name}_deq_cdising_hyb",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="cd_ising"),
         "hybrid"),
        (f"{dataset_name}_deq_heisen_iso",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="heisenberg"),
         "isolated"),
        (f"{dataset_name}_deq_heisen_hyb",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="heisenberg"),
         "hybrid"),
        (f"{dataset_name}_deq_xyz_iso",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="xyz_diagonal"),
         "isolated"),
        (f"{dataset_name}_deq_xyz_hyb",
         make_feature_fn(exact_dequantization, mi_pairs=mi_pairs,
                         mi_weights=mi_weights, map_type="xyz_diagonal"),
         "hybrid"),
        # Tecnica #1
        (f"{dataset_name}_pairwise_iso",
         make_feature_fn(pairwise_arithmetic, mi_pairs=mi_pairs), "isolated"),
        (f"{dataset_name}_pairwise_hyb",
         make_feature_fn(pairwise_arithmetic, mi_pairs=mi_pairs), "hybrid"),
        # Tecnica #2 (skip para datasets com muitos atributos: memoria explosiva)
        *([
            (f"{dataset_name}_poly2_iso",
             make_feature_fn(polynomial_degree2), "isolated"),
            (f"{dataset_name}_poly2_hyb",
             make_feature_fn(polynomial_degree2), "hybrid"),
        ] if X_arr.shape[1] <= 50 else []),
        # Tecnica #3
        (f"{dataset_name}_rff_iso",
         make_feature_fn(random_fourier_features, n_components=n_q), "isolated"),
        (f"{dataset_name}_rff_hyb",
         make_feature_fn(random_fourier_features, n_components=n_q), "hybrid"),
        # Tecnica #4
        (f"{dataset_name}_nystroem_rbf_iso",
         make_feature_fn(nystroem_rbf, n_components=n_q), "isolated"),
        (f"{dataset_name}_nystroem_rbf_hyb",
         make_feature_fn(nystroem_rbf, n_components=n_q), "hybrid"),
        # Tecnica #5
        (f"{dataset_name}_nystroem_poly_iso",
         make_feature_fn(nystroem_poly, n_components=n_q, degree=2), "isolated"),
        (f"{dataset_name}_nystroem_poly_hyb",
         make_feature_fn(nystroem_poly, n_components=n_q, degree=2), "hybrid"),
        # Tecnica #6
        (f"{dataset_name}_kpca_iso",
         make_feature_fn(kernel_pca_rbf, n_components=n_q), "isolated"),
        (f"{dataset_name}_kpca_hyb",
         make_feature_fn(kernel_pca_rbf, n_components=n_q), "hybrid"),
        # Tecnica #7
        (f"{dataset_name}_elm_iso",
         make_feature_fn(random_nonlinear_projection, n_components=n_q), "isolated"),
        (f"{dataset_name}_elm_hyb",
         make_feature_fn(random_nonlinear_projection, n_components=n_q), "hybrid"),
        # Tecnica #8
        (f"{dataset_name}_ae_iso",
         make_feature_fn(shallow_autoencoder, latent_dim=n_q, device="cuda"),
         "isolated"),
        (f"{dataset_name}_ae_hyb",
         make_feature_fn(shallow_autoencoder, latent_dim=n_q, device="cuda"),
         "hybrid"),
        # Tecnica #9
        (f"{dataset_name}_bins_iso",
         make_feature_fn(binning_and_cross, mi_pairs=mi_pairs), "isolated"),
        (f"{dataset_name}_bins_hyb",
         make_feature_fn(binning_and_cross, mi_pairs=mi_pairs), "hybrid"),
    ]

    # Tecnicas quanticas (se pqfmlib disponivel)
    if include_quantum:
        try:
            from src import quantum_features as qf
            qf._check_pqfmlib()
            configurations += [
                (f"{dataset_name}_cdising_zz_iso",
                 make_feature_fn(qf.generate_cd_ising_features, observables="ZZ"),
                 "isolated"),
                (f"{dataset_name}_cdising_zz_hyb",
                 make_feature_fn(qf.generate_cd_ising_features, observables="ZZ"),
                 "hybrid"),
                (f"{dataset_name}_heisenberg_z_iso",
                 make_feature_fn(qf.generate_heisenberg_features, observables="Z"),
                 "isolated"),
                (f"{dataset_name}_heisenberg_z_hyb",
                 make_feature_fn(qf.generate_heisenberg_features, observables="Z"),
                 "hybrid"),
                (f"{dataset_name}_xyz_diag_z_iso",
                 make_feature_fn(qf.generate_xyz_features, variant="diagonal",
                                 observables="Z"),
                 "isolated"),
                (f"{dataset_name}_xyz_diag_z_hyb",
                 make_feature_fn(qf.generate_xyz_features, variant="diagonal",
                                 observables="Z"),
                 "hybrid"),
            ]
        except ImportError:
            print("  [!] pqfmlib nao disponivel — pulando tecnicas quanticas.")

    all_results = []
    for label, feat_fn, mode in configurations:
        print(f"  Rodando: {label} ({mode}) ...")
        try:
            df = timed(label, lambda: pipeline.run_nested_cv(
                X, y, label=label, feature_fn=feat_fn, mode=mode
            ))
            all_results.append(df)
        except Exception as e:
            print(f"  [ERRO] {label}: {e}")
            continue

    _pipeline_mod.N_REPEATS = original_repeats

    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)
    stats_df = stats.build_stats_table(results_df, baseline_label)
    return results_df, stats_df


# ---------------------------------------------------------------------------
# Etapa 0 — Checkpoint de reprodutibilidade
# ---------------------------------------------------------------------------

def step0_checkpoint():
    """Roda APENAS o baseline nas 3 bases disponiveis (sem CRY1) e compara
    com os valores da dissertacao (Tabelas 4.1, 4.2, 4.3).
    """
    print("\n=== ETAPA 0: Checkpoint de reprodutibilidade ===")

    # Valores de referencia da dissertacao (AUC media, coluna 'Classico')
    ref_auc = {
        "cardiac": 0.8784,
        "wdbc": 0.9847,
        "ispy1": 0.5315,
    }

    for dname, loader in [
        ("cardiac", datasets.load_cardiac_disease),
        ("wdbc", datasets.load_breast_cancer_pathology),
        ("ispy1", datasets.load_breast_cancer_radiology),
    ]:
        print(f"\n--- {dname.upper()} ---")
        try:
            X, y = loader()
            print(f"  Shape: {X.shape}, classes: {y.value_counts().to_dict()}")
        except Exception as e:
            print(f"  [ERRO ao carregar] {e}")
            continue

        label = f"{dname}_baseline"
        df = timed(label, lambda: pipeline.run_nested_cv(X, y, label=label))
        auc_mean = df["auc"].mean()
        ref = ref_auc[dname]
        diff = auc_mean - ref
        print(f"  AUC obtida:    {auc_mean:.4f}")
        print(f"  AUC dissertacao: {ref:.4f}")
        print(f"  Diferenca:     {diff:+.4f}  {'OK' if abs(diff) < 0.03 else 'ATENCAO'}")
        save(df, f"etapa0_{dname}_baseline")

    print("\n=== Etapa 0 concluida ===")


# ---------------------------------------------------------------------------
# Etapas 1–4
# ---------------------------------------------------------------------------

def step_run(step: int, quick: bool = False):
    loaders = {
        1: ("cardiac", datasets.load_cardiac_disease),
        2: ("wdbc", datasets.load_breast_cancer_pathology),
        3: ("ispy1", datasets.load_breast_cancer_radiology),
        4: ("toxicity", datasets.load_molecular_toxicity),
    }
    names = {1: "Cardiac Disease", 2: "Breast Cancer Pathology (WDBC)",
             3: "Breast Cancer Radiology (I-SPY1)", 4: "Molecular Toxicity (CRY1)"}

    dname, loader = loaders[step]
    print(f"\n=== ETAPA {step}: {names[step]} ===")

    try:
        X, y = loader()
    except FileNotFoundError as e:
        print(f"[ERRO] {e}")
        return

    print(f"Shape: {X.shape}, classes: {y.value_counts().to_dict()}")

    n_repeats = 3 if quick else 10
    results_df, stats_df = run_all_techniques(
        X, y, dname, n_repeats_outer=n_repeats
    )

    if results_df.empty:
        print("  Sem resultados para salvar.")
        return

    save(results_df, f"etapa{step}_{dname}_resultados")
    save(stats_df, f"etapa{step}_{dname}_estatistica_pareada")

    # imprime tabela de sintese (AUC)
    auc_summary = (
        results_df.groupby("label")["auc"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    print("\n--- Sintese AUC ---")
    print(auc_summary.to_string())

    if not stats_df.empty:
        print("\n--- Status estatistico (AUC) ---")
        auc_stats = stats_df[stats_df["metric"] == "auc"][
            ["label", "delta_mean", "ci_low", "ci_high", "p_wilcoxon", "status"]
        ].sort_values("delta_mean", ascending=False)
        print(auc_stats.to_string(index=False))

    print(f"\n=== Etapa {step} concluida ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="0",
                        help="Etapa a rodar: 0, 1, 2, 3, 4, ou 'all'")
    parser.add_argument("--quick", action="store_true",
                        help="Modo rapido: 3 repeticoes externas (para debug)")
    args = parser.parse_args()

    if args.step == "0":
        step0_checkpoint()
    elif args.step == "all":
        step0_checkpoint()
        for s in [1, 2, 3, 4]:
            step_run(s, quick=args.quick)
    else:
        s = int(args.step)
        if s == 0:
            step0_checkpoint()
        else:
            step_run(s, quick=args.quick)
