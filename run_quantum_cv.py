"""
Executa nested CV com features quânticas pré-computadas pelo pqfmlib.

Fluxo:
  1. Carrega dataset
  2. Escreve CSV pqfmlib-compativel (X padronizado globalmente + target)
  3. Chama .venv311/bin/python precompute_quantum.py via subprocess
  4. Carrega NPZ com Xq_all_raw (n_samples, n_q_features)
  5. Roda nested CV via run_nested_cv_precomputed (usa train_idx/test_idx)
     - modo "isolated": apenas Xq
     - modo "hybrid": concat(X_original_sc, Xq)
  6. Salva results/etapa{step}_{dataset}_quantum_resultados.csv

Nota: Xq e computado sobre o dataset INTEIRO (abordagem transductiva) — os
pares de qubits e pesos J sao calculados com todas as amostras. Isso e
consistente com a dissertacao de Carmo (2026) mas introduz vazamento de
informacao leve no calculo de MI. Para evitar leakage total, usariamos
precomputacao por fold (muito mais lento).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_cardiac_disease, load_breast_cancer_pathology, load_breast_cancer_radiology
from src.pipeline import run_nested_cv_precomputed
from src.stats import build_stats_table
import src.pipeline as _pipeline_mod

VENV311 = str(ROOT / ".venv311" / "bin" / "python")
PRECOMPUTE_SCRIPT = str(ROOT / "precompute_quantum.py")
QUANTUM_DIR = str(ROOT / "results" / "quantum_precomputed")
RESULTS_DIR = str(ROOT / "results")

Q_ENC = {"cardiac": 13, "wdbc": 20, "ispy1": 20}

DATASETS = {
    1: ("cardiac", "Cardiac Disease — Cleveland", load_cardiac_disease),
    2: ("wdbc", "Breast Cancer WDBC", load_breast_cancer_pathology),
    3: ("ispy1", "Breast Cancer I-SPY1", load_breast_cancer_radiology),
}

MAP_NAMES = ["cdising", "heisenberg", "xyz_diag"]
LABEL_DISPLAY = {
    "cdising":    "pqfm_cdising",
    "heisenberg": "pqfm_heisenberg",
    "xyz_diag":   "pqfm_xyz",
}


def prepare_pqfm_csv(X_arr: np.ndarray, y_arr: np.ndarray, path: str) -> np.ndarray:
    """Padroniza X (globalmente) e salva CSV pqfmlib (ultima coluna = target)."""
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_arr)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    df = pd.DataFrame(X_sc, columns=[f"f{i}" for i in range(X_sc.shape[1])])
    df["target"] = y_arr.astype(float)
    df.to_csv(path, index=False)
    return X_sc


def need_precompute(dname: str, overwrite: bool) -> list[str]:
    missing = []
    for m in MAP_NAMES:
        f = Path(QUANTUM_DIR) / f"{dname}_{m}.npz"
        if not f.exists() or overwrite:
            missing.append(m)
    return missing


def run_precompute(dname: str, csv_path: str, q_enc: int, overwrite: bool) -> bool:
    missing = need_precompute(dname, overwrite)
    if not missing:
        print(f"  [pqfm] NPZs ja existem para {dname}.")
        return True

    print(f"  [pqfm] Computando mapas {missing} para {dname}...", flush=True)
    cmd = [
        VENV311, PRECOMPUTE_SCRIPT,
        "--dataset", dname,
        "--csv", csv_path,
        "--out", QUANTUM_DIR,
        "--q_enc", str(q_enc),
    ]
    if overwrite:
        cmd.append("--overwrite")

    t0 = time.time()
    ret = subprocess.run(cmd).returncode
    print(f"  [pqfm] exit={ret}, {time.time()-t0:.0f}s", flush=True)
    return ret == 0


def run_quantum_step(step: int, dname: str, title: str, loader,
                     n_repeats: int, overwrite_npz: bool):
    print(f"\n=== Etapa {step} (QUANTUM): {title} ===", flush=True)
    _pipeline_mod.N_REPEATS = n_repeats

    X, y = loader()
    X_arr = X.values if hasattr(X, "values") else np.asarray(X, dtype=float)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y, dtype=int)

    Path(QUANTUM_DIR).mkdir(parents=True, exist_ok=True)
    csv_path = str(Path(QUANTUM_DIR) / f"_input_{dname}.csv")
    prepare_pqfm_csv(X_arr, y_arr, csv_path)

    q_enc = Q_ENC.get(dname, 20)
    ok = run_precompute(dname, csv_path, q_enc, overwrite_npz)
    if not ok:
        print(f"  ERRO na pre-computacao. Abortando etapa {step}.")
        return

    all_records: list[pd.DataFrame] = []

    for map_name in MAP_NAMES:
        npz_path = Path(QUANTUM_DIR) / f"{dname}_{map_name}.npz"
        if not npz_path.exists():
            print(f"  [{map_name}] NPZ nao encontrado, pulando.")
            continue

        Xq_full = np.load(npz_path)["Xq"]
        if Xq_full.shape[0] != X_arr.shape[0]:
            print(f"  [{map_name}] AVISO: Xq tem {Xq_full.shape[0]} linhas, esperado {X_arr.shape[0]}. Pulando.")
            continue

        print(f"  [{map_name}] Xq shape: {Xq_full.shape}", flush=True)
        base_label = LABEL_DISPLAY[map_name]

        for mode, suffix in [("isolated", "iso"), ("hybrid", "hyb")]:
            label = f"{dname}_{base_label}_{suffix}"
            print(f"  Rodando: {label} ...", flush=True)
            t0 = time.time()
            try:
                df_fold = run_nested_cv_precomputed(
                    X_arr, y_arr, Xq_full, label=label, mode=mode, n_jobs=-1
                )
                all_records.append(df_fold)
                print(f"  [{label}] {time.time()-t0:.0f}s  AUC={df_fold['auc'].mean():.4f}", flush=True)
            except Exception as e:
                print(f"  [{label}] ERRO: {e}", flush=True)
                import traceback; traceback.print_exc()

    if not all_records:
        print("  Nenhum resultado quântico gerado.")
        return

    df_quantum = pd.concat(all_records, ignore_index=True)
    out_q = Path(RESULTS_DIR) / f"etapa{step}_{dname}_quantum_resultados.csv"
    df_quantum.to_csv(out_q, index=False)
    print(f"  -> salvo: {out_q}")

    # estatistica pareada combinando classico + quantum vs baseline
    classic_path = Path(RESULTS_DIR) / f"etapa{step}_{dname}_resultados.csv"
    if classic_path.exists():
        df_all = pd.concat([pd.read_csv(classic_path), df_quantum], ignore_index=True)
        baseline_label = f"{dname}_baseline"
        df_stat = build_stats_table(df_all, baseline_label)
        stat_path = Path(RESULTS_DIR) / f"etapa{step}_{dname}_quantum_estatistica_pareada.csv"
        df_stat.to_csv(stat_path, index=False)
        print(f"  -> salvo: {stat_path}")

    auc_q = df_quantum.groupby("label")["auc"].mean().sort_values(ascending=False)
    print("\n--- AUC medio (quantum) ---")
    print(auc_q.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--overwrite_npz", action="store_true")
    args = parser.parse_args()

    for step in args.steps:
        if step not in DATASETS:
            continue
        dname, title, loader = DATASETS[step]
        run_quantum_step(step, dname, title, loader, args.n_repeats, args.overwrite_npz)

    print("\n=== run_quantum_cv.py concluido ===")


if __name__ == "__main__":
    main()
