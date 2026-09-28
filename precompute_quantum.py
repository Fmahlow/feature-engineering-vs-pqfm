"""
Pré-computa features quânticas (CD-Ising, Heisenberg, XYZ) usando pqfmlib.

Chamado pelo venv311 (Python >= 3.10 requerido pela lib):
  .venv311/bin/python precompute_quantum.py --dataset cardiac --csv /tmp/cardiac.csv --out results/quantum_precomputed/

Salva por dataset e mapa:
  results/quantum_precomputed/{dataset}_{map}.npz
    arrays: Xq (n_samples, n_q_features)

A computação é feita UMA VEZ para o dataset inteiro (abordagem transductiva,
consistente com a dissertação de Carmo 2026).  O nested CV posterior (run_quantum_cv.py)
fatia as linhas por fold sem re-computar as features quânticas.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


Q_ENC_DEFAULT = 20   # numero de qubits; limitado para viabilidade do statevector


def run_qfm(cls, kwargs: dict, csv_path: str, data_dir: str, name_file: str) -> np.ndarray:
    """Instancia e executa um PQFM, retorna Xq_all_raw."""
    import dataclasses
    qfm = cls(name_file=name_file, data_dir=data_dir, **kwargs)
    result = qfm.run()
    Xq = np.asarray(result["Xq_all_raw"])
    return Xq


def precompute(dataset: str, csv_path: str, out_dir: str, q_enc: int, overwrite: bool):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # carrega CSV para obter shape
    df = pd.read_csv(csv_path)
    n_samples, n_cols = df.shape
    n_features = n_cols - 1
    q_enc_use = min(q_enc, n_features)
    print(f"[{dataset}] {n_samples} amostras, {n_features} features, q_enc={q_enc_use}", flush=True)

    # pqfmlib quer: data_dir/<name_file>.csv
    # copiamos o csv para data_dir temporário
    tmp_dir = Path(out_dir) / "_pqfm_data"
    tmp_dir.mkdir(exist_ok=True)
    name_file = f"_pqfm_{dataset}"
    dest = tmp_dir / f"{name_file}.csv"
    import shutil
    shutil.copy(csv_path, dest)

    from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM

    MAPS = {
        "cdising": (
            CDIsingProjectiveQFM,
            {"q_enc": q_enc_use, "ideal": True, "seed": 42,
             "k_max": 2, "output_root": str(Path(out_dir) / "_pqfm_output")},
        ),
        "heisenberg": (
            HeisenbergProjectiveQFM,
            {"q_enc": q_enc_use, "ideal": True, "seed": 42,
             "R": 2, "alpha": 0.1,
             "output_root": str(Path(out_dir) / "_pqfm_output")},
        ),
        "xyz_diag": (
            XYZProjectiveQFM,
            {"q_enc": q_enc_use, "ideal": True, "seed": 42,
             "encoding_mode": "multi_axis", "features_per_qubit": 3,
             "output_root": str(Path(out_dir) / "_pqfm_output")},
        ),
    }

    for map_name, (cls, kwargs) in MAPS.items():
        out_file = Path(out_dir) / f"{dataset}_{map_name}.npz"
        if out_file.exists() and not overwrite:
            print(f"  [{map_name}] já existe, pulando.", flush=True)
            continue

        print(f"  [{map_name}] computando...", flush=True)
        try:
            Xq = run_qfm(cls, kwargs, str(dest), str(tmp_dir), name_file)
            np.savez_compressed(out_file, Xq=Xq)
            print(f"  [{map_name}] OK → {Xq.shape} → {out_file}", flush=True)
        except Exception as e:
            print(f"  [{map_name}] ERRO: {e}", flush=True)
            import traceback; traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="nome do dataset (ex: cardiac)")
    parser.add_argument("--csv", required=True, help="caminho para CSV com ultima coluna = target")
    parser.add_argument("--out", default="results/quantum_precomputed")
    parser.add_argument("--q_enc", type=int, default=Q_ENC_DEFAULT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    precompute(args.dataset, args.csv, args.out, args.q_enc, args.overwrite)


if __name__ == "__main__":
    main()
