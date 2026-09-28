"""
Teste de vazamento transdutivo do PQFM CD-Ising.

Compara, com a mesma versão do pqfmlib e o mesmo nested CV (5x10):
  - "global":  imputação, padronização e matriz J (MI entre atributos) ajustadas
               com TODAS as amostras antes da CV (protocolo de run_quantum_cv.py
               e da dissertação);
  - "perfold": imputação, padronização e J ajustadas só no treino externo de cada
               dobra; o circuito é então aplicado a treino e teste com esses
               parâmetros (sem nenhuma informação do teste).

O rótulo y nunca é usado pelo pqfmlib em nenhum dos dois modos (a MI é entre
atributos). A diferença global − perfold mede o efeito do vazamento
transdutivo não supervisionado.

Uso (no ambiente com pqfmlib, Python >= 3.10):
  python run_quantum_cv_perfold.py --dataset cardiac
  python run_quantum_cv_perfold.py --dataset ispy1
  python run_quantum_cv_perfold.py --dataset ispy1 --n_repeats 2   # checagem rápida (10 dobras)

Instruções completas: INSTRUCOES_TESTE_VAZAMENTO.md

Saída: results/vazamento_{dataset}_cdising_resultados.csv
       results/vazamento_{dataset}_cdising_estatistica_pareada.csv
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.datasets import load_breast_cancer_radiology, load_cardiac_disease
import src.pipeline as _pipeline_mod
from src.pipeline import run_nested_cv, run_nested_cv_precomputed
from src.stats import paired_delta_summary

RESULTS_DIR = ROOT / "results"
Q_ENC = {"cardiac": 13, "ispy1": 20}
LOADERS = {"cardiac": load_cardiac_disease, "ispy1": load_breast_cancer_radiology}

_TMP = tempfile.mkdtemp(prefix="pqfm_perfold_")


def cdising_transform(X_fit: np.ndarray, X_apply: np.ndarray, q_enc: int) -> np.ndarray:
    """Calcula J e os blocos só com X_fit e aplica o circuito CD-Ising a X_apply."""
    from pqfmlib import CDIsingProjectiveQFM

    qfm = CDIsingProjectiveQFM(
        name_file="perfold", data_dir=_TMP, output_root=_TMP,
        q_enc=q_enc, ideal=True, seed=42, k_max=2,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        qfm.X_q_all = np.asarray(X_fit, dtype=float)
        qfm.num_features = int(qfm.X_q_all.shape[1])
        qfm.setup_backend_and_estimator()
        qfm.prepare_output_folder()
        qfm.compute_global_J_and_feature_blocks()
        qfm.prepare_blocks_and_edges()
        qfm.X_q_all = np.asarray(X_apply, dtype=float)
        qfm.execute_quantum_feature_map()
    return np.asarray(qfm.Xq_all_raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(LOADERS))
    parser.add_argument("--n_repeats", type=int, default=10,
                        help="repetições do 5-fold externo (10 = protocolo completo, 50 dobras)")
    args = parser.parse_args()
    _pipeline_mod.N_REPEATS = args.n_repeats
    dname, q_enc = args.dataset, Q_ENC[args.dataset]

    X, y = LOADERS[dname]()
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    # baseline recalculado no mesmo ambiente (versões de libs podem mudar a AUC)
    records = [run_nested_cv(X_arr, y_arr, label=f"{dname}_baseline")]
    print(f"[baseline] AUC={records[0]['auc'].mean():.4f}", flush=True)

    # modo global (transdutivo): mesmo preparo de run_quantum_cv.prepare_pqfm_csv
    t0 = time.time()
    X_glob = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_arr))
    Xq_full = cdising_transform(X_glob, X_glob, q_enc)
    print(f"[global] Xq {Xq_full.shape} em {time.time() - t0:.0f}s", flush=True)
    for mode, suffix in [("isolated", "iso"), ("hybrid", "hyb")]:
        label = f"{dname}_cdising_global_{suffix}"
        df = run_nested_cv_precomputed(X_arr, y_arr, Xq_full, label=label, mode=mode)
        print(f"[{label}] AUC={df['auc'].mean():.4f}", flush=True)
        records.append(df)

    # modo por dobra: feature_fn recebe X já imputado/padronizado com fit no treino
    # cache por dobra: os modos iso e híb reutilizam o mesmo circuito
    cache: dict[bytes, np.ndarray] = {}

    def feature_fn(X_tr_sc, X_te_sc):
        X_apply = np.vstack([X_tr_sc, X_te_sc])
        key = X_apply.tobytes()
        if key not in cache:
            cache[key] = cdising_transform(X_tr_sc, X_apply, q_enc)
        Xq = cache[key]
        return Xq[: len(X_tr_sc)], Xq[len(X_tr_sc):]

    for mode, suffix in [("isolated", "iso"), ("hybrid", "hyb")]:
        label = f"{dname}_cdising_perfold_{suffix}"
        t0 = time.time()
        df = run_nested_cv(X_arr, y_arr, label=label, feature_fn=feature_fn, mode=mode)
        print(f"[{label}] AUC={df['auc'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        records.append(df)

    df_all = pd.concat(records, ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df_all.to_csv(RESULTS_DIR / f"vazamento_{dname}_cdising_resultados.csv", index=False)

    # estatística pareada: cada modo vs baseline, e perfold vs global
    auc = {lab: g.sort_values("fold")["auc"].reset_index(drop=True)
           for lab, g in df_all.groupby("label")}
    base_auc = auc[f"{dname}_baseline"]
    rows = []
    for suffix in ["iso", "hyb"]:
        g, p = f"{dname}_cdising_global_{suffix}", f"{dname}_cdising_perfold_{suffix}"
        rows.append({"comparacao": f"{g} vs baseline", **paired_delta_summary(auc[g], base_auc)})
        rows.append({"comparacao": f"{p} vs baseline", **paired_delta_summary(auc[p], base_auc)})
        rows.append({"comparacao": f"{p} vs {g}", **paired_delta_summary(auc[p], auc[g])})
    df_stat = pd.DataFrame(rows)
    df_stat.to_csv(RESULTS_DIR / f"vazamento_{dname}_cdising_estatistica_pareada.csv", index=False)
    print(df_stat.to_string(index=False))


if __name__ == "__main__":
    main()
