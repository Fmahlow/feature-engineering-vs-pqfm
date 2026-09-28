"""
Gera as figuras de barras comparando baseline, PQFMs e tecnicas classicas.
Executar apos run_experiments.py --step N.

Uso: python plot_results.py --step 1  (ou 2, 3, 4, all)
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METRICS = ["auc", "f1", "precision", "recall", "accuracy"]
METRIC_LABELS = {"auc": "AUC", "f1": "F1", "precision": "Precisão",
                 "recall": "Recall", "accuracy": "Acurácia"}

STATUS_COLORS = {
    "Consistente": "#2ca02c",
    "Promissor": "#1f77b4",
    "Sugestivo": "#ff7f0e",
    "Tendencia positiva": "#9467bd",
    "Sem ganho": "#d62728",
    "baseline": "#7f7f7f",
}

LABEL_SHORT = {
    "baseline": "Baseline",
    "deq_cdising_iso": "Deq-CDIsing (iso)",
    "deq_cdising_hyb": "Deq-CDIsing (hyb)",
    "deq_heisen_iso": "Deq-Heisen (iso)",
    "deq_heisen_hyb": "Deq-Heisen (hyb)",
    "deq_xyz_iso": "Deq-XYZ (iso)",
    "deq_xyz_hyb": "Deq-XYZ (hyb)",
    "pairwise_iso": "#1 Pairwise (iso)",
    "pairwise_hyb": "#1 Pairwise (hyb)",
    "poly2_iso": "#2 Poly-2 (iso)",
    "poly2_hyb": "#2 Poly-2 (hyb)",
    "rff_iso": "#3 RFF (iso)",
    "rff_hyb": "#3 RFF (hyb)",
    "nystroem_rbf_iso": "#4 Nyst-RBF (iso)",
    "nystroem_rbf_hyb": "#4 Nyst-RBF (hyb)",
    "nystroem_poly_iso": "#5 Nyst-Poly (iso)",
    "nystroem_poly_hyb": "#5 Nyst-Poly (hyb)",
    "kpca_iso": "#6 KPCA (iso)",
    "kpca_hyb": "#6 KPCA (hyb)",
    "elm_iso": "#7 ELM (iso)",
    "elm_hyb": "#7 ELM (hyb)",
    "ae_iso": "#8 AE (iso)",
    "ae_hyb": "#8 AE (hyb)",
    "bins_iso": "#9 Bins (iso)",
    "bins_hyb": "#9 Bins (hyb)",
    "cdising_zz_iso": "QM-CDIsing ZZ (iso)",
    "cdising_zz_hyb": "QM-CDIsing ZZ (hyb)",
    "heisenberg_z_iso": "QM-Heisen Z (iso)",
    "heisenberg_z_hyb": "QM-Heisen Z (hyb)",
    "xyz_diag_z_iso": "QM-XYZ-diag Z (iso)",
    "xyz_diag_z_hyb": "QM-XYZ-diag Z (hyb)",
}


def short_label(full_label: str) -> str:
    for k, v in LABEL_SHORT.items():
        if full_label.endswith(k):
            return v
    return full_label.split("_", 1)[-1] if "_" in full_label else full_label


def plot_step(step: int, dname: str):
    res_path = os.path.join(RESULTS_DIR, f"etapa{step}_{dname}_resultados.csv")
    stat_path = os.path.join(RESULTS_DIR, f"etapa{step}_{dname}_estatistica_pareada.csv")

    if not os.path.exists(res_path):
        print(f"  Arquivo nao encontrado: {res_path}")
        return

    results = pd.read_csv(res_path)
    stats_df = pd.read_csv(stat_path) if os.path.exists(stat_path) else pd.DataFrame()

    # AUC summary
    auc_summary = (
        results.groupby("label")["auc"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    # status por label (para colorir barras)
    status_map = {}
    if not stats_df.empty:
        auc_stats = stats_df[stats_df["metric"] == "auc"]
        for _, row in auc_stats.iterrows():
            status_map[row["label"]] = row["status"]

    fig, ax = plt.subplots(figsize=(14, max(6, len(auc_summary) * 0.4)))
    labels = [short_label(l) for l in auc_summary["label"]]
    means = auc_summary["mean"].values
    stds = auc_summary["std"].values
    full_labels = auc_summary["label"].values

    colors = []
    for fl in full_labels:
        if "baseline" in fl:
            colors.append(STATUS_COLORS["baseline"])
        else:
            s = status_map.get(fl, "Tendencia positiva")
            colors.append(STATUS_COLORS.get(s, "#9467bd"))

    bars = ax.barh(labels, means, xerr=stds, color=colors, alpha=0.85, capsize=3)

    # linha vertical no baseline
    baseline_auc = results[results["label"].str.contains("baseline")]["auc"].mean()
    ax.axvline(baseline_auc, color="black", linestyle="--", linewidth=1.2, label="Baseline")

    ax.set_xlabel("AUC média (±std, 50 dobras)")
    ax.set_title(f"Etapa {step} — {dname.upper()} — Comparativo AUC")
    ax.legend(fontsize=8)

    # legenda de cores
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=v, label=k) for k, v in STATUS_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, f"etapa{step}_{dname}_auc.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Figura salva: {fig_path}")

    # figura com todas as metricas (top-10 configuracoes por AUC)
    top_labels = auc_summary.tail(10)["label"].tolist()
    top_df = results[results["label"].isin(top_labels)]
    summary_all = top_df.groupby("label")[METRICS].mean()

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(METRICS))
    width = 0.8 / len(top_labels)
    for k, lbl in enumerate(top_labels):
        if lbl not in summary_all.index:
            continue
        vals = [summary_all.loc[lbl, m] for m in METRICS]
        ax2.bar(x + k * width, vals, width, label=short_label(lbl), alpha=0.8)

    ax2.set_xticks(x + width * len(top_labels) / 2)
    ax2.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax2.set_ylabel("Score médio")
    ax2.set_title(f"Etapa {step} — {dname.upper()} — Top-10 configurações")
    ax2.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    fig2_path = os.path.join(FIG_DIR, f"etapa{step}_{dname}_all_metrics.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    print(f"  Figura salva: {fig2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all")
    args = parser.parse_args()

    step_map = {1: "cardiac", 2: "wdbc", 3: "ispy1", 4: "toxicity"}
    steps = list(step_map.keys()) if args.step == "all" else [int(args.step)]
    for s in steps:
        dname = step_map[s]
        print(f"\n--- Gerando figuras: Etapa {s} ({dname}) ---")
        plot_step(s, dname)
