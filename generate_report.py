"""
Gera o RELATORIO_FINAL.md consolidando os resultados das 4 etapas,
espelhando a estrutura da Secao 4.5 / Tabela 4.3 da dissertacao.

Uso: python generate_report.py
(Roda com os arquivos disponiveis em results/; pula etapas nao concluidas.)
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results")

DATASETS = {
    1: ("cardiac", "Cardiac Disease — Cleveland (303×13)"),
    2: ("wdbc", "Breast Cancer Pathology — WDBC (569×30)"),
    3: ("ispy1", "Breast Cancer Radiology — I-SPY1 (161×370)"),
    4: ("toxicity", "Molecular Toxicity — CRY1 (171×156)"),
}

# AUCs de referencia da dissertacao (coluna "Classico", Tabelas 4.1/4.2)
REF_BASELINE = {"cardiac": 0.8784, "wdbc": 0.9847, "ispy1": 0.5315, "toxicity": None}

# Melhores AUC PQFM da dissertacao (melhor config para cada base)
REF_PQFM_BEST = {
    "cardiac": ("XYZ-cross/shared-feature", 0.8873),  # Tabela 4.2 hibr. Classico+PCA
    "wdbc": ("Heisenberg SHAP", 0.9911),              # Tabela 4.1 SHAP Quântico
    "ispy1": ("XYZ-diagonal Z-local", 0.5622),        # Fig 4.3
    "toxicity": None,
}

LABEL_MAP = {
    "baseline": "Baseline clássico",
    "deq_cdising_iso": "Deq. CD-Ising (iso)",
    "deq_cdising_hyb": "Deq. CD-Ising (híb)",
    "deq_heisen_iso": "Deq. Heisenberg (iso)",
    "deq_heisen_hyb": "Deq. Heisenberg (híb)",
    "deq_xyz_iso": "Deq. XYZ-diag (iso)",
    "deq_xyz_hyb": "Deq. XYZ-diag (híb)",
    "pairwise_iso": "#1 Aritmética par (iso)",
    "pairwise_hyb": "#1 Aritmética par (híb)",
    "poly2_iso": "#2 Polinômio-2 (iso)",
    "poly2_hyb": "#2 Polinômio-2 (híb)",
    "rff_iso": "#3 RFF-RBF (iso)",
    "rff_hyb": "#3 RFF-RBF (híb)",
    "nystroem_rbf_iso": "#4 Nystroem-RBF (iso)",
    "nystroem_rbf_hyb": "#4 Nystroem-RBF (híb)",
    "nystroem_poly_iso": "#5 Nystroem-Poly (iso)",
    "nystroem_poly_hyb": "#5 Nystroem-Poly (híb)",
    "kpca_iso": "#6 Kernel PCA (iso)",
    "kpca_hyb": "#6 Kernel PCA (híb)",
    "elm_iso": "#7 ELM/Proj.Rand (iso)",
    "elm_hyb": "#7 ELM/Proj.Rand (híb)",
    "ae_iso": "#8 Autoencoder (iso)",
    "ae_hyb": "#8 Autoencoder (híb)",
    "bins_iso": "#9 Binning+cross (iso)",
    "bins_hyb": "#9 Binning+cross (híb)",
}

def short_key(full_label: str) -> str:
    for k in LABEL_MAP:
        if full_label.endswith(k):
            return k
    return full_label.split("_", 1)[-1]


def load_step(step: int, dname: str):
    res_path = os.path.join(RESULTS_DIR, f"etapa{step}_{dname}_resultados.csv")
    stat_path = os.path.join(RESULTS_DIR, f"etapa{step}_{dname}_estatistica_pareada.csv")
    if not os.path.exists(res_path):
        return None, None
    res = pd.read_csv(res_path)
    stat = pd.read_csv(stat_path) if os.path.exists(stat_path) else pd.DataFrame()
    return res, stat


def md_table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |",
             "| " + " | ".join(["---"]*len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def format_step(step: int, dname: str, title: str) -> str:
    res, stat = load_step(step, dname)
    if res is None:
        return f"### Etapa {step} — {title}\n\n*(não concluída ainda)*\n"

    auc_summary = (
        res.groupby("label")["auc"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    baseline_auc = auc_summary[auc_summary["label"].str.contains("baseline")]["mean"].values
    baseline_auc = baseline_auc[0] if len(baseline_auc) else None

    ref_bl = REF_BASELINE.get(dname)
    ref_pqfm = REF_PQFM_BEST.get(dname)

    out = [f"### Etapa {step} — {title}\n"]

    # comparacao com dissertacao
    if baseline_auc is not None and ref_bl is not None:
        diff = baseline_auc - ref_bl
        out.append(f"**Baseline obtido:** AUC = {baseline_auc:.4f} "
                   f"(dissertação: {ref_bl:.4f}, Δ = {diff:+.4f})\n")
    if ref_pqfm:
        out.append(f"**Melhor PQFM da dissertação:** {ref_pqfm[0]}, AUC = {ref_pqfm[1]:.4f}\n")

    # tabela AUC top-10
    top = auc_summary.head(12).copy()
    top["Técnica"] = top["label"].apply(lambda l: LABEL_MAP.get(short_key(l), short_key(l)))
    top["AUC média"] = top["mean"].map("{:.4f}".format)
    top["±std"] = top["std"].map("{:.4f}".format)

    if not stat.empty:
        auc_s = stat[stat["metric"] == "auc"].set_index("label")
        top["Status"] = top["label"].map(
            lambda l: auc_s.loc[l, "status"] if l in auc_s.index else "Baseline"
        )
        top["p-valor"] = top["label"].map(
            lambda l: f"{auc_s.loc[l,'p_wilcoxon']:.4f}" if l in auc_s.index else "—"
        )
    else:
        top["Status"] = "—"
        top["p-valor"] = "—"

    table_df = top[["Técnica", "AUC média", "±std", "Status", "p-valor"]].reset_index(drop=True)
    out.append("\n**Top-10 configurações por AUC média:**\n")
    out.append(md_table(table_df))

    # contagem de status
    if not stat.empty:
        auc_stat = stat[stat["metric"] == "auc"]
        status_counts = auc_stat["status"].value_counts()
        out.append(f"\n**Distribuição de status (AUC, vs. baseline):** "
                   + ", ".join(f"{k}: {v}" for k, v in status_counts.items()))

    # melhor classica vs melhor dequantizacao
    best_classical = auc_summary[
        ~auc_summary["label"].str.contains("deq|baseline|cdising|heisen")
    ].head(1)
    best_deq = auc_summary[auc_summary["label"].str.contains("deq")].head(1)

    if not best_classical.empty:
        bc = best_classical.iloc[0]
        out.append(f"\n**Melhor técnica clássica (#1–#9):** {LABEL_MAP.get(short_key(bc['label']), bc['label'])} "
                   f"— AUC = {bc['mean']:.4f}")
    if not best_deq.empty:
        bd = best_deq.iloc[0]
        out.append(f"\n**Melhor dequantização (#0):** {LABEL_MAP.get(short_key(bd['label']), bd['label'])} "
                   f"— AUC = {bd['mean']:.4f}")

    out.append("")
    return "\n".join(out)


def main():
    lines = [
        "# Relatório Final: Feature Engineering Clássico vs. PQFMs\n",
        "> Gerado automaticamente por `generate_report.py`\n",
        "## 1. Pergunta de pesquisa\n",
        "Transformações clássicas não-lineares (produtos, kernels, projeções aleatórias, ",
        "dequantização analítica dos PQFMs) produzem ganho igual, maior ou nulo em relação ",
        "ao baseline clássico, comparado às configurações de PQFM (CD-Ising, Heisenberg, XYZ) ",
        "reportadas na dissertação de Carmo (2026), com o mesmo pipeline e protocolo estatístico?\n",
        "## 2. Protocolo\n",
        "- **Classificador:** `GradientBoostingClassifier` (sklearn), `random_state=42`\n",
        "- **Grade:** `n_estimators`∈{80,100,120,140,160}, `max_depth`∈{2,5,7,9,11}, `criterion`∈{friedman_mse,squared_error}\n",
        "- **CV externa:** `RepeatedStratifiedKFold(5×10)` = 50 dobras\n",
        "- **CV interna:** `GridSearchCV + PredefinedSplit` (80/20 estratificado)\n",
        "- **Métrica primária:** AUC; secundárias: F1, precisão, recall, acurácia\n",
        "- **Análise estatística:** Δᵢ = M_i(técnica) − M_i(baseline), IC 95% (t), Wilcoxon unilateral\n",
        "- **Rótulos:** Consistente (IC>0, p<0.05) > Promissor (p<0.05) > Sugestivo (0.05≤p<0.10) > Tendência > Sem ganho\n\n",
        "## 3. Resultados por dataset\n",
    ]

    for step, (dname, title) in DATASETS.items():
        lines.append(format_step(step, dname, title))
        lines.append("")

    # sintese final
    lines.append("## 4. Síntese comparativa\n")

    summary_rows = []
    for step, (dname, title) in DATASETS.items():
        res, stat = load_step(step, dname)
        if res is None:
            summary_rows.append({
                "Dataset": dname.upper(), "Baseline AUC": "—",
                "Ref. dissertação": str(REF_BASELINE.get(dname)),
                "Melhor clássica AUC": "—", "Status melhor clássica": "—",
                "Melhor deq. AUC": "—",
            })
            continue

        auc = res.groupby("label")["auc"].mean()
        bl_auc = auc[auc.index.str.contains("baseline")].values[0] if any(auc.index.str.contains("baseline")) else None

        classical_mask = ~auc.index.str.contains("deq|baseline|cdising|heisen|xyz")
        best_cl_auc = auc[classical_mask].max() if classical_mask.any() else None
        best_cl_label = auc[classical_mask].idxmax() if classical_mask.any() else "—"

        deq_mask = auc.index.str.contains("deq")
        best_deq_auc = auc[deq_mask].max() if deq_mask.any() else None

        status_cl = "—"
        if not stat.empty and best_cl_label != "—":
            auc_stat = stat[stat["metric"] == "auc"].set_index("label")
            if best_cl_label in auc_stat.index:
                status_cl = auc_stat.loc[best_cl_label, "status"]

        summary_rows.append({
            "Dataset": dname.upper(),
            "Baseline AUC": f"{bl_auc:.4f}" if bl_auc else "—",
            "Ref. dissertação": f"{REF_BASELINE[dname]:.4f}" if REF_BASELINE.get(dname) else "—",
            "Melhor clássica AUC": f"{best_cl_auc:.4f}" if best_cl_auc else "—",
            "Status melhor clássica": status_cl,
            "Melhor deq. AUC": f"{best_deq_auc:.4f}" if best_deq_auc else "—",
        })

    summary_df = pd.DataFrame(summary_rows)
    lines.append(md_table(summary_df))

    lines.append("\n### Resposta à pergunta de pesquisa\n")
    lines.append(
        "*(Preencher após conclusão de todas as etapas — ver distribuição de status acima.)*\n"
    )
    lines.append(
        "- Em **X de 4 datasets**, pelo menos uma técnica clássica atingiu status *Consistente* ou *Promissor* vs. baseline.\n"
    )
    lines.append(
        "- A dequantização analítica (#0) atingiu status *X* nos datasets disponíveis — "
        "confirmando/refutando que o ganho é reproduzível sem computação quântica.\n"
    )

    report = "\n".join(lines)
    path = os.path.join(ROOT, "RELATORIO_FINAL.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"Relatório gerado: {path}")
    print(report[:500] + "...")


if __name__ == "__main__":
    main()
