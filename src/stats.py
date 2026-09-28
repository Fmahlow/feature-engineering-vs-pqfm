"""
Analise estatistica pareada, replicando a Secao 3.7 da dissertacao.

Para cada dobra externa i: delta_i = M_i(tecnica) - M_i(classico).
Reporta media de delta, IC 95% (t de Student) e p-valor do teste de
Wilcoxon unilateral (alternative="greater"), com os 4 rotulos:
Consistente / Promissor / Sugestivo / Tendencia positiva.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

STATUS_LABELS = ("Consistente", "Promissor", "Sugestivo", "Tendencia positiva", "Sem ganho")


def paired_delta_summary(metric_case: pd.Series, metric_baseline: pd.Series) -> dict:
    """Recebe as 50 metricas pareadas de uma tecnica e do baseline,
    devolve um dict com delta_mean, ci_low, ci_high, p_wilcoxon e status.
    """
    deltas = np.asarray(metric_case) - np.asarray(metric_baseline)
    n = len(deltas)
    delta_mean = deltas.mean()

    # IC 95% pela distribuicao t (Secao 3.7)
    se = deltas.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = delta_mean - t_crit * se
    ci_high = delta_mean + t_crit * se

    # Wilcoxon unilateral: H_1: tecnica > baseline
    try:
        _, p_wilcoxon = stats.wilcoxon(deltas, alternative="greater")
    except ValueError:
        # todos os deltas sao zero
        p_wilcoxon = 1.0

    label = classify_status(delta_mean, ci_low, ci_high, p_wilcoxon)

    return {
        "delta_mean": delta_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_wilcoxon": p_wilcoxon,
        "status": label,
    }


def classify_status(delta_mean: float, ci_low: float, ci_high: float, p_value: float) -> str:
    """Aplica os 4 criterios da Secao 3.7/8 do plano."""
    if delta_mean <= 0:
        return "Sem ganho"
    if ci_low > 0 and p_value < 0.05:
        return "Consistente"
    if p_value < 0.05:
        return "Promissor"
    if 0.05 <= p_value < 0.10:
        return "Sugestivo"
    return "Tendencia positiva"


def build_stats_table(results_df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    """A partir de um DataFrame com colunas [label, fold, auc, f1, precision, recall, accuracy],
    calcula a analise pareada de cada label vs. baseline_label.

    Retorna DataFrame com uma linha por (label, metrica) e colunas
    [label, metric, delta_mean, ci_low, ci_high, p_wilcoxon, status].
    """
    baseline = results_df[results_df["label"] == baseline_label].set_index("fold")
    metrics = ["auc", "f1", "precision", "recall", "accuracy"]
    rows = []
    for label, grp in results_df[results_df["label"] != baseline_label].groupby("label"):
        grp = grp.set_index("fold")
        for m in metrics:
            if m not in grp.columns:
                continue
            case_vals = grp[m].reindex(baseline.index)
            base_vals = baseline[m]
            valid = case_vals.notna() & base_vals.notna()
            if valid.sum() < 5:
                continue
            summary = paired_delta_summary(case_vals[valid], base_vals[valid])
            rows.append({"label": label, "metric": m, **summary})

    return pd.DataFrame(rows)
