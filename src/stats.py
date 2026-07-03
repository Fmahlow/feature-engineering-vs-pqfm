"""
Analise estatistica pareada, replicando a Secao 3.7 da dissertacao.
Ver Secao 8 do plano.

Para cada dobra externa i: delta_i = M_i(tecnica) - M_i(classico).
Reporta media de delta, IC 95% (t de Student) e p-valor do teste de
Wilcoxon unilateral (alternative="greater"), com os 4 rotulos:
Consistente / Promissor / Sugestivo / Tendencia positiva.
"""

from __future__ import annotations

import pandas as pd

STATUS_LABELS = ("Consistente", "Promissor", "Sugestivo", "Tendencia positiva", "Sem ganho")


def paired_delta_summary(metric_case: pd.Series, metric_baseline: pd.Series) -> dict:
    """Recebe as 50 metricas pareadas (mesma dobra) de uma tecnica e do
    baseline classico correspondente, devolve delta medio, IC95% e p de
    Wilcoxon (scipy.stats.wilcoxon, alternative="greater"), alem do rotulo.
    """
    raise NotImplementedError(
        "TODO: scipy.stats.wilcoxon(metric_case - metric_baseline, "
        "alternative='greater'); IC95% via scipy.stats.t.interval sobre "
        "as 50 diferencas pareadas."
    )


def classify_status(delta_mean: float, ci_low: float, ci_high: float, p_value: float) -> str:
    """Aplica os 4 criterios da Secao 3.7/8 do plano, nesta ordem de
    prioridade: Consistente > Promissor > Sugestivo > Tendencia positiva.
    """
    if delta_mean <= 0:
        return "Sem ganho"
    if ci_low > 0 and p_value < 0.05:
        return "Consistente"
    if p_value < 0.05:
        return "Promissor"
    if 0.05 <= p_value < 0.10:
        return "Sugestivo"
    return "Tendencia positiva"
