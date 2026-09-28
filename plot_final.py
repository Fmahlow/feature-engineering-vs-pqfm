"""Gera figuras comparativas completas (clássico + dequantizacao + PQFM)."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

STATUS_COLOR = {
    'Consistente':        '#2ecc71',
    'Promissor':          '#27ae60',
    'Sugestivo':          '#f39c12',
    'Tendencia positiva': '#e67e22',
    'Sem ganho':          '#bdc3c7',
    'BASELINE':           '#2c3e50',
}

SHORT = {
    'kpca_hyb': 'KPCA (híb)', 'kpca_iso': 'KPCA (iso)',
    'ae_hyb': 'AE (híb)', 'ae_iso': 'AE (iso)',
    'nystroem_poly_hyb': 'Nyst-Poly (híb)', 'nystroem_poly_iso': 'Nyst-Poly (iso)',
    'nystroem_rbf_hyb': 'Nyst-RBF (híb)', 'nystroem_rbf_iso': 'Nyst-RBF (iso)',
    'pairwise_hyb': 'Pairwise (híb)', 'pairwise_iso': 'Pairwise (iso)',
    'bins_hyb': 'Bins+Cross (híb)', 'bins_iso': 'Bins+Cross (iso)',
    'rff_hyb': 'RFF (híb)', 'rff_iso': 'RFF (iso)',
    'poly2_hyb': 'Poly-2 (híb)', 'poly2_iso': 'Poly-2 (iso)',
    'elm_hyb': 'ELM (híb)', 'elm_iso': 'ELM (iso)',
    'deq_cdising_hyb': 'Deq CD-Ising (híb)', 'deq_cdising_iso': 'Deq CD-Ising (iso)',
    'deq_heisen_hyb': 'Deq Heisenberg (híb)', 'deq_heisen_iso': 'Deq Heisenberg (iso)',
    'deq_xyz_hyb': 'Deq XYZ (híb)', 'deq_xyz_iso': 'Deq XYZ (iso)',
    'pqfm_cdising_hyb': 'PQFM CD-Ising (híb)', 'pqfm_cdising_iso': 'PQFM CD-Ising (iso)',
    'pqfm_heisenberg_hyb': 'PQFM Heisenberg (híb)', 'pqfm_heisenberg_iso': 'PQFM Heisenberg (iso)',
    'pqfm_xyz_hyb': 'PQFM XYZ (híb)', 'pqfm_xyz_iso': 'PQFM XYZ (iso)',
}


def group_key(lbl: str) -> int:
    if 'pqfm' in lbl: return 3
    if 'deq' in lbl: return 2
    if 'baseline' in lbl: return 0
    return 1


def plot_step(step: int, dname: str, title: str):
    df_cl   = pd.read_csv(f'results/etapa{step}_{dname}_resultados.csv')
    stat_cl = pd.read_csv(f'results/etapa{step}_{dname}_estatistica_pareada.csv')
    df_q    = pd.read_csv(f'results/etapa{step}_{dname}_quantum_resultados.csv')
    stat_q  = pd.read_csv(f'results/etapa{step}_{dname}_quantum_estatistica_pareada.csv')

    df_all   = pd.concat([df_cl, df_q], ignore_index=True)
    stat_all = pd.concat([stat_cl, stat_q], ignore_index=True)

    auc = df_all.groupby('label')['auc'].agg(['mean', 'std']).reset_index()
    baseline_label = f'{dname}_baseline'
    bl_auc = float(auc.loc[auc['label'] == baseline_label, 'mean'].values[0])

    stat_auc = stat_all[stat_all['metric'] == 'auc'].set_index('label')

    rows = auc[auc['label'] != baseline_label].copy()
    rows['group'] = rows['label'].apply(group_key)
    rows = rows.sort_values(['group', 'mean'], ascending=[True, False])

    names, means, stds, colors = [], [], [], []
    for _, row in rows.iterrows():
        lbl = str(row['label'])
        short = lbl.replace(dname + '_', '')
        names.append(SHORT.get(short, short))
        means.append(float(row['mean']))
        stds.append(float(row['std']))
        st = str(stat_auc.loc[lbl, 'status']) if lbl in stat_auc.index else 'Sem ganho'
        colors.append(STATUS_COLOR.get(st, '#95a5a6'))

    n = len(names)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 0.38)))
    y_pos = list(range(n - 1, -1, -1))

    ax.barh(y_pos, means[::-1], xerr=stds[::-1], color=colors[::-1],
            alpha=0.85, height=0.7, capsize=3, error_kw={'linewidth': 1})
    ax.axvline(bl_auc, color='#2c3e50', linewidth=2, linestyle='--',
               label=f'Baseline ({bl_auc:.4f})')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8.5)
    ax.set_xlabel('AUC média (50 dobras externos)')
    ax.set_title(f'Etapa {step} — {title}\nClássico | Dequantização | PQFM (pqfmlib, statevector ideal)',
                 fontsize=11)

    patches = [mpatches.Patch(color=c, label=s)
               for s, c in STATUS_COLOR.items() if s != 'BASELINE']
    ax.legend(handles=patches, loc='lower right', fontsize=7, ncol=2)
    plt.tight_layout()

    out = f'results/figures/etapa{step}_{dname}_completo.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {out}')


if __name__ == '__main__':
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    for step, dname, title in [
        (1, 'cardiac', 'Cardiac Disease (303×13)'),
        (2, 'wdbc',    'WDBC (569×30)'),
        (3, 'ispy1',   'I-SPY1 (161×370)'),
    ]:
        try:
            plot_step(step, dname, title)
        except Exception as e:
            print(f'  ERRO {dname}: {e}')
    print('Concluído.')
