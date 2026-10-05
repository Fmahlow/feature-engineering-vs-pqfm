"""Gera a figura do teste de vazamento transdutivo do CD-Ising (global vs por dobra)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

COLOR = {
    'baseline': '#8a8984',
    'global':   '#2a78d6',
    'perfold':  '#eb6834',
}
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

PROTO_NAME = {
    'baseline': 'Baseline',
    'global':   'CD-Ising global\n(ajuste com todas as amostras)',
    'perfold':  'CD-Ising por dobra\n(ajuste só no treino)',
}

DATASETS = [
    ('cardiac', 'Cardiac (303×13)'),
    ('ispy1',   'I-SPY1 (161×370)'),
]


def style(ax):
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='both', length=0, colors=INK_2, labelsize=9)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_auc(ax, dname, title):
    """AUC de cada uma das 50 dobras externas, modo híbrido."""
    df = pd.read_csv(f'results/vazamento_{dname}_cdising_resultados.csv')
    labels = {
        'baseline': f'{dname}_baseline',
        'global':   f'{dname}_cdising_global_hyb',
        'perfold':  f'{dname}_cdising_perfold_hyb',
    }
    rng = np.random.default_rng(0)
    bl_mean = df.loc[df['label'] == labels['baseline'], 'auc'].mean()
    ax.axvline(bl_mean, color=COLOR['baseline'], linewidth=1.2, linestyle='--', zorder=1)

    for y, key in zip((2, 1, 0), ('baseline', 'global', 'perfold')):
        auc = df.loc[df['label'] == labels[key], 'auc'].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(auc))
        ax.scatter(auc, y + jitter, s=16, color=COLOR[key], alpha=0.35,
                   linewidths=0, zorder=2)
        ax.scatter([auc.mean()], [y], s=90, color=COLOR[key], edgecolors='white',
                   linewidths=1.5, zorder=3)
        ax.annotate(f'{auc.mean():.4f}', (auc.mean(), y), xytext=(0, 13),
                    textcoords='offset points', ha='center', fontsize=9.5,
                    color=INK, fontweight='bold', zorder=4)

    ax.set_ylim(-0.6, 2.7)
    ax.set_yticks((2, 1, 0))
    ax.set_yticklabels([PROTO_NAME[k] for k in ('baseline', 'global', 'perfold')],
                       fontsize=9, color=INK)
    ax.set_xlabel('AUC por dobra (modo híbrido; ponto grande = média das 50 dobras)',
                  fontsize=9, color=INK_2)
    ax.set_title(title, fontsize=12, color=INK, loc='left', fontweight='bold')
    style(ax)


def plot_delta(ax, dname):
    """Δ AUC vs baseline (pp) com IC 95%, a partir da estatística pareada."""
    stat = pd.read_csv(f'results/vazamento_{dname}_cdising_estatistica_pareada.csv')
    stat = stat.set_index('comparacao')
    rows = [
        (3, 'Híbrido', 'global',  f'{dname}_cdising_global_hyb vs baseline'),
        (2, 'Híbrido', 'perfold', f'{dname}_cdising_perfold_hyb vs baseline'),
        (1, 'Isolado', 'global',  f'{dname}_cdising_global_iso vs baseline'),
        (0, 'Isolado', 'perfold', f'{dname}_cdising_perfold_iso vs baseline'),
    ]
    ax.axvline(0, color=COLOR['baseline'], linewidth=1.2, linestyle='--', zorder=1)
    for y, _, key, comp in rows:
        d, lo, hi = 100 * stat.loc[comp, ['delta_mean', 'ci_low', 'ci_high']].astype(float)
        ax.plot([lo, hi], [y, y], color=COLOR[key], linewidth=2, solid_capstyle='round',
                zorder=2)
        ax.scatter([d], [y], s=70, color=COLOR[key], edgecolors='white', linewidths=1.5,
                   zorder=3)
        ax.annotate(f'{d:+.2f} pp  [{lo:+.1f}; {hi:+.1f}]', (hi, y), xytext=(7, 0),
                    textcoords='offset points', va='center', fontsize=9, color=INK)

    lo_all = 100 * stat['ci_low'].astype(float).min()
    hi_all = 100 * stat['ci_high'].astype(float).max()
    span = hi_all - lo_all
    ax.set_xlim(lo_all - 0.05 * span, hi_all + 0.55 * span)
    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks((3, 2, 1, 0))
    ax.set_yticklabels([f'{mode} · {"global" if key == "global" else "por dobra"}'
                        for _, mode, key, _ in rows], fontsize=9, color=INK)
    ax.set_xlabel('Δ AUC vs baseline, em pp (média e IC 95%; 0 = sem ganho)',
                  fontsize=9, color=INK_2)
    style(ax)


if __name__ == '__main__':
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6),
                             gridspec_kw={'height_ratios': [1, 1.05]})
    for col, (dname, title) in enumerate(DATASETS):
        plot_auc(axes[0, col], dname, title)
        plot_delta(axes[1, col], dname)

    handles = [Line2D([], [], marker='o', linestyle='', markersize=9, color=COLOR[k],
                      label=PROTO_NAME[k].replace('\n', ' '))
               for k in ('baseline', 'global', 'perfold')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Teste de vazamento transdutivo do CD-Ising: o ganho some quando o mapa '
                 'é ajustado só no treino',
                 fontsize=13.5, color=INK, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), w_pad=3, h_pad=2)

    out = 'results/figures/vazamento_cdising_global_vs_por_dobra.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> {out}')
