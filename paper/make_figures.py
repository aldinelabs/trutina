"""Figures for the paper, drawn from the numbers in the technical report. Output: figures/*.pdf (vector, fonts embedded).

  uv run --with matplotlib python make_figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'pdf.fonttype': 42, 'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'legend.frameon': False, 'lines.linewidth': 1.2,
})
FG, MUTED, ACCENT = '#111111', '#8a8a85', '#b4452f'

# --- Figure 1: reading the option letters at each layer -------------------------------------------
lens = [40.0, 39.8, 38.8, 27.7, 27.6, 38.7, 39.8, 31.2, 31.2, 31.2, 25.9, 25.6, 39.1, 40.4, 34.8, 32.5,
        35.2, 41.3, 32.2, 44.1, 54.0, 61.8, 59.7, 64.2, 64.5, 63.3, 63.5, 63.7, 65.0, 65.5, 64.8, 63.9]
exits = [(12, 42.5), (16, 60.9), (20, 72.2), (24, 73.9), (32, 74.65)]
mass = {16: .2, 17: .2, 18: .7, 19: .3, 20: 1.2, 21: 2.2, 22: 7.0, 23: 3.3, 24: 26.6, 25: 17.4, 26: 16.5, 27: 16.0,
        28: 42.6, 29: 54.2, 30: 85.8, 31: 94.2, 32: 99.9}
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={'width_ratios': [1.25, 1]})
a.plot(range(1, 33), lens, color=MUTED, label='untrained backbone (logit lens)')
a.plot(*zip(*exits), color=ACCENT, marker='o', ms=3.5, label='trained exits (v3)')
a.axhline(47.0, color=FG, lw=0.6, ls=(0, (3, 3)))
a.text(1, 49, 'label frequency only, 47.0', fontsize=6.5)
a.set_xlabel('layer read'); a.set_ylabel('LocalLLaMA accuracy (%)'); a.set_ylim(20, 80)
a.legend(loc='upper left', fontsize=6.5)
b.plot(list(mass), list(mass.values()), color=FG)
b.set_xlabel('layer read'); b.set_ylabel('mass on option letters (%)'); b.set_ylim(0, 100)
fig.tight_layout(); fig.savefig('figures/fig_layers.pdf'); plt.close(fig)

# --- Figure 2: continued training and forgetting (two v2.1 runs) ---------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.plot([0, 1000, 2000], [74.25, 73.8, 70.05], color=MUTED, ls='--', marker='o', ms=3, label=r'lr $5\!\times\!10^{-5}$, 23% replay')
ax.plot([0, 1000, 2000, 3000, 4000, 6000, 7000], [74.25, 72.85, 73.9, 74.35, 74.7, 74.35, 74.55], color=ACCENT, marker='o', ms=3,
        label=r'lr $2\!\times\!10^{-5}$, 74% replay')
ax.set_xlabel('training step'); ax.set_ylabel('LocalLLaMA accuracy (%)'); ax.set_ylim(69, 76)
ax.legend(loc='lower right', fontsize=7)
fig.tight_layout(); fig.savefig('figures/fig_forgetting.pdf'); plt.close(fig)

# --- Figure 3: paired differences v3 - v2.1 ------------------------------------------------------
rows = [('JudgeBench (620 pairs)', 3.1, 1.1, 5.2), ('DecisionBench hard (80 states)', 2.5, -1.6, 7.2),
        ('JevBench hard (94 tasks)', 2.1, -3.2, 7.4), ('validation, >1,024 tokens (1,480)', 0.9, 0.1, 1.9),
        ('image validation (1,924)', 0.5, -0.3, 1.3), ('JevBench public (231 tasks)', 0.4, -1.7, 2.6),
        ('DecisionBench medium (80 states)', 0.2, -1.6, 1.9), ('this-that (1,710)', 0.1, -1.2, 1.5),
        ('LocalLLaMA (2,000)', -0.1, -1.0, 0.9), ('MASSIVE (2,974)', -0.1, -0.7, 0.5),
        ('fast-decisions (1,600)', -0.3, -1.0, 0.5), ('original text validation (5,578)', -0.3, -0.7, 0.2),
        ('validation, >4,096 tokens (51)', -3.9, -10.2, 0.0)]
fig, ax = plt.subplots(figsize=(3.3, 3.0))
for i, (lab, d, lo, hi) in enumerate(rows):
    y = len(rows) - 1 - i
    sig = lo > 0 or hi < 0
    ax.plot([lo, hi], [y, y], color=MUTED, lw=1.0)
    ax.plot(d, y, 'o', ms=4, mfc=FG if sig else 'white', mec=FG, mew=0.8)
ax.axvline(0, color=FG, lw=0.6)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.5)
ax.set_xlabel('v3 minus v2.1 (points)'); ax.set_xlim(-11, 8)
fig.tight_layout(); fig.savefig('figures/fig_paired.pdf', bbox_inches='tight'); plt.close(fig)

# --- Figure 4: validation accuracy at each exit during the v3 run --------------------------------
steps = [0, 1500, 3000, 4500, 6000, 7500]
series = {'12': [46.9, 48.3, 47.7, 48.0, 47.9, 47.9], '16': [69.8, 72.7, 72.9, 73.2, 73.0, 73.0],
          '20': [83.5, 86.4, 86.9, 87.0, 87.2, 87.2], '24': [84.6, 87.7, 88.2, 88.5, 88.6, 88.5],
          '32': [85.7, 88.9, 89.2, 89.5, 89.5, 89.4]}
fig, ax = plt.subplots(figsize=(3.3, 2.3))
shades = ['#c9c9c4', '#a3a39e', '#7d7d78', '#555550', ACCENT]
for (k, v), c in zip(series.items(), shades):
    ax.plot(steps, v, color=c, marker='o', ms=2.5, label=f'layer {k}')
ax.set_xlabel('training step'); ax.set_ylabel('v3 validation accuracy (%)'); ax.set_ylim(40, 92)
ax.legend(loc='center right', fontsize=6.5, ncol=1)
fig.tight_layout(); fig.savefig('figures/fig_v3curves.pdf'); plt.close(fig)
print('ok')
