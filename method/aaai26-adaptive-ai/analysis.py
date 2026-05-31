#!/usr/bin/env python3
"""
analysis.py  --  Generate ALL paper figures from saved CSV results.

Categories:
  A. College Admission (CA) figures  -- synthetic experiment
  B. WoofNette (WN) figures          -- image classification simulation

Usage:
    python analysis.py                        # generate everything
    python analysis.py --experiment ca        # college-admission only
    python analysis.py --experiment wn        # woofnette only
    python analysis.py --experiment combined  # combined 1x4 panel only
"""

import argparse
import itertools
import os
import shutil
import warnings
from collections import OrderedDict

# Use writable temp caches on shared systems where $HOME config dirs may be read-only.
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import yaml
from scipy.stats import ttest_rel
from numpy.linalg import norm

USE_LATEX = shutil.which('latex') is not None
matplotlib.rcParams['text.usetex'] = USE_LATEX

rcParams_ticks = {
    'figure.dpi': 300,
    'xtick.bottom': True,
    'ytick.left': True,
    'legend.loc': 'best',
    'legend.fancybox': True,
    'axes.edgecolor': 'grey',
    'axes.xmargin': 0.05,
    'text.usetex': USE_LATEX,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'lines.linewidth': 2,
    'lines.markersize': 8,
}
sns.set_theme(style='ticks', rc=rcParams_ticks)

# Nature-style colour palette
nature_colors = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
    '#8491B4', '#91D1C2', '#DC0000',
    '#FFB81C', '#F2C500', '#E6AB02', '#D9A76A', '#B8860B',
    '#F2F2F2', '#CCCCCC', '#999999', '#666666', '#333333',
    '#7E6148', '#B09C85', '#A1CAF1', '#E7D4B5', '#94B594',
]

# Core six colours used throughout
colors = ['#E64B35', '#A1CAF1', '#8491B4', '#E6AB02', '#91D1C2', '#00A087']
extended_colors = [c for c in nature_colors if c not in colors]

# Model-specific colours
COLOR_SINGLE  = '#E64B35'
COLOR_ALIGNED = '#A1CAF1'
COLOR_COMP    = '#8491B4'
COLOR_ADAPT   = '#E6AB02'
COLOR_ADAPT_LINE = '#91D1C2'

barw = 0.35
human_lw = 2.5

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
CA_RESULTS  = os.path.join(RESULTS_DIR, 'college_admission')
WN_RESULTS  = os.path.join(RESULTS_DIR, 'woofnette')

os.makedirs(FIGURES_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(PROJECT_DIR, 'config.yaml')
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
else:
    config = {}


# ===========================================================================
#  College Admission data generation & model helpers
# ===========================================================================
RAND_STATE = 42
_rng = np.random.default_rng(RAND_STATE)


def _reset_rng():
    """Reset the module-level RNG for reproducibility across figures."""
    global _rng
    _rng = np.random.default_rng(RAND_STATE)


def generate_data(N, r, delta, tau=0.5):
    """Generate the College Admission Dataset."""
    G = _rng.choice([1, 0], size=N, p=[r, 1 - r])
    X = _rng.uniform(0, 1, size=(N, 2))
    y = np.zeros(N)
    for i in range(N):
        if G[i] == 1:   # aligned (privileged)
            val = (0.5 + delta) * X[i, 1] + (0.5 - delta) * X[i, 0] - tau
        else:            # complementary (underprivileged)
            val = (0.5 + delta) * X[i, 0] + (0.5 - delta) * X[i, 1] - tau
        y[i] = 1 if val >= 0 else 0
    return X, y, G


def fit_logistic(X, y, sample_weight=None, fit_intercept=True):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(fit_intercept=fit_intercept, solver='lbfgs',
                               max_iter=1000)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_single_ai(X, y, G, p=0.5, human_pred=None,
                     fit_intercept=True, random_state=None):
    """Theorem 4 implementation: g(theta) = p*L_a + (1-p)*L_c."""
    from sklearn.linear_model import LogisticRegression
    X, y, G = np.asarray(X), np.asarray(y), np.asarray(G)
    n = len(y)
    n_a = np.sum(G == 1)
    n_c = np.sum(G == 0)

    target = y.copy() if human_pred is None else np.where(G == 1, human_pred, y)

    sample_weight = np.zeros(n)
    if n_a > 0:
        sample_weight[G == 1] = p / n_a
    if n_c > 0:
        sample_weight[G == 0] = (1 - p) / n_c
    sample_weight *= n

    model = LogisticRegression(fit_intercept=fit_intercept, solver='lbfgs',
                                max_iter=1000, random_state=random_state)
    model.fit(X, target, sample_weight=sample_weight)
    return model


def get_intercept(model):
    return np.array(model.intercept_).ravel()[0]


def get_coefficients(model):
    return model.coef_.flatten()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_logistic_loss(theta, X_aug, y):
    from sklearn.metrics import log_loss
    logits = X_aug.dot(theta)
    probs = sigmoid(logits)
    return log_loss(y, probs)


def add_bias_column(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def quad_fit(x, y, x_new):
    return np.poly1d(np.polyfit(x, y, 2))(x_new)


def _load_csv(path):
    if not os.path.exists(path):
        print(f"  [SKIP] Missing: {path}")
        return None
    return pd.read_csv(path)


# ===========================================================================
#  Human simulation helpers (for team-accuracy sweep)
# ===========================================================================
def get_decision_human(y_test, G_test, acc_align=0.8, acc_comp=0.5,
                       conf_under_max=0.1, conf_over_max=0.1,
                       min_conf=0.0, max_conf=1.0,
                       advanced_conf=False, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_test)
    h_pred = np.zeros(N, dtype=int)
    h_conf = np.zeros(N)
    for i in range(N):
        p = acc_align if G_test[i] == 1 else acc_comp
        correct = rng.random() < p
        h_pred[i] = y_test[i] if correct else 1 - y_test[i]
        low = max(p - conf_under_max, min_conf)
        high = min(p + conf_over_max, max_conf)
        if high < low:
            high = low
        h_conf[i] = rng.uniform(low, high)
    if advanced_conf:
        max_extra = 0.05
        noise = rng.uniform(0, max_extra, N)
        mask_correct = h_pred == y_test
        h_conf[mask_correct] += noise[mask_correct]
        h_conf[~mask_correct] -= noise[~mask_correct]
    h_conf = np.clip(h_conf, min_conf, max_conf)
    return h_pred, h_conf


def get_team_decisions_cgpr(h_pred, h_conf, ai_pred, r, tau=0.5, seed=0):
    rng = np.random.default_rng(seed)
    N = len(h_pred)
    team = np.zeros(N, dtype=int)
    for i in range(N):
        if h_conf[i] > tau:
            team[i] = h_pred[i]
        else:
            team[i] = ai_pred[i] if rng.random() < r else h_pred[i]
    return team


def adaptive_predict_hard(X, selector, ma, mc):
    pa = ma.predict_proba(X)[:, 1]
    pc = mc.predict_proba(X)[:, 1]
    return (np.where(selector == 1, pa, pc) >= 0.5).astype(int)


def _sample_ca_plot_points(X, y, G, n_samples=40, random_state=RAND_STATE):
    """Sample a deterministic subset so related CA figures overlay the same points."""
    rng_plot = np.random.default_rng(random_state)
    idx = rng_plot.choice(len(y), size=min(n_samples, len(y)), replace=False)
    return idx, X[idx], y[idx], G[idx]


# ===========================================================================
#  A.  COLLEGE ADMISSION FIGURES
# ===========================================================================

def ca_fig_data(delta=0.25, r=0.5):
    """Figure: Dataset scatter (3-panel: Complementary, Aligned, All)."""
    print("  Generating CA dataset scatter...")
    _reset_rng()
    X, y, G = generate_data(10000, r=r, delta=delta)
    _, Xs, ys, Gs = _sample_ca_plot_points(X, y, G)

    color_aligned = '#00A087'
    color_comp = '#E64B35'

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    titles = ['Complementary Region', 'Aligned Region', 'Combined']
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=20, fontweight='bold', pad=14)
        ax.set_xlabel('GPA', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_facecolor('white')
    axes[0].set_ylabel('Score', fontsize=22, fontweight='bold')

    def _scatter(ax, mask):
        mpos = mask & (ys == 1)
        colors_pos = np.where(Gs[mpos] == 1, color_aligned, color_comp)
        ax.scatter(Xs[mpos, 0], Xs[mpos, 1], marker='+', c=colors_pos,
                   s=280, linewidths=3.5, zorder=3)
        mneg = mask & (ys == 0)
        colors_neg = np.where(Gs[mneg] == 1, color_aligned, color_comp)
        ax.scatter(Xs[mneg, 0], Xs[mneg, 1], marker='_', c=colors_neg,
                   s=280, linewidths=3.5, zorder=3)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

    _scatter(axes[0], Gs == 0)
    _scatter(axes[1], Gs == 1)
    _scatter(axes[2], np.ones_like(Gs, dtype=bool))

    legend_elements = [
        mlines.Line2D([0], [0], marker='+', color='black', markerfacecolor='black',
                       markersize=18, linewidth=0, markeredgewidth=3.5, label='Admitted'),
        mlines.Line2D([0], [0], marker='_', color='black', markerfacecolor='black',
                       markersize=18, linewidth=0, markeredgewidth=3.5, label='Rejected'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=18, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out = os.path.join(FIGURES_DIR, f'ca_data_delta{delta}_r{r}.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_db(delta=0.25, r=0.5):
    """Figure: Decision boundaries (3-model panel)."""
    print("  Generating CA decision boundaries...")
    _reset_rng()
    X, y, G = generate_data(10000, r=r, delta=delta)

    m_align = fit_logistic(X[G == 1], y[G == 1])
    m_comp  = fit_logistic(X[G == 0], y[G == 0])
    m_single = fit_logistic(X, y)
    models = [m_comp, m_align, m_single]
    model_names = ['Complementary AI', 'Aligned AI', 'Standard AI']

    _, Xs, ys, Gs = _sample_ca_plot_points(X, y, G)
    color_aligned = '#00A087'
    color_comp = '#E64B35'
    region_positive = '#91D1C2'
    region_negative = '#F39B7F'

    M = len(models)
    fig, axes = plt.subplots(1, M, figsize=(14 / 3 * M, 4.5), sharex=True, sharey=True)

    x_min, x_max = X[:, 0].min() - .05, X[:, 0].max() + .05
    y_min, y_max = X[:, 1].min() - .05, X[:, 1].max() + .05
    xx = np.linspace(x_min, x_max, 200)
    yy = np.linspace(y_min, y_max, 200)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]

    for i, model in enumerate(models):
        ax = axes[i]
        ax.set_title(model_names[i], fontsize=20, fontweight='bold', pad=14)
        ax.set_xlabel('GPA', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_facecolor('white')
        if i == 0:
            ax.set_ylabel('Score', fontsize=22, fontweight='bold')

        probs = model.predict_proba(grid)[:, 1].reshape(XX.shape)
        Z = (probs >= 0.5).astype(int)
        ax.contourf(XX, YY, Z, levels=[-0.5, 0.5, 1.5],
                    colors=[region_negative, region_positive], alpha=0.35)
        ax.contour(XX, YY, probs, levels=[0.5], colors='#333333', linewidths=3.0)

        mpos = ys == 1
        ax.scatter(Xs[mpos, 0], Xs[mpos, 1], marker='+',
                   c=np.where(Gs[mpos] == 1, color_aligned, color_comp),
                   s=260, linewidths=3.2, zorder=3)
        mneg = ys == 0
        ax.scatter(Xs[mneg, 0], Xs[mneg, 1], marker='_',
                   c=np.where(Gs[mneg] == 1, color_aligned, color_comp),
                   s=260, linewidths=3.2, zorder=3)
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('gray')

    legend_elements = [
        mlines.Line2D([0], [0], marker='+', color='black', markerfacecolor='black',
                      markersize=18, linewidth=0, markeredgewidth=3.5, label='Admitted'),
        mlines.Line2D([0], [0], marker='_', color='black', markerfacecolor='black',
                      markersize=18, linewidth=0, markeredgewidth=3.5, label='Rejected'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=18, frameon=False, bbox_to_anchor=(0.5, -0.05))

    fig.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out = os.path.join(FIGURES_DIR, f'ca_db_delta{delta}_r{r}.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
#  Experiment runners (regenerate data, fit models, measure gaps)
# ---------------------------------------------------------------------------

def _run_distance_experiment():
    """Sweep delta, compute specialist distance and accuracy gaps."""
    csv_path = os.path.join(CA_RESULTS, 'specialist_divergence.csv')
    df = _load_csv(csv_path)
    if df is not None:
        methods_acc = {
            'Adaptive AI': df['gap_acc_adaptive'].to_numpy(),
            'Aligned AI': df['gap_acc_aligned'].to_numpy(),
            'Complementary AI': df['gap_acc_complementary'].to_numpy(),
        }
        return df['D'].to_numpy(), methods_acc

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, accuracy_score
    _reset_rng()
    N, r, tau, test_size = 10000, 0.5, 0.5, 0.3
    delta_values = np.linspace(0, 0.5, 10)
    D_list, gap_acc_adaptive, gap_acc_align, gap_acc_comp = [], [], [], []

    for delta in delta_values:
        X, y, G = generate_data(N, r, delta, tau=tau)
        X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
            X, y, G, test_size=test_size, stratify=G, random_state=RAND_STATE)

        mdl_a = fit_logistic(X_tr[G_tr == 1], y_tr[G_tr == 1])
        mdl_c = fit_logistic(X_tr[G_tr == 0], y_tr[G_tr == 0])
        mdl_s = fit_logistic(X_tr, y_tr)

        wa = get_coefficients(mdl_a)
        wc = get_coefficients(mdl_c)
        wa = wa / (norm(wa) + 1e-12)
        wc = wc / (norm(wc) + 1e-12)
        D_list.append(norm(wa - wc))

        acc_s = accuracy_score(y_te, mdl_s.predict(X_te)) * 100
        p_adapt = np.where(G_te == 1,
                           mdl_a.predict_proba(X_te)[:, 1],
                           mdl_c.predict_proba(X_te)[:, 1])
        acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
        gap_acc_adaptive.append(acc_a - acc_s)
        gap_acc_align.append(
            accuracy_score(y_te, mdl_a.predict(X_te)) * 100 - acc_s)
        gap_acc_comp.append(
            accuracy_score(y_te, mdl_c.predict(X_te)) * 100 - acc_s)

    D_arr = np.array(D_list)
    methods_acc = {
        'Adaptive AI': np.array(gap_acc_adaptive),
        'Aligned AI':  np.array(gap_acc_align),
        'Complementary AI': np.array(gap_acc_comp),
    }
    return D_arr, methods_acc


def _run_distance_experiment_extended():
    """Multi-model distance experiment (LR, DT, RF, MLP, SVM)."""
    csv_path = os.path.join(CA_RESULTS, 'specialist_divergence_multi.csv')
    df = _load_csv(csv_path)
    if df is not None:
        name_map = {
            'Logistic Regression': 'LogisticRegression',
            'Decision Tree': 'DecisionTree',
            'Random Forest': 'RandomForest',
            'MLP': 'MLP',
            'SVM (RBF)': 'SVM_RBF',
        }
        combined_acc = {}
        for model_name, group_df in df.groupby('model', sort=False):
            key = name_map.get(model_name, model_name)
            combined_acc[key] = (
                group_df['D_prob'].to_numpy(),
                group_df['gap_acc_adaptive'].to_numpy(),
            )
        return combined_acc

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    _reset_rng()

    N, r, tau, test_size = 10000, 0.5, 0.5, 0.3
    delta_values = np.linspace(0, 0.5, 10)
    RS = 0

    model_defs = {
        'LogisticRegression': LogisticRegression(fit_intercept=True, random_state=RS),
        'DecisionTree': DecisionTreeClassifier(random_state=RS),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RS),
        'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100, random_state=RS),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=RS),
    }
    combined_acc = {}

    for mdl_name, mdl_template in model_defs.items():
        from sklearn.base import clone
        D_list, gap_acc = [], []
        for delta in delta_values:
            X, y, G = generate_data(N, r, delta, tau=tau)
            X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
                X, y, G, test_size=test_size, stratify=G, random_state=RS)

            ma = clone(mdl_template).fit(X_tr[G_tr == 1], y_tr[G_tr == 1])
            mc = clone(mdl_template).fit(X_tr[G_tr == 0], y_tr[G_tr == 0])
            ms = clone(mdl_template).fit(X_tr, y_tr)

            # probabilistic distance
            pa = ma.predict_proba(X_te)[:, 1]
            pc = mc.predict_proba(X_te)[:, 1]
            D_list.append(norm(pa - pc) / np.sqrt(len(pa)))

            acc_s = accuracy_score(y_te, ms.predict(X_te)) * 100
            p_adapt = np.where(G_te == 1, pa, pc)
            acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
            gap_acc.append(acc_a - acc_s)

        combined_acc[mdl_name] = (np.array(D_list), np.array(gap_acc))

    return combined_acc


def _run_human_accuracy_experiment():
    """Sweep human accuracy alpha, compute team accuracy gains."""
    csv_path = os.path.join(CA_RESULTS, 'imperfect_human.csv')
    df = _load_csv(csv_path)
    if df is not None:
        team_acc = {
            'Human + Adaptive perfect': df['team_adaptive_perfect'].to_numpy(),
            'Human + Adaptive weighted': df['team_adaptive_weighted'].to_numpy(),
            'Human + Single AI': df['team_single_ai'].to_numpy(),
            'Human + Aligned AI': df['team_aligned_ai'].to_numpy(),
            'Human + Complementary AI': df['team_complementary_ai'].to_numpy(),
        }
        return df['alpha'].to_numpy(), team_acc

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    _reset_rng()

    N, r_prop, delta, tau = 10000, 0.5, 0.25, 0.5
    alpha_vals = np.linspace(0.5, 1.0, 11)
    acc_comp_human = 0.5

    X, y, G = generate_data(N, r_prop, delta, tau)
    X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
        X, y, G, test_size=0.3, random_state=42)

    m_comp = fit_logistic(X_tr[G_tr == 0], y_tr[G_tr == 0])

    team_acc = {
        'Human + Adaptive weighted': [],
        'Human + Single AI': [],
        'Human + Aligned AI': [],
        'Human + Complementary AI': [],
    }

    for acc_align in alpha_vals:
        h_tr, conf_tr = get_decision_human(y_tr, G_tr,
                                           acc_align=acc_align, acc_comp=acc_comp_human,
                                           advanced_conf=False)
        h_te, conf_te = get_decision_human(y_te, G_te,
                                           acc_align=acc_align, acc_comp=acc_comp_human,
                                           advanced_conf=False)

        m_single = train_single_ai(X_tr, y_tr, G_tr, p=r_prop,
                                   human_pred=h_tr, random_state=42)
        m_align = fit_logistic(X_tr[G_tr == 1], h_tr[G_tr == 1])

        # weighted specialists
        m_wa = fit_logistic(X_tr, h_tr, sample_weight=conf_tr)
        m_wc = fit_logistic(X_tr, y_tr, sample_weight=1 - conf_tr)
        pred_wt = adaptive_predict_hard(X_te, G_te, m_wa, m_wc)

        # reliance rates
        da_mask = G_te == 1
        r_single = 1 - np.mean(h_te[da_mask] != m_single.predict(X_te[da_mask]))
        r_align  = 1 - np.mean(h_te[da_mask] != m_align.predict(X_te[da_mask]))
        r_comp   = 1 - np.mean(h_te[da_mask] != m_comp.predict(X_te[da_mask]))
        r_wt     = 1 - np.mean(h_te[da_mask] != pred_wt[da_mask])

        team_acc['Human + Adaptive weighted'].append(
            accuracy_score(y_te,
                get_team_decisions_cgpr(h_te, conf_te, pred_wt, r_wt, tau)))
        team_acc['Human + Single AI'].append(
            accuracy_score(y_te,
                get_team_decisions_cgpr(h_te, conf_te,
                    m_single.predict(X_te), r_single, tau)))
        team_acc['Human + Aligned AI'].append(
            accuracy_score(y_te,
                get_team_decisions_cgpr(h_te, conf_te,
                    m_align.predict(X_te), r_align, tau)))
        team_acc['Human + Complementary AI'].append(
            accuracy_score(y_te,
                get_team_decisions_cgpr(h_te, conf_te,
                    m_comp.predict(X_te), r_comp, tau)))

    return alpha_vals, team_acc


def _run_certainty_experiment_entropy():
    """Sweep group certainty (entropy-based), compute accuracy gains."""
    csv_path = os.path.join(CA_RESULTS, 'group_uncertainty.csv')
    df = _load_csv(csv_path)
    if df is not None:
        baseline = df['acc_single'].to_numpy()
        acc_adapt_gain = df['acc_adaptive'].to_numpy() - baseline
        acc_align_gain = df['acc_aligned'].to_numpy() - baseline
        acc_comp_gain = df['acc_complementary'].to_numpy() - baseline
        return df['group_cert'].to_numpy(), acc_adapt_gain, acc_align_gain, acc_comp_gain

    from sklearn.metrics import log_loss, accuracy_score
    _reset_rng()

    N, r, delta, tau = 10000, 0.5, 0.25, 0.5
    certainties = np.linspace(0.5, 1.0, 10)

    X, y, G = generate_data(N, r, delta, tau=tau)
    mdl_single = fit_logistic(X, y)
    mdl_align_per = fit_logistic(X[G == 1], y[G == 1])
    mdl_comp_per  = fit_logistic(X[G == 0], y[G == 0])

    acc_single = accuracy_score(y, mdl_single.predict(X)) * 100
    acc_align_val = accuracy_score(y, mdl_align_per.predict(X)) * 100
    acc_comp_val  = accuracy_score(y, mdl_comp_per.predict(X)) * 100

    acc_w = []
    group_cert_list = []
    rng_route = np.random.default_rng(0)

    for cert in certainties:
        p_test = np.where(G == 1, cert, 1 - cert)

        eps = 1e-12
        pclip = np.clip(p_test, eps, 1 - eps)
        H = (-pclip * np.log(pclip) - (1 - pclip) * np.log(1 - pclip)).mean()
        H_norm = H / (2 * np.log(2.0))
        group_cert_list.append(1.0 - H_norm)

        mdl_align_w = fit_logistic(X, y, sample_weight=p_test)
        mdl_comp_w  = fit_logistic(X, y, sample_weight=(1 - p_test))

        pa = mdl_align_w.predict_proba(X)[:, 1]
        pc = mdl_comp_w.predict_proba(X)[:, 1]
        route_to_a = rng_route.random(len(p_test)) < p_test
        pw = np.where(route_to_a, pa, pc)
        acc_w.append(accuracy_score(y, (pw >= 0.5).astype(int)) * 100)

    x_vals = np.array(group_cert_list)
    baseline = np.array([acc_single] * len(certainties))
    acc_adapt_gain = np.array(acc_w) - baseline
    acc_align_gain = np.array([acc_align_val] * len(certainties)) - baseline
    acc_comp_gain  = np.array([acc_comp_val] * len(certainties)) - baseline

    return x_vals, acc_adapt_gain, acc_align_gain, acc_comp_gain


def _run_proportion_experiment(delta_fixed=0.25):
    """Sweep alignment group proportion r, compute accuracy gains."""
    csv_path = os.path.join(CA_RESULTS, 'group_proportion.csv')
    df = _load_csv(csv_path)
    if df is not None:
        acc_dict_r = {
            'Adaptive AI': df['gap_acc_adaptive'].to_numpy(),
            'Aligned AI': df['gap_acc_aligned'].to_numpy(),
            'Complementary AI': df['gap_acc_complementary'].to_numpy(),
        }
        return df['r'].to_numpy(), acc_dict_r

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    _reset_rng()

    N, tau, test_size = 10000, 0.5, 0.3
    r_values_sweep = np.linspace(0.05, 0.95, 20)

    r_used = []
    gap_acc_adaptive_r, gap_acc_align_r, gap_acc_comp_r = [], [], []

    for r in r_values_sweep:
        X, y, G = generate_data(N, r, delta_fixed, tau=tau)
        X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
            X, y, G, test_size=test_size, stratify=G, random_state=RAND_STATE)

        X_tr_a, y_tr_a = X_tr[G_tr == 1], y_tr[G_tr == 1]
        X_tr_c, y_tr_c = X_tr[G_tr == 0], y_tr[G_tr == 0]
        if (len(y_tr_a) < 2 or len(np.unique(y_tr_a)) < 2 or
                len(y_tr_c) < 2 or len(np.unique(y_tr_c)) < 2):
            continue

        r_used.append(r)
        mdl_s = train_single_ai(X_tr, y_tr, G_tr, p=r, human_pred=None,
                                fit_intercept=True, random_state=RAND_STATE)
        mdl_a = fit_logistic(X_tr_a, y_tr_a)
        mdl_c = fit_logistic(X_tr_c, y_tr_c)

        acc_s = accuracy_score(y_te, mdl_s.predict(X_te)) * 100
        p_adapt = np.where(G_te == 1,
                           mdl_a.predict_proba(X_te)[:, 1],
                           mdl_c.predict_proba(X_te)[:, 1])
        acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
        gap_acc_adaptive_r.append(acc_a - acc_s)
        gap_acc_align_r.append(
            accuracy_score(y_te, mdl_a.predict(X_te)) * 100 - acc_s)
        gap_acc_comp_r.append(
            accuracy_score(y_te, mdl_c.predict(X_te)) * 100 - acc_s)

    r_arr = np.array(r_used)
    acc_dict_r = {
        'Adaptive AI': np.array(gap_acc_adaptive_r),
        'Aligned AI':  np.array(gap_acc_align_r),
        'Complementary AI': np.array(gap_acc_comp_r),
    }
    return r_arr, acc_dict_r


def _run_tradeoff_experiment():
    """Sweep r and alpha to compute trade-off ratio."""
    csv_path = os.path.join(CA_RESULTS, 'tradeoff.csv')
    df = _load_csv(csv_path)
    if df is not None:
        alpha_values = sorted(df['alpha'].unique())
        r_values = None
        tradeoff_dict = {}
        for alpha in alpha_values:
            df_alpha = df[df['alpha'] == alpha]
            if r_values is None:
                r_values = df_alpha['r'].to_numpy()
            tradeoff_dict[alpha] = df_alpha['tradeoff'].to_numpy()
        return r_values, alpha_values, tradeoff_dict

    _reset_rng()
    N, delta_val, tau = 10000, 0.25, 0.5
    r_values = np.linspace(0.5, 0.99, 10)
    alpha_values = [0.6, 0.8, 1.0]
    epsilon = 1e-6

    tradeoff_dict = {}
    for alpha in alpha_values:
        tradeoff_list = []
        for r in r_values:
            X_raw, y, G = generate_data(N, r, delta_val, tau=tau)
            X = X_raw  # no standardization to match notebook

            mdl_aligned = fit_logistic(X[G == 1], y[G == 1])
            mdl_single  = fit_logistic(X, y)

            b_a, w_a = get_intercept(mdl_aligned), get_coefficients(mdl_aligned)
            b_s, w_s = get_intercept(mdl_single),  get_coefficients(mdl_single)

            w_a_norm = w_a / (norm(w_a) + 1e-12)
            w_s_norm = w_s / (norm(w_s) + 1e-12)

            X_a_aug = add_bias_column(X[G == 1])
            X_c_aug = add_bias_column(X[G == 0])
            theta_s = np.concatenate(([b_s], w_s))
            theta_a = np.concatenate(([b_a], w_a))

            base_loss_a = compute_logistic_loss(theta_s, X_a_aug, y[G == 1])
            base_loss_c = compute_logistic_loss(theta_s, X_c_aug, y[G == 0])

            grad_a = np.zeros_like(theta_s)
            grad_c = np.zeros_like(theta_s)
            for j in range(theta_s.size):
                theta_pert = theta_s.copy()
                theta_pert[j] += epsilon
                grad_a[j] = (compute_logistic_loss(theta_pert, X_a_aug, y[G == 1])
                              - base_loss_a) / epsilon
                grad_c[j] = (compute_logistic_loss(theta_pert, X_c_aug, y[G == 0])
                              - base_loss_c) / epsilon

            direction = theta_a - theta_s
            direction /= (norm(direction) + 1e-12)
            da = grad_a.dot(direction)
            dc = grad_c.dot(direction)
            tradeoff_list.append(-dc / (da * (2 * alpha - 1) + 1e-12))

        tradeoff_dict[alpha] = tradeoff_list

    return r_values, alpha_values, tradeoff_dict


def _run_weight_sweep_experiment():
    """Load or regenerate the fixed-weight single-model sweep."""
    csv_path = os.path.join(CA_RESULTS, 'weight_sweep.csv')
    df = _load_csv(csv_path)
    if df is None:
        from college_admission import run_weight_sweep
        df = run_weight_sweep()

    r_values = np.sort(df['r'].unique())
    fixed_weights = [
        float(w) for w in sorted(df['fixed_weight'].dropna().unique())
    ]

    acc_curves = {}
    loss_curves = {}
    for w in fixed_weights:
        key = f'w={w:.2f}'
        df_w = df[df['model'] == key].sort_values('r')
        acc_curves[key] = df_w['acc_gain_vs_standard'].to_numpy()
        loss_curves[key] = df_w['loss_gain_vs_standard'].to_numpy()

    df_adapt = df[df['model'] == 'adaptive'].sort_values('r')
    acc_curves['Adaptive AI'] = df_adapt['acc_gain_vs_standard'].to_numpy()
    loss_curves['Adaptive AI'] = df_adapt['loss_gain_vs_standard'].to_numpy()
    return r_values, acc_curves, loss_curves


# ---------------------------------------------------------------------------
#  Standalone CA figure functions
# ---------------------------------------------------------------------------

def ca_fig_accuracy_gap_vs_distance():
    """Accuracy gap vs specialist distance (LR only) with quadratic fit."""
    print("  Generating CA accuracy gap vs distance...")
    D_arr, methods_acc = _run_distance_experiment()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    matching_colors = {
        'Adaptive AI': COLOR_ADAPT_LINE,
        'Aligned AI': COLOR_ALIGNED,
        'Complementary AI': COLOR_COMP,
    }
    style_mapping = {
        'Aligned AI': ('^', 8), 'Complementary AI': ('o', 8),
        'Adaptive AI': ('*', 10),
    }

    for name, vals in methods_acc.items():
        color = matching_colors[name]
        mk, mks = style_mapping[name]
        s_val = mks**2.1 if name != 'Adaptive AI' else mks**2.3
        ax.scatter(D_arr, vals, label=name, color=color, s=s_val,
                   marker=mk, edgecolor='white', linewidth=0.8, alpha=0.9)
        D_fit = np.linspace(D_arr.min(), D_arr.max(), 200)
        ax.plot(D_fit, quad_fit(D_arr, vals, D_fit), '--', linewidth=2.5,
                color=color, alpha=0.8)

    ax.set_xlabel('Specialist Divergence', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy Gain', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_accuracy_gap_vs_distance_sq.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_accuracy_gap_vs_distance_extended():
    """Multi-model accuracy gap vs distance (LR, DT, RF, MLP, SVM)."""
    print("  Generating CA accuracy gap vs distance (extended)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_acc = _run_distance_experiment_extended()

    model_colors = {
        'LogisticRegression': extended_colors[0],
        'DecisionTree': extended_colors[1],
        'RandomForest': extended_colors[2],
        'MLP': extended_colors[3],
        'SVM_RBF': extended_colors[4],
    }
    markers = ['o', 's', 'D', '^', 'v']

    fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))
    for i, (name, (D_arr, acc_arr)) in enumerate(combined_acc.items()):
        ax.scatter(D_arr, acc_arr, color=model_colors[name], s=70,
                   marker=markers[i], alpha=0.9, label=name,
                   edgecolor='white', linewidth=0.8)
        D_fit = np.linspace(D_arr.min(), D_arr.max(), 200)
        ax.plot(D_fit, quad_fit(D_arr, acc_arr, D_fit), '--',
                color=model_colors[name], linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Specialist Distance', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy Gain', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray',
              fancybox=True, borderpad=0.75, labelspacing=0.5, loc='best')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_accuracy_gap_vs_distance_extended.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_certainty_entropy():
    """Accuracy gain vs group certainty (entropy-based)."""
    print("  Generating CA accuracy gain vs certainty (entropy)...")
    x_vals, acc_adapt_gain, acc_align_gain, acc_comp_gain = \
        _run_certainty_experiment_entropy()

    palette = {
        'AlignmentOnly':  {'color': COLOR_ALIGNED, 'linestyle': '--', 'marker': '^', 'linewidth': 3},
        'ComplementOnly': {'color': COLOR_COMP, 'linestyle': '--', 'marker': 'o', 'linewidth': 3},
        'Adaptive':       {'color': COLOR_ADAPT_LINE, 'linestyle': '-', 'marker': '*'},
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(x_vals, acc_adapt_gain, label='Adaptive AI', **palette['Adaptive'], markersize=10)
    ax.plot(x_vals, acc_align_gain, label='Aligned AI', **palette['AlignmentOnly'])
    ax.plot(x_vals, acc_comp_gain, label='Complementary AI', **palette['ComplementOnly'])

    ax.set_xlabel('Group Certainty', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy Gain', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    ymin = min(acc_align_gain.min(), acc_comp_gain.min(), acc_adapt_gain.min()) - 1
    ymax = max(acc_align_gain.max(), acc_comp_gain.max(), acc_adapt_gain.max()) + 1
    ax.set_ylim(ymin, ymax)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray',
              fancybox=True, borderpad=0.75, labelspacing=0.5, loc='upper left')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_accuracy_gain_vs_certainty_entropy_sq.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_team_accuracy_gain_vs_human():
    """Team accuracy gain vs human accuracy in alignment region."""
    print("  Generating CA team accuracy gain vs human accuracy...")
    alpha_vals, team_acc = _run_human_accuracy_experiment()

    matching_colors = {
        'Baseline': COLOR_SINGLE, 'AlignmentOnly': COLOR_ALIGNED,
        'ComplementOnly': COLOR_COMP, 'Adaptive': COLOR_ADAPT_LINE,
    }
    style = {
        'Human + Adaptive weighted': (matching_colors['Adaptive'], '--', '*', 10, 'Adaptive AI'),
        'Human + Single AI':         (matching_colors['Baseline'], '--', 's', 8, 'Single AI'),
        'Human + Aligned AI':        (matching_colors['AlignmentOnly'], '--', '^', 8, 'Aligned AI'),
        'Human + Complementary AI':  (matching_colors['ComplementOnly'], '--', 'o', 8, 'Complementary AI'),
    }

    baseline_curve = np.array(team_acc['Human + Single AI']) * 100

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, curve in team_acc.items():
        if name == 'Human + Single AI' or name not in style:
            continue
        color, ls, mk, mks, label = style[name]
        plot_values = (np.array(curve) * 100) - baseline_curve
        ax.plot(alpha_vals, plot_values, label=label, color=color,
                linestyle=ls, marker=mk, markersize=mks, linewidth=2.5)

    ax.set_xlabel('Human Accuracy (Aligned Group)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy Gain', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray',
              fancybox=True, borderpad=0.75, labelspacing=0.5, loc='best')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_team_accuracy_gain_vs_human_accuracy_sq.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_team_accuracy_vs_human():
    """Absolute team accuracy vs human accuracy."""
    print("  Generating CA team accuracy vs human accuracy (absolute)...")
    alpha_vals, team_acc = _run_human_accuracy_experiment()

    matching_colors = {
        'Baseline': COLOR_SINGLE, 'AlignmentOnly': COLOR_ALIGNED,
        'ComplementOnly': COLOR_COMP, 'Adaptive': COLOR_ADAPT_LINE,
    }
    style = {
        'Human + Adaptive weighted': (matching_colors['Adaptive'], '-', '*', 10, 'Adaptive AI'),
        'Human + Single AI':         (matching_colors['Baseline'], '-', 's', 8, 'Single AI'),
        'Human + Aligned AI':        (matching_colors['AlignmentOnly'], '-', '^', 8, 'Aligned AI'),
        'Human + Complementary AI':  (matching_colors['ComplementOnly'], '-', 'o', 8, 'Complementary AI'),
    }

    fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))
    for name, curve in team_acc.items():
        if name not in style:
            continue
        color, ls, mk, mks, label = style[name]
        ax.plot(alpha_vals, np.array(curve) * 100, label=label, color=color,
                linestyle=ls, marker=mk, markersize=mks, linewidth=2.5)

    ax.set_xlabel('Human Accuracy (Aligned Group)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Team Accuracy (%)', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray',
              fancybox=True, borderpad=0.75, labelspacing=0.5, loc='best')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_team_accuracy_vs_human_accuracy.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_proportion(delta_fixed=0.25):
    """Accuracy gap vs group proportion."""
    print("  Generating CA accuracy gap vs proportion...")
    r_arr, acc_dict_r = _run_proportion_experiment(delta_fixed)

    matching_colors = {
        'Adaptive AI': COLOR_ADAPT_LINE, 'Aligned AI': COLOR_ALIGNED,
        'Complementary AI': COLOR_COMP,
    }
    style_mapping = {
        'Aligned AI': ('^', 8), 'Complementary AI': ('o', 8),
        'Adaptive AI': ('*', 10),
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, vals in acc_dict_r.items():
        if len(vals) == 0:
            continue
        color = matching_colors.get(name, colors[0])
        mk, mks = style_mapping.get(name, ('o', 8))
        s_val = mks**2.1 if name != 'Adaptive AI' else mks**2.3
        ax.scatter(r_arr, vals, label=name, color=color, s=s_val,
                   marker=mk, edgecolor='white', linewidth=0.8, alpha=0.9)
        r_fit = np.linspace(r_arr.min(), r_arr.max(), 200)
        ax.plot(r_fit, quad_fit(r_arr, vals, r_fit), '--', linewidth=2.5,
                color=color, alpha=0.8)

    ax.set_xlabel('Aligned Group Fraction', fontsize=16, fontweight='bold')
    ax.set_ylabel('Accuracy Gain', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f'ca_accuracy_gap_vs_proportion_delta{delta_fixed}_sq.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def ca_fig_weight_sweep():
    """Accuracy and log-loss gaps across fixed single-model weights."""
    print("  Generating CA fixed-weight sweep figures...")
    r_values, acc_curves, loss_curves = _run_weight_sweep_experiment()

    model_order = ['w=0.00', 'w=0.25', 'w=0.50', 'w=0.75', 'w=1.00', 'Adaptive AI']
    model_colors = {
        'w=0.00': extended_colors[0],
        'w=0.25': extended_colors[1],
        'w=0.50': extended_colors[2],
        'w=0.75': extended_colors[3],
        'w=1.00': extended_colors[4],
        'Adaptive AI': COLOR_ADAPT_LINE,
    }
    model_markers = {
        'w=0.00': 'o',
        'w=0.25': 's',
        'w=0.50': 'D',
        'w=0.75': '^',
        'w=1.00': 'v',
        'Adaptive AI': '*',
    }
    model_labels = {
        'w=0.00': r'Single AI ($w = 0$, Compl.)',
        'w=0.25': r'Single AI ($w = 0.25$)',
        'w=0.50': r'Single AI ($w = 0.5$)',
        'w=0.75': r'Single AI ($w = 0.75$)',
        'w=1.00': r'Single AI ($w = 1$, Aligned)',
        'Adaptive AI': 'Adaptive AI',
    }

    def _plot_curves(curves, ylabel, out_name):
        fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))
        legend_handles = []
        for name in model_order:
            vals = curves[name]
            mk = model_markers[name]
            color = model_colors[name]
            linewidth = 2.8 if name == 'Adaptive AI' else 2.2
            markersize = 11 if name == 'Adaptive AI' else 7
            r_fit = np.linspace(r_values.min(), r_values.max(), 200)
            vals_fit = quad_fit(r_values, vals, r_fit)
            ax.plot(
                r_fit, vals_fit, linewidth=linewidth, color=color
            )
            ax.scatter(
                r_values, vals, marker=mk, s=markersize**2.2,
                color=color, edgecolor='white', linewidth=0.9, zorder=3
            )
            legend_handles.append(mlines.Line2D(
                [], [], color=color, linewidth=linewidth, marker=mk,
                markersize=markersize, label=model_labels[name]
            ))

        ax.axhline(0.0, linestyle='--', linewidth=1.8, color='grey', alpha=0.8)
        legend_handles.insert(-1, mlines.Line2D(
            [], [], color='grey', linestyle='--', linewidth=1.8,
            label=r'Standard AI ($w = p$)'
        ))

        ax.set_xlabel('Aligned Group Fraction', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_facecolor('white')
        fig.set_facecolor('white')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.legend(
            handles=legend_handles, fontsize=10, frameon=True, framealpha=0.9,
            edgecolor='gray', fancybox=True, borderpad=0.65,
            labelspacing=0.45, loc='best'
        )
        plt.tight_layout()
        out = os.path.join(FIGURES_DIR, out_name)
        plt.savefig(out, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"    Saved: {out}")

    _plot_curves(
        acc_curves,
        'Accuracy Gain',
        'ca_weight_sweep_accuracy_gain.pdf',
    )
    _plot_curves(
        loss_curves,
        'Log-Loss Improvement',
        'ca_weight_sweep_logloss_improvement.pdf',
    )


def ca_fig_tradeoff():
    """Trade-off vs specialization with multiple alpha."""
    print("  Generating CA tradeoff vs specialization...")
    r_values, alpha_values, tradeoff_dict = _run_tradeoff_experiment()

    alpha_colors = {
        0.6: extended_colors[0], 0.8: extended_colors[1], 1.0: extended_colors[2],
    }
    linestyles = [':', '--', '-']
    markers = ['s', 's', 's']

    fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))
    for i, alpha in enumerate(alpha_values):
        t_arr = np.array(tradeoff_dict[alpha])
        label_str = rf'$\alpha$ = {alpha:.1f}'
        ax.plot(r_values, t_arr, marker=markers[i], linestyle=linestyles[i],
                markersize=8, linewidth=2.5, color=alpha_colors[alpha],
                label=label_str)

    ax.set_xlabel('Model Specialization', fontsize=16, fontweight='bold')
    ax.set_ylabel('Tradeoff', fontsize=16, fontweight='bold', labelpad=0)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    ax.legend(title_fontsize=14, fontsize=12, frameon=True, framealpha=0.9,
              edgecolor='gray', fancybox=True, borderpad=0.75,
              labelspacing=0.5, loc='best')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'ca_tradeoff_vs_specialization.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


# ---------------------------------------------------------------------------
#  Combined 1x4 panel (the key paper figure)
# ---------------------------------------------------------------------------

def _plot_performance_gap_on_ax(ax, D, data_dict, add_quad_fit=True):
    mc = {
        'Baseline': colors[0], 'Single AI': colors[0],
        'Aligned AI': colors[1], 'Complementary AI': colors[2],
        'Adaptive AI': colors[4],
    }
    sm = {
        'Baseline': ('s', 9), 'Single AI': ('s', 9),
        'Aligned AI': ('^', 9), 'Complementary AI': ('o', 9),
        'Adaptive AI': ('*', 11),
    }
    for name, vals in data_dict.items():
        label = 'Single AI' if name in ['Baseline', 'Single AI'] else name
        color = mc.get(label, colors[0])
        mk, mks = sm.get(label, ('o', 9))
        s_val = mks**2.1 if name != 'Adaptive AI' else mks**2.3
        ax.scatter(D, vals, label=label, color=color, s=s_val,
                   marker=mk, edgecolor='white', linewidth=1.0, alpha=0.9)
        if add_quad_fit:
            D_fit = np.linspace(D.min(), D.max(), 200)
            ax.plot(D_fit, quad_fit(D, vals, D_fit), '--', linewidth=3.5,
                    color=color, alpha=0.8)
    ax.set_ylabel('Accuracy Gain', fontsize=20, fontweight='bold')
    ax.set_xlabel('Specialist Divergence', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=16)


def _plot_human_accuracy_gain_on_ax(ax, alpha_vals, team_acc):
    mc = {
        'Baseline': colors[0], 'AlignmentOnly': colors[1],
        'ComplementOnly': colors[2], 'Adaptive': colors[4],
    }
    style = {
        'Human + Adaptive weighted': (mc['Adaptive'], '--', '*', 12, 'Adaptive AI'),
        'Human + Single AI':         (mc['Baseline'], '--', 's', 10, 'Single AI'),
        'Human + Aligned AI':        (mc['AlignmentOnly'], '--', '^', 10, 'Aligned AI'),
        'Human + Complementary AI':  (mc['ComplementOnly'], '--', 'o', 10, 'Complementary AI'),
    }
    baseline_curve = np.array(team_acc['Human + Single AI']) * 100
    for name, curve in team_acc.items():
        if name in ['Human + Adaptive perfect', 'Human + Single AI']:
            continue
        color, ls, mk, mks, label = style[name]
        plot_values = (np.array(curve) * 100) - baseline_curve
        ax.plot(alpha_vals, plot_values, label=label, color=color,
                linestyle=ls, marker=mk, markersize=mks, linewidth=3.5)
    ax.set_xlabel('Human Accuracy (Aligned Group)', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=16)


def _plot_proportion_gain_on_ax(ax, r_arr, acc_dict_r, add_quad_fit=True):
    mc = {
        'Baseline': colors[0], 'Single AI': colors[0],
        'Aligned AI': colors[1], 'Complementary AI': colors[2],
        'Adaptive AI': colors[4],
    }
    sm = {
        'Single AI': ('s', 9), 'Aligned AI': ('^', 9),
        'Complementary AI': ('o', 9), 'Adaptive AI': ('*', 11),
    }
    for name, vals in acc_dict_r.items():
        if len(vals) == 0:
            continue
        color = mc.get(name, colors[0])
        mk, mks = sm.get(name, ('o', 9))
        s_val = mks**2.1 if name != 'Adaptive AI' else mks**2.3
        ax.scatter(r_arr, vals, label=name, color=color, s=s_val,
                   marker=mk, edgecolor='white', linewidth=1.0, alpha=0.9)
        if add_quad_fit:
            r_fit = np.linspace(r_arr.min(), r_arr.max(), 200)
            ax.plot(r_fit, quad_fit(r_arr, vals, r_fit), '--', linewidth=3.5,
                    color=color, alpha=0.8)
    ax.set_xlabel('Aligned Group Fraction', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=16)


def _plot_certainty_gain_on_ax(ax, x_vals, acc_adapt_gain, acc_align_gain,
                                acc_comp_gain):
    mc = {
        'AlignmentOnly': colors[1], 'ComplementOnly': colors[2],
        'Adaptive': colors[4],
    }
    pal = {
        'AlignmentOnly':  {'color': mc['AlignmentOnly'], 'linestyle': '--', 'marker': '^', 'linewidth': 3.5, 'markersize': 10},
        'ComplementOnly': {'color': mc['ComplementOnly'], 'linestyle': '--', 'marker': 'o', 'linewidth': 3.5, 'markersize': 10},
        'Adaptive':       {'color': mc['Adaptive'], 'linestyle': '--', 'marker': '*'},
    }
    ax.plot(x_vals, acc_adapt_gain, label='Adaptive AI', **pal['Adaptive'], markersize=12)
    ax.plot(x_vals, acc_align_gain, label='Aligned AI', **pal['AlignmentOnly'])
    ax.plot(x_vals, acc_comp_gain, label='Complementary AI', **pal['ComplementOnly'])
    ax.set_xlabel('Group Certainty', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ymin = min(acc_align_gain.min(), acc_comp_gain.min(), acc_adapt_gain.min()) - 1
    ymax = max(acc_align_gain.max(), acc_comp_gain.max(), acc_adapt_gain.max()) + 1
    ax.set_ylim(ymin, ymax)


def ca_fig_combined():
    """Combined 1x4 panel figure (the key paper figure)."""
    print("  Generating CA combined 1x4 panel figure...")
    print("    Running sub-experiment 1/4: specialist distance...")
    D_arr, methods_acc = _run_distance_experiment()
    print("    Running sub-experiment 2/4: human accuracy...")
    alpha_vals, team_acc = _run_human_accuracy_experiment()
    print("    Running sub-experiment 3/4: group proportion...")
    r_arr, acc_dict_r = _run_proportion_experiment()
    print("    Running sub-experiment 4/4: group certainty (entropy)...")
    x_vals, acc_adapt_gain, acc_align_gain, acc_comp_gain = \
        _run_certainty_experiment_entropy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5), facecolor='white', sharey=False)

    _plot_performance_gap_on_ax(axes[0], D_arr, methods_acc)
    _plot_human_accuracy_gain_on_ax(axes[1], alpha_vals, team_acc)
    _plot_proportion_gain_on_ax(axes[2], r_arr, acc_dict_r)
    _plot_certainty_gain_on_ax(axes[3], x_vals, acc_adapt_gain,
                                acc_align_gain, acc_comp_gain)

    # Shared legend at top
    all_handles, all_labels = [], []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    by_label = OrderedDict(zip(all_labels, all_handles))
    label_order = ['Single AI', 'Aligned AI', 'Complementary AI', 'Adaptive AI']
    final_handles, final_labels = [], []
    for lbl in label_order:
        if lbl in by_label:
            final_handles.append(by_label.pop(lbl))
            final_labels.append(lbl)
    for lbl, h in by_label.items():
        final_handles.append(h)
        final_labels.append(lbl)

    fig.legend(final_handles, final_labels, loc='upper center',
               ncol=len(final_labels), fontsize=22, frameon=False,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0.03, 1, 0.92], w_pad=3.0)
    out = os.path.join(FIGURES_DIR, 'ca_gains_by_parameter.pdf')
    plt.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


# ===========================================================================
#  B.  WOOFNETTE FIGURES
# ===========================================================================

def _load_woofnette_results():
    """Load WoofNette results produced by woofnette.py."""
    csv_path = os.path.join(WN_RESULTS, 'simulation.csv')
    if not os.path.exists(csv_path):
        print("  [SKIP] No WoofNette results CSV found.")
        return None
    return pd.read_csv(csv_path)


def wn_fig_bar_chart():
    """Bar chart: individual AI accuracy + team accuracy across model types."""
    print("  Generating WoofNette bar chart...")
    df = _load_woofnette_results()
    if df is None:
        return

    pct = lambda s: 100 * s

    human_mean = pct(df['ind_human']).mean()
    human_sem  = pct(df['ind_human']).sem()

    cols_ai  = [c for c in df.columns if c.startswith('ind_') and 'human' not in c]
    cols_hai = [c for c in df.columns if c.startswith('team_')]

    ai_mean, ai_sem   = pct(df[cols_ai]).mean().values, pct(df[cols_ai]).sem().values
    hai_mean, hai_sem  = pct(df[cols_hai]).mean().values, pct(df[cols_hai]).sem().values

    labels = [
        "Standard AI", "Aligned AI", "Complementary AI",
        "Behavior-Aware AI", "Adaptive AI (Oracle)", "Adaptive AI (RRS)",
    ]
    bar_colors = ['#E64B35', '#A1CAF1', '#8491B4', '#E6AB02', '#91D1C2', '#00A087']
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))

    # AI only bars
    ax.bar(x - barw / 2, ai_mean, barw, yerr=ai_sem, color=bar_colors,
           alpha=0.65, capsize=3, label="AI", edgecolor='white', linewidth=1)
    # Human+AI bars
    ax.bar(x + barw / 2, hai_mean, barw, yerr=hai_sem, color=bar_colors,
           alpha=0.95, hatch='//', edgecolor='k', linewidth=0.8,
           capsize=3, label="Human+AI")

    # Human accuracy reference
    lower, upper = human_mean - human_sem, human_mean + human_sem
    ax.axhspan(lower, upper, color='grey', alpha=0.2)
    ax.axhline(human_mean, ls='--', lw=human_lw, color='grey', label="Human")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='center', fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight='bold')
    ax.set_ylim(50, 86)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray',
              fancybox=True, borderpad=0.75, labelspacing=0.5, loc='upper left')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'woofnette_performance.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def wn_fig_team_only_bar_chart():
    """Bar chart: team accuracy only (cover-photo style)."""
    print("  Generating WoofNette team-only bar chart...")
    df = _load_woofnette_results()
    if df is None:
        return

    pct = lambda s: 100 * s
    human_mean = pct(df['ind_human']).mean()
    human_sem  = pct(df['ind_human']).sem()

    cols_ai  = [c for c in df.columns if c.startswith('ind_') and 'human' not in c]
    cols_hai = [c for c in df.columns if c.startswith('team_')]
    ai_mean, ai_sem = pct(df[cols_ai]).mean().values, pct(df[cols_ai]).sem().values
    hai_mean, hai_sem = pct(df[cols_hai]).mean().values, pct(df[cols_hai]).sem().values

    labels = [
        "Standard (Empirical\nRisk Minimization)",
        "Alignment-\nOptimized",
        "Complementarity-\nOptimized",
        "Team-Optimized\n(Behavior-Aware)",
        "Adaptive\nEnsemble (Oracle)",
        "Adaptive\nEnsemble (RRS)",
    ]
    bar_colors = ['#E64B35', '#A1CAF1', '#8491B4', '#E6AB02', '#91D1C2', '#00A087']
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.69 * 0.6, 8.27 * 0.5))

    ax.bar(x, hai_mean, yerr=hai_sem, width=barw, color=bar_colors,
           edgecolor='black', hatch='//', capsize=4, alpha=0.95, linewidth=0.8)

    ax.axhspan(human_mean - human_sem, human_mean + human_sem,
               color='grey', alpha=0.2)
    ax.axhspan(ai_mean[0] - ai_sem[0], ai_mean[0] + ai_sem[0],
               color='#E64B35', alpha=0.15)
    ax.axhline(y=human_mean, color='grey', linestyle='--',
               linewidth=human_lw, alpha=0.8)
    ax.axhline(y=ai_mean[0], color='#E64B35', linestyle='--',
               linewidth=human_lw, alpha=0.8)

    ax.text(len(x) - 4.3, human_mean + 0.3, 'Human Accuracy',
            color='grey', fontsize=10, ha='right', fontweight='bold')
    ax.text(len(x) - 4.05, ai_mean[0] + 0.3, 'AI Accuracy (Standard)',
            color='#E64B35', fontsize=10, ha='right', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha='center', fontsize=9)
    ax.set_xlabel("AI Teammate", fontsize=16, fontweight='bold')
    ax.set_ylabel("Team Accuracy (%)", fontsize=16, fontweight='bold')
    ax.set_ylim(50, 78)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray', axis='y')
    ax.set_facecolor('white'); fig.set_facecolor('white')
    plt.yticks(fontsize=14)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'woofnette_team_performance.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


def wn_fig_ttests():
    """Paired t-tests table for statistical significance."""
    print("  Generating WoofNette paired t-tests table...")
    df = _load_woofnette_results()
    if df is None:
        return

    pct = lambda s: 100 * s
    cols_hai = [c for c in df.columns if c.startswith('team_')]

    records = []
    for a, b in itertools.combinations(cols_hai, 2):
        t, p = ttest_rel(pct(df[a]), pct(df[b]))
        records.append({"model_a": a, "model_b": b, "t_stat": t, "p_value": p})
    df_tests = pd.DataFrame(records).sort_values("p_value")

    out = os.path.join(FIGURES_DIR, 'woofnette_paired_ttests.csv')
    df_tests.to_csv(out, index=False)
    print(f"    Saved: {out}")
    print(df_tests.to_string(index=False))


def wn_fig_samples():
    """WoofNette sample images (2x5 grid)."""
    print("  Generating WoofNette sample images...")
    label_dict = {
        0: 'Church', 1: 'Garbage Truck', 2: 'Gas Pump', 3: 'Golf Ball',
        4: 'Parachute', 5: 'Rhodesian Ridgeback', 6: 'Australian Terrier',
        7: 'Border Terrier', 8: 'Old English Sheepdog', 9: 'Dingo',
    }
    class_to_folder = {
        0: 'church', 1: 'garbage truck', 2: 'gas pump', 3: 'golf ball',
        4: 'parachute', 5: 'Rhodesian ridgeback', 6: 'Australian terrier',
        7: 'Border terrier', 8: 'Old English sheepdog', 9: 'Dingo',
    }

    data_dir = config.get('paths', {}).get('data', '')
    raw_image_dir = config.get('paths', {}).get('woofnette_images')
    if raw_image_dir and not os.path.isabs(raw_image_dir):
        raw_image_dir = os.path.join(PROJECT_DIR, raw_image_dir)

    img_dir = None
    if raw_image_dir and os.path.exists(raw_image_dir):
        img_dir = raw_image_dir
    elif data_dir and os.path.exists(os.path.join(data_dir, 'test')):
        img_dir = os.path.join(data_dir, 'test')
    else:
        print("    [SKIP] No image directory found")
        return

    try:
        from PIL import Image
    except ImportError:
        print("    [SKIP] PIL not available")
        return

    np.random.seed(42)
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    for c in range(10):
        row, col = 0 if c < 5 else 1, c % 5
        ax = axes[row, col]

        class_name = class_to_folder[c]
        class_images = sorted([
            f for f in os.listdir(img_dir)
            if f.startswith(class_name + '_') and f.endswith('.png')
        ])
        img_path = os.path.join(img_dir, np.random.choice(class_images)) if class_images else None

        if img_path is not None:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center',
                    transform=ax.transAxes)

        ax.axis('off')
        ax.set_title(label_dict[c], fontsize=16)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'woofnette_samples.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"    Saved: {out}")


# ===========================================================================
#  MAIN
# ===========================================================================

def generate_ca_figures():
    """Generate all College Admission figures."""
    print("\n" + "=" * 60)
    print("College Admission Figures")
    print("=" * 60)
    ca_fig_data()
    ca_fig_db()
    ca_fig_accuracy_gap_vs_distance()
    ca_fig_accuracy_gap_vs_distance_extended()
    ca_fig_certainty_entropy()
    ca_fig_team_accuracy_gain_vs_human()
    ca_fig_team_accuracy_vs_human()
    ca_fig_proportion()
    ca_fig_weight_sweep()
    ca_fig_tradeoff()


def generate_combined_figure():
    """Generate the combined 1x4 panel figure."""
    print("\n" + "=" * 60)
    print("Combined 1x4 Panel Figure")
    print("=" * 60)
    ca_fig_combined()


def generate_wn_figures():
    """Generate all WoofNette figures."""
    print("\n" + "=" * 60)
    print("WoofNette Figures")
    print("=" * 60)
    wn_fig_bar_chart()
    wn_fig_team_only_bar_chart()
    wn_fig_ttests()
    wn_fig_samples()


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures from saved results.")
    parser.add_argument(
        '--experiment', '-e', type=str, default='all',
        choices=['all', 'ca', 'wn', 'combined'],
        help="Which set of figures to generate (default: all)")
    args = parser.parse_args()

    if args.experiment in ('all', 'ca'):
        generate_ca_figures()
    if args.experiment in ('all', 'combined'):
        generate_combined_figure()
    if args.experiment in ('all', 'wn'):
        generate_wn_figures()

    print("\n" + "=" * 60)
    print(f"All requested figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
