"""College Admission (CAd) experiments for Adaptive AI.

Usage:
    python college_admission.py
    python college_admission.py --experiment specialist_divergence
    python college_admission.py --experiment group_uncertainty
    python college_admission.py --experiment imperfect_human
    python college_admission.py --experiment group_proportion
    python college_admission.py --experiment tradeoff
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from utils import load_config

PROJECT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'college_admission')


def _get_config():
    """Load college_admission defaults from config.yaml."""
    cfg = load_config()
    return cfg.get('college_admission', {})


def generate_data(N, r, delta, tau=0.5, rng=None):
    """Generate synthetic college admission data with aligned/complementary groups.

    Parameters
    ----------
    N : int
        Number of instances.
    r : float
        Proportion of aligned (privileged) group.
    delta : float in [0, 0.5]
        Controls how different the decision boundaries are between groups.
        delta=0 means identical boundaries; delta=0.5 means maximally different.
    tau : float
        Decision threshold.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    X : ndarray of shape (N, 2)
        Features (GPA, SCORE) drawn uniformly from [0, 1].
    y : ndarray of shape (N,)
        Binary labels.
    G : ndarray of shape (N,)
        Group indicators (1 = aligned, 0 = complementary).
    """
    if rng is None:
        rng = np.random.default_rng()

    G = rng.choice([1, 0], size=N, p=[r, 1 - r])
    X = rng.uniform(0, 1, size=(N, 2))
    y = np.zeros(N)
    for i in range(N):
        if G[i] == 1:  # aligned region
            val = (0.5 + delta) * X[i, 1] + (0.5 - delta) * X[i, 0] - tau
        else:           # complementary region
            val = (0.5 + delta) * X[i, 0] + (0.5 - delta) * X[i, 1] - tau
        y[i] = 1 if val >= 0 else 0
    return X, y, G


def fit_logistic(X, y, sample_weight=None, fit_intercept=True):
    """Fit a logistic regression model."""
    model = LogisticRegression(fit_intercept=fit_intercept)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_single_ai(X, y, G, p=0.5, human_pred=None, fit_intercept=True,
                     random_state=None):
    """Train a single AI model per Theorem 4.

    Minimizes g(theta) = p * L_a(theta) + (1-p) * L_c(theta) where:
    - L_a = alignment loss (target = human_pred in D_a)
    - L_c = complementarity loss (target = ground truth in D_c)

    When human is perfect (human_pred is None), this reduces to weighted ERM.

    Parameters
    ----------
    X, y, G : arrays
        Features, labels, group indicators (1=aligned, 0=complementary).
    p : float
        Weight for alignment loss (should match alignment group proportion).
    human_pred : array, optional
        Human predictions. If None, assumes perfect human (human_pred = y in D_a).
    fit_intercept : bool
        Whether to fit intercept.
    random_state : int, optional
        Random state for LogisticRegression.

    Returns
    -------
    model : LogisticRegression
        Trained model.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    G = np.asarray(G)

    n = len(y)
    n_a = np.sum(G == 1)
    n_c = np.sum(G == 0)

    # Construct target: human_pred in D_a, ground truth in D_c
    if human_pred is None:
        target = y.copy()
    else:
        human_pred = np.asarray(human_pred)
        target = np.where(G == 1, human_pred, y)

    # Construct sample weights: p/n_a for D_a, (1-p)/n_c for D_c
    sample_weight = np.zeros(n)
    if n_a > 0:
        sample_weight[G == 1] = p / n_a
    if n_c > 0:
        sample_weight[G == 0] = (1 - p) / n_c
    sample_weight = sample_weight * n  # scale for numerical stability

    model = LogisticRegression(
        fit_intercept=fit_intercept,
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X, target, sample_weight=sample_weight)
    return model


def compute_group_probs(G, certainty):
    """Assign per-instance probability p(x) = Pr(D_a | x).

    Symmetric: G=1 gets p=certainty, G=0 gets p=1-certainty.
    """
    return np.where(G == 1, certainty, 1 - certainty)


def adaptive_predict(X, p_test, m_a, m_c, rng=None, stochastic=True):
    """Adaptive predictor that routes instances by p_test.

    Parameters
    ----------
    X : array
        Feature matrix.
    p_test : array
        Per-instance probability of belonging to the aligned group.
    m_a, m_c : classifiers
        Aligned and complementary specialist models (must have predict_proba).
    rng : numpy.random.Generator, optional
        RNG for stochastic routing.
    stochastic : bool
        If True, use Bernoulli(p_test) routing; else hard threshold at 0.5.

    Returns
    -------
    proba : array
        Predicted probability of positive class.
    """
    pa = m_a.predict_proba(X)[:, 1]
    pc = m_c.predict_proba(X)[:, 1]
    if stochastic:
        if rng is None:
            rng = np.random.default_rng()
        route_to_a = rng.random(len(p_test)) < p_test
    else:
        route_to_a = (p_test >= 0.5)
    return np.where(route_to_a, pa, pc)


def adaptive_predict_hard(X, selector, ma, mc):
    """Hard routing: pick ma if selector==1, else mc.

    Returns binary predictions.
    """
    pa = ma.predict_proba(X)[:, 1]
    pc = mc.predict_proba(X)[:, 1]
    return np.where(selector == 1, pa, pc) >= 0.5


def get_intercept(model):
    """Extract intercept as a scalar."""
    return np.array(model.intercept_).ravel()[0]


def get_coefficients(model):
    """Extract flattened coefficient vector."""
    return model.coef_.flatten()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_logistic_loss(theta, X_aug, y):
    """Compute logistic (cross-entropy) loss for augmented X and parameter vector."""
    logits = X_aug.dot(theta)
    probs = sigmoid(logits)
    return log_loss(y, probs)


def add_bias_column(X):
    """Add a column of ones for intercept handling."""
    return np.c_[np.ones((X.shape[0], 1)), X]


# ---------------------------------------------------------------------------
# Human simulation and CGPR team model
# ---------------------------------------------------------------------------

def get_decision_human(y_test, G_test, acc_align, acc_comp,
                       confs_max=(0.1, 0.1), rng=None):
    """Simulate imperfect human decisions.

    Parameters
    ----------
    y_test : array
        Ground truth labels.
    G_test : array
        Group indicators.
    acc_align : float
        Human accuracy in the aligned group.
    acc_comp : float
        Human accuracy in the complementary group.
    confs_max : tuple of (float, float)
        (conf_over_max, conf_under_max) for confidence noise.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    h_pred : array of int
        Human predictions.
    h_conf : array of float
        Human confidence scores.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(y_test)
    h_pred = np.zeros(N, dtype=int)
    h_conf = np.zeros(N, dtype=float)
    conf_over_max, conf_under_max = confs_max

    for i in range(N):
        p = acc_align if G_test[i] == 1 else acc_comp
        correct = rng.random() < p
        h_pred[i] = y_test[i] if correct else 1 - y_test[i]
        low = max(p - conf_under_max, 0.0)
        high = min(p + conf_over_max, 1.0)
        if high < low:
            high = low
        h_conf[i] = rng.uniform(low, high)

    h_conf = np.clip(h_conf, 0.0, 1.0)
    return h_pred, h_conf


def get_team_decisions_cgpr(h_pred, h_conf, ai_pred, r, tau=0.5, seed=0):
    """CGPR team decision model.

    If human confidence > tau, use human prediction.
    Otherwise, use AI prediction with probability r, else human.

    Parameters
    ----------
    h_pred : array
        Human predictions.
    h_conf : array
        Human confidence scores.
    ai_pred : array
        AI predictions.
    r : float
        Reliance rate (agreement between human and AI in alignment region).
    tau : float
        Confidence threshold.
    seed : int
        Random seed.

    Returns
    -------
    team : array of int
        Team decisions.
    """
    rng = np.random.default_rng(seed)
    N = len(h_pred)
    team = np.zeros(N, dtype=int)
    for i in range(N):
        if h_conf[i] > tau:
            team[i] = h_pred[i]
        else:
            team[i] = ai_pred[i] if rng.random() < r else h_pred[i]
    return team


# ===========================================================================
# Experiments
# ===========================================================================

def run_specialist_divergence():
    """Sweep delta, compute accuracy gap vs specialist distance.

    Also runs a multi-model variant with LR, DT, RF, MLP, SVM.
    Saves two CSVs: specialist_divergence.csv and specialist_divergence_multi.csv.
    """
    print("Running specialist_divergence experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    tau = cfg.get('tau', 0.5)
    test_size = cfg.get('test_size', 0.3)
    RAND_STATE = cfg.get('random_state', 42)
    r = 0.5
    delta_values = np.linspace(0, 0.5, 10)

    rng = np.random.default_rng(RAND_STATE)

    # ----- Single-model (LogisticRegression) sweep -----
    rows = []
    for delta in delta_values:
        X, y, G = generate_data(N, r, delta, tau=tau, rng=rng)
        X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
            X, y, G, test_size=test_size, stratify=G, random_state=RAND_STATE)

        mdl_a = fit_logistic(X_tr[G_tr == 1], y_tr[G_tr == 1])
        mdl_c = fit_logistic(X_tr[G_tr == 0], y_tr[G_tr == 0])
        mdl_s = fit_logistic(X_tr, y_tr)

        # Specialist distance (normalized weight vectors)
        wa = get_coefficients(mdl_a)
        wc = get_coefficients(mdl_c)
        wa_n = wa / (norm(wa) + 1e-12)
        wc_n = wc / (norm(wc) + 1e-12)
        D = norm(wa_n - wc_n)

        # Baseline
        acc_s = accuracy_score(y_te, mdl_s.predict(X_te)) * 100
        loss_s = log_loss(y_te, mdl_s.predict_proba(X_te)[:, 1])

        # Adaptive AI (oracle routing)
        p_adapt = np.where(
            G_te == 1,
            mdl_a.predict_proba(X_te)[:, 1],
            mdl_c.predict_proba(X_te)[:, 1])
        acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
        loss_a = log_loss(y_te, p_adapt)

        # Aligned-only
        acc_align = accuracy_score(y_te, mdl_a.predict(X_te)) * 100
        loss_align = log_loss(y_te, mdl_a.predict_proba(X_te)[:, 1])

        # Complementary-only
        acc_comp = accuracy_score(y_te, mdl_c.predict(X_te)) * 100
        loss_comp = log_loss(y_te, mdl_c.predict_proba(X_te)[:, 1])

        rows.append({
            'delta': delta,
            'D': D,
            'acc_single': acc_s,
            'acc_adaptive': acc_a,
            'acc_aligned': acc_align,
            'acc_complementary': acc_comp,
            'gap_acc_adaptive': acc_a - acc_s,
            'gap_acc_aligned': acc_align - acc_s,
            'gap_acc_complementary': acc_comp - acc_s,
            'loss_single': loss_s,
            'loss_adaptive': loss_a,
            'loss_aligned': loss_align,
            'loss_complementary': loss_comp,
            'gap_loss_adaptive': loss_s - loss_a,
            'gap_loss_aligned': loss_s - loss_align,
            'gap_loss_complementary': loss_s - loss_comp,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'specialist_divergence.csv'), index=False)
    print(f"  Saved specialist_divergence.csv ({len(df)} rows)")

    # ----- Multi-model variant -----
    RAND_STATE_MULTI = 0
    rng_multi = np.random.default_rng(RAND_STATE_MULTI)

    model_specs = {
        'Logistic Regression': LogisticRegression(fit_intercept=True,
                                                   random_state=RAND_STATE_MULTI),
        'Decision Tree': DecisionTreeClassifier(random_state=RAND_STATE_MULTI),
        'Random Forest': RandomForestClassifier(n_estimators=100,
                                                 random_state=RAND_STATE_MULTI),
        'MLP': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=100,
                              random_state=RAND_STATE_MULTI),
        'SVM (RBF)': SVC(kernel='rbf', probability=True,
                          random_state=RAND_STATE_MULTI),
    }

    rows_multi = []
    for name, base_model in model_specs.items():
        for delta in delta_values:
            X, y, G = generate_data(N, r, delta, tau=tau, rng=rng_multi)
            X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
                X, y, G, test_size=test_size, stratify=G,
                random_state=RAND_STATE_MULTI)

            mdl_s = clone(base_model)
            mdl_a = clone(base_model)
            mdl_c = clone(base_model)

            # The MLP sweep is only used for qualitative trend verification in the
            # extended figure, so we suppress non-fatal convergence warnings here.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                mdl_s.fit(X_tr, y_tr)
                mdl_a.fit(X_tr[G_tr == 1], y_tr[G_tr == 1])
                mdl_c.fit(X_tr[G_tr == 0], y_tr[G_tr == 0])

            # Baseline
            acc_s = accuracy_score(y_te, mdl_s.predict(X_te)) * 100

            # Adaptive AI (oracle)
            p_adapt = np.where(
                G_te == 1,
                mdl_a.predict_proba(X_te)[:, 1],
                mdl_c.predict_proba(X_te)[:, 1])
            acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100

            # Specialist distance (probability-based RMS)
            p_a_ref = mdl_a.predict_proba(X_te)[:, 1]
            p_c_ref = mdl_c.predict_proba(X_te)[:, 1]
            D_prob = np.linalg.norm(p_a_ref - p_c_ref) / np.sqrt(len(p_a_ref))

            rows_multi.append({
                'model': name,
                'delta': delta,
                'D_prob': D_prob,
                'acc_single': acc_s,
                'acc_adaptive': acc_a,
                'gap_acc_adaptive': acc_a - acc_s,
            })

        print(f"  {name} done")

    df_multi = pd.DataFrame(rows_multi)
    df_multi.to_csv(os.path.join(RESULTS_DIR, 'specialist_divergence_multi.csv'),
                    index=False)
    print(f"  Saved specialist_divergence_multi.csv ({len(df_multi)} rows)")
    return df


def run_group_uncertainty():
    """Sweep group certainty from 0.5 to 1.0 with stochastic routing.

    Saves group_uncertainty.csv with entropy-based x-axis values.
    """
    print("Running group_uncertainty experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    delta = cfg.get('delta', 0.25)
    tau = cfg.get('tau', 0.5)
    RAND_STATE = cfg.get('random_state', 42)
    r = 0.5
    certainties = np.linspace(0.5, 1.0, 10)

    rng = np.random.default_rng(RAND_STATE)

    X, y, G = generate_data(N, r, delta, tau=tau, rng=rng)

    # Pre-train baseline and perfect specialists
    mdl_single = fit_logistic(X, y)
    mdl_align_per = fit_logistic(X[G == 1], y[G == 1])
    mdl_comp_per = fit_logistic(X[G == 0], y[G == 0])

    # Static baselines
    acc_single = accuracy_score(y, mdl_single.predict(X)) * 100
    loss_single = log_loss(y, mdl_single.predict_proba(X)[:, 1])
    acc_align = accuracy_score(y, mdl_align_per.predict(X)) * 100
    loss_align = log_loss(y, mdl_align_per.predict_proba(X)[:, 1])
    acc_comp = accuracy_score(y, mdl_comp_per.predict(X)) * 100
    loss_comp = log_loss(y, mdl_comp_per.predict_proba(X)[:, 1])

    rows = []
    rng_route = np.random.default_rng(0)

    for cert in certainties:
        p_test = compute_group_probs(G, cert)

        # Normalized entropy
        eps = 1e-12
        pclip = np.clip(p_test, eps, 1 - eps)
        H = (-pclip * np.log(pclip) - (1 - pclip) * np.log(1 - pclip)).mean()
        H_norm = H / (2 * np.log(2.0))
        group_cert = 1.0 - H_norm

        # Weighted specialists
        mdl_align_w = fit_logistic(X, y, sample_weight=p_test)
        mdl_comp_w = fit_logistic(X, y, sample_weight=(1 - p_test))

        # Stochastic routing
        pw = adaptive_predict(X, p_test, mdl_align_w, mdl_comp_w,
                              rng=rng_route, stochastic=True)
        loss_w = log_loss(y, pw)
        acc_w = accuracy_score(y, (pw >= 0.5).astype(int)) * 100

        rows.append({
            'certainty': cert,
            'H_norm': H_norm,
            'group_cert': group_cert,
            'acc_single': acc_single,
            'acc_aligned': acc_align,
            'acc_complementary': acc_comp,
            'acc_adaptive': acc_w,
            'loss_single': loss_single,
            'loss_aligned': loss_align,
            'loss_complementary': loss_comp,
            'loss_adaptive': loss_w,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'group_uncertainty.csv'), index=False)
    print(f"  Saved group_uncertainty.csv ({len(df)} rows)")
    return df


def run_imperfect_human():
    """Sweep alpha (human accuracy in aligned group) with CGPR team model.

    Evaluates team accuracy for adaptive vs single AI under imperfect human.
    Saves imperfect_human.csv.
    """
    print("Running imperfect_human experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    delta = cfg.get('delta', 0.25)
    tau = cfg.get('tau', 0.5)
    RAND_STATE = cfg.get('random_state', 42)
    r_prop = 0.5
    acc_comp_val = 0.5
    alpha_vals = np.linspace(0.5, 1.0, 11)

    rng = np.random.default_rng(RAND_STATE)

    X, y, G = generate_data(N, r_prop, delta, tau, rng=rng)
    X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
        X, y, G, test_size=0.3, random_state=RAND_STATE)

    # Complementary specialist (trained once, uses ground truth)
    m_comp = fit_logistic(X_tr[G_tr == 0], y_tr[G_tr == 0])

    rows = []
    for acc_align in alpha_vals:
        # Simulate imperfect human on both splits
        h_tr, conf_tr = get_decision_human(
            y_tr, G_tr, acc_align=acc_align, acc_comp=acc_comp_val, rng=rng)
        h_te, conf_te = get_decision_human(
            y_te, G_te, acc_align=acc_align, acc_comp=acc_comp_val, rng=rng)

        # Train Single AI per Theorem 4
        m_single = train_single_ai(
            X_tr, y_tr, G_tr, p=r_prop, human_pred=h_tr, random_state=RAND_STATE)

        # Aligned specialist (trained on human predictions in D_a)
        m_align = fit_logistic(X_tr[G_tr == 1], h_tr[G_tr == 1])
        pred_perf = adaptive_predict_hard(X_te, G_te, m_align, m_comp)

        # Weighted specialists (confidence-based)
        m_wa = fit_logistic(X_tr, h_tr, sample_weight=conf_tr)
        m_wc = fit_logistic(X_tr, y_tr, sample_weight=1 - conf_tr)
        pred_wt = adaptive_predict_hard(X_te, G_te, m_wa, m_wc)

        # Compute reliance r (alignment region)
        da_mask = G_te == 1
        r_single = 1 - np.mean(
            h_te[da_mask] != m_single.predict(X_te[da_mask]))
        r_align = 1 - np.mean(
            h_te[da_mask] != m_align.predict(X_te[da_mask]))
        r_comp_val = 1 - np.mean(
            h_te[da_mask] != m_comp.predict(X_te[da_mask]))
        r_perf = 1 - np.mean(
            h_te[da_mask] != pred_perf[da_mask])
        r_wt = 1 - np.mean(
            h_te[da_mask] != pred_wt[da_mask])

        # CGPR team decisions
        team_perf = accuracy_score(y_te, get_team_decisions_cgpr(
            h_te, conf_te, pred_perf.astype(int), r_perf, tau))
        team_wt = accuracy_score(y_te, get_team_decisions_cgpr(
            h_te, conf_te, pred_wt.astype(int), r_wt, tau))
        team_single = accuracy_score(y_te, get_team_decisions_cgpr(
            h_te, conf_te, m_single.predict(X_te), r_single, tau))
        team_align = accuracy_score(y_te, get_team_decisions_cgpr(
            h_te, conf_te, m_align.predict(X_te), r_align, tau))
        team_comp = accuracy_score(y_te, get_team_decisions_cgpr(
            h_te, conf_te, m_comp.predict(X_te), r_comp_val, tau))

        rows.append({
            'alpha': acc_align,
            'team_adaptive_perfect': team_perf,
            'team_adaptive_weighted': team_wt,
            'team_single_ai': team_single,
            'team_aligned_ai': team_align,
            'team_complementary_ai': team_comp,
            'r_single': r_single,
            'r_perf': r_perf,
            'r_wt': r_wt,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'imperfect_human.csv'), index=False)
    print(f"  Saved imperfect_human.csv ({len(df)} rows)")
    return df


def run_group_proportion():
    """Sweep r (aligned group fraction), accuracy gap vs proportion.

    Saves group_proportion.csv.
    """
    print("Running group_proportion experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    delta_fixed = cfg.get('delta', 0.25)
    tau = cfg.get('tau', 0.5)
    test_size = cfg.get('test_size', 0.3)
    RAND_STATE = cfg.get('random_state', 42)
    r_values = np.linspace(0.05, 0.95, 20)

    rng = np.random.default_rng(RAND_STATE)

    rows = []
    for r in r_values:
        X, y, G = generate_data(N, r, delta_fixed, tau=tau, rng=rng)
        X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
            X, y, G, test_size=test_size, stratify=G, random_state=RAND_STATE)

        X_tr_a, y_tr_a = X_tr[G_tr == 1], y_tr[G_tr == 1]
        X_tr_c, y_tr_c = X_tr[G_tr == 0], y_tr[G_tr == 0]

        # Skip if insufficient data for specialists
        if (len(y_tr_a) < 2 or len(np.unique(y_tr_a)) < 2 or
                len(y_tr_c) < 2 or len(np.unique(y_tr_c)) < 2):
            print(f"  Skipping r={r:.2f}: insufficient data for specialists")
            continue

        # Single AI per Theorem 4 (perfect human)
        mdl_s = train_single_ai(X_tr, y_tr, G_tr, p=r, human_pred=None,
                                fit_intercept=True, random_state=RAND_STATE)
        mdl_a = fit_logistic(X_tr_a, y_tr_a)
        mdl_c = fit_logistic(X_tr_c, y_tr_c)

        # Evaluate
        acc_s = accuracy_score(y_te, mdl_s.predict(X_te)) * 100

        p_adapt = np.where(
            G_te == 1,
            mdl_a.predict_proba(X_te)[:, 1],
            mdl_c.predict_proba(X_te)[:, 1])
        acc_a = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
        acc_align = accuracy_score(y_te, mdl_a.predict(X_te)) * 100
        acc_comp = accuracy_score(y_te, mdl_c.predict(X_te)) * 100

        rows.append({
            'r': r,
            'acc_single': acc_s,
            'acc_adaptive': acc_a,
            'acc_aligned': acc_align,
            'acc_complementary': acc_comp,
            'gap_acc_adaptive': acc_a - acc_s,
            'gap_acc_aligned': acc_align - acc_s,
            'gap_acc_complementary': acc_comp - acc_s,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'group_proportion.csv'), index=False)
    print(f"  Saved group_proportion.csv ({len(df)} rows)")
    return df


def run_weight_sweep():
    """Sweep fixed single-model weights and compare against the standard model.

    Saves weight_sweep.csv with accuracy and log-loss gaps relative to the
    standard single model trained with weight w=r.
    """
    print("Running weight_sweep experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    delta = cfg.get('delta', 0.25)
    tau = cfg.get('tau', 0.5)
    test_size = cfg.get('test_size', 0.3)
    RAND_STATE = cfg.get('random_state', 42)

    r_values = np.linspace(0.05, 0.95, 20)
    fixed_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(RAND_STATE)

    rows = []
    for r in r_values:
        X, y, G = generate_data(N, r, delta, tau=tau, rng=rng)
        X_tr, X_te, y_tr, y_te, G_tr, G_te = train_test_split(
            X, y, G, test_size=test_size, stratify=G, random_state=RAND_STATE)

        X_tr_a, y_tr_a = X_tr[G_tr == 1], y_tr[G_tr == 1]
        X_tr_c, y_tr_c = X_tr[G_tr == 0], y_tr[G_tr == 0]
        if (len(y_tr_a) < 2 or len(np.unique(y_tr_a)) < 2 or
                len(y_tr_c) < 2 or len(np.unique(y_tr_c)) < 2):
            print(f"  Skipping r={r:.2f}: insufficient data for specialists")
            continue

        mdl_standard = train_single_ai(
            X_tr, y_tr, G_tr, p=r, human_pred=None,
            fit_intercept=True, random_state=RAND_STATE)
        mdl_align = fit_logistic(X_tr_a, y_tr_a)
        mdl_comp = fit_logistic(X_tr_c, y_tr_c)

        acc_standard = accuracy_score(y_te, mdl_standard.predict(X_te)) * 100
        loss_standard = log_loss(y_te, mdl_standard.predict_proba(X_te)[:, 1])

        p_adapt = np.where(
            G_te == 1,
            mdl_align.predict_proba(X_te)[:, 1],
            mdl_comp.predict_proba(X_te)[:, 1])
        acc_adapt = accuracy_score(y_te, (p_adapt >= 0.5).astype(int)) * 100
        loss_adapt = log_loss(y_te, p_adapt)
        rows.append({
            'r': r,
            'model': 'adaptive',
            'fixed_weight': np.nan,
            'acc': acc_adapt,
            'loss': loss_adapt,
            'acc_gain_vs_standard': acc_adapt - acc_standard,
            'loss_gain_vs_standard': loss_standard - loss_adapt,
        })

        for w in fixed_weights:
            mdl_w = train_single_ai(
                X_tr, y_tr, G_tr, p=w, human_pred=None,
                fit_intercept=True, random_state=RAND_STATE)
            acc_w = accuracy_score(y_te, mdl_w.predict(X_te)) * 100
            loss_w = log_loss(y_te, mdl_w.predict_proba(X_te)[:, 1])
            rows.append({
                'r': r,
                'model': f'w={w:.2f}',
                'fixed_weight': w,
                'acc': acc_w,
                'loss': loss_w,
                'acc_gain_vs_standard': acc_w - acc_standard,
                'loss_gain_vs_standard': loss_standard - loss_w,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'weight_sweep.csv'), index=False)
    print(f"  Saved weight_sweep.csv ({len(df)} rows)")
    return df


def run_tradeoff():
    """Trade-off ratio vs specialization for multiple alpha values.

    Computes the local trade-off ratio d L_c / d L_a along the direction
    from the single model to the aligned specialist.

    Saves tradeoff.csv.
    """
    print("Running tradeoff experiment...")
    cfg = _get_config()
    N = cfg.get('n_instances', 10000)
    delta_val = cfg.get('delta', 0.25)
    tau = cfg.get('tau', 0.5)
    RAND_STATE = cfg.get('random_state', 42)

    r_values = np.linspace(0.5, 0.99, 10)
    alpha_values = [0.6, 0.8, 1.0]
    epsilon = 1e-6

    rng = np.random.default_rng(RAND_STATE)

    rows = []
    for alpha in alpha_values:
        for r in r_values:
            X, y, G = generate_data(N, r, delta_val, tau=tau, rng=rng)

            X_aligned = X[G == 1]
            y_aligned = y[G == 1]
            X_complementary = X[G == 0]
            y_complementary = y[G == 0]

            mdl_aligned = fit_logistic(X_aligned, y_aligned)
            mdl_single = fit_logistic(X, y)

            b_aligned = get_intercept(mdl_aligned)
            w_aligned = get_coefficients(mdl_aligned)
            b_single = get_intercept(mdl_single)
            w_single = get_coefficients(mdl_single)

            # Distance between single and aligned weight vectors (normalized)
            w_a_n = w_aligned / (norm(w_aligned) + 1e-12)
            w_s_n = w_single / (norm(w_single) + 1e-12)
            distance = norm(w_s_n - w_a_n)

            # Finite-difference gradient computation
            X_aligned_aug = add_bias_column(X_aligned)
            X_complementary_aug = add_bias_column(X_complementary)
            theta_single = np.concatenate(([b_single], w_single))
            theta_aligned = np.concatenate(([b_aligned], w_aligned))

            base_loss_a = compute_logistic_loss(
                theta_single, X_aligned_aug, y_aligned)
            base_loss_c = compute_logistic_loss(
                theta_single, X_complementary_aug, y_complementary)

            grad_aligned = np.zeros_like(theta_single)
            grad_complementary = np.zeros_like(theta_single)
            for i in range(theta_single.size):
                theta_pert = theta_single.copy()
                theta_pert[i] += epsilon
                loss_a = compute_logistic_loss(
                    theta_pert, X_aligned_aug, y_aligned)
                loss_c = compute_logistic_loss(
                    theta_pert, X_complementary_aug, y_complementary)
                grad_aligned[i] = (loss_a - base_loss_a) / epsilon
                grad_complementary[i] = (loss_c - base_loss_c) / epsilon

            # Local trade-off ratio
            direction = theta_aligned - theta_single
            direction /= (norm(direction) + 1e-12)
            da = grad_aligned.dot(direction)
            dc = grad_complementary.dot(direction)
            tradeoff = -dc / (da * (2 * alpha - 1) + 1e-12)

            rows.append({
                'alpha': alpha,
                'r': r,
                'distance': distance,
                'tradeoff': tradeoff,
                'da': da,
                'dc': dc,
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'tradeoff.csv'), index=False)
    print(f"  Saved tradeoff.csv ({len(df)} rows)")
    return df


# ===========================================================================
# CLI
# ===========================================================================

EXPERIMENTS = {
    'specialist_divergence': run_specialist_divergence,
    'group_uncertainty': run_group_uncertainty,
    'imperfect_human': run_imperfect_human,
    'group_proportion': run_group_proportion,
    'weight_sweep': run_weight_sweep,
    'tradeoff': run_tradeoff,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='College Admission experiments for Adaptive AI (AAAI 2026)')
    parser.add_argument(
        '--experiment', type=str, default='all',
        choices=['all'] + list(EXPERIMENTS.keys()),
        help='Which experiment to run (default: all)')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.experiment == 'all':
        for name, func in EXPERIMENTS.items():
            func()
    else:
        EXPERIMENTS[args.experiment]()

    print("\nAll done!")
