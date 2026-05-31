"""WoofNette experiment for Adaptive AI.

Trains and evaluates multiple AI model types (baseline, aligned, complementary,
behavior-aware, adaptive-oracle, adaptive-RRS) across multiple seeds.
Results are saved as CSVs in results/woofnette/.

Usage:
    python woofnette.py
    python woofnette.py --seeds 0
    python woofnette.py --seeds 0 1
    python woofnette.py --skip-existing
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import (
    load_config, set_all_seeds, require_directory, resolve_optional_path,
    build_classifier, set_max_to_one, acc_ohe,
    simulate_human, compute_instance_weights, get_threshold_distribution,
    team_loss_exact_cgpr, true_top1_acc, human_agreement,
    oracle_route, rrs_route,
    cgpr_team_static, cgpr_team_online,
    make_accuracy_threshold_callback, log_seed_results,
)

PROJECT_DIR = os.path.dirname(__file__)
RESULTS_DIR_DEFAULT = os.path.join(PROJECT_DIR, 'results', 'woofnette')


def ensure_woofnette_arrays(data_dir):
    """Require the shared WoofNette arrays to be present in data_dir."""
    required = ('X_train.npy', 'X_test.npy', 'Y_train.npy', 'Y_test.npy')
    missing = [
        name for name in required
        if not os.path.exists(os.path.join(data_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Missing WoofNette array files in data directory: "
            + ', '.join(missing)
            + ". Place X_train.npy, X_test.npy, Y_train.npy, and Y_test.npy "
            "under paths.data before running the simulation."
        )


def resolve_proxy_checkpoint(config, data_dir):
    """Resolve the routing-proxy checkpoint used for WoofNette weighting."""
    proxy_cfg = config['woofnette']['confidence_predictor']
    proxy_target = proxy_cfg.get('target', proxy_cfg.get('model_type', 'agreement'))
    explicit_path = resolve_optional_path(proxy_cfg.get('model_path'), base_dir=PROJECT_DIR)
    if explicit_path:
        return explicit_path, proxy_target, 'config:model_path'

    backbone = proxy_cfg['backbone']
    model_type = proxy_cfg.get('model_type', 'confidence')
    prefix = 'm_conf' if model_type == 'confidence' else 'm_agr'
    return os.path.join(data_dir, f'{prefix}_{backbone}.keras'), proxy_target, 'config:model_type'


def load_confusion_matrix(cm_path, classes):
    """Load and normalize the pilot-study confusion matrix."""
    df_counts = pd.read_csv(cm_path, index_col=0)
    df_norm = df_counts.div(df_counts.sum(axis=1), axis=0)
    return df_norm.loc[classes, classes].values


def get_predictions_and_confidence(model, X, batch_size=512):
    """Get hard predictions (one-hot) and max-confidence from a model."""
    probs = model.predict(X, batch_size=batch_size)
    preds = set_max_to_one(probs)
    confs = probs.max(axis=1)
    return preds, confs


def train_baseline(X_tr, Y_tr, X_te, Y_te, config, seed, skip_existing=False,
                   model_dir=None):
    """Train standard baseline AI on true labels."""
    dnn = config['woofnette']['backbone']
    K = config['woofnette']['num_classes']
    bs = config['woofnette']['batch_size']
    acc_thres = config['woofnette']['acc_threshold']

    model_path = os.path.join(model_dir, f'm_r_{dnn}_{seed}.keras') if model_dir else None

    if skip_existing and model_path and os.path.exists(model_path):
        print(f"  Loading baseline from {os.path.basename(model_path)}")
        return tf.keras.models.load_model(model_path)

    print(f"  Training Baseline AI (seed={seed})")
    set_all_seeds(seed)
    model = build_classifier(K, backbone=dnn)
    cb = make_accuracy_threshold_callback(monitor='accuracy', threshold=acc_thres)
    model.fit(X_tr, Y_tr, epochs=config['woofnette']['max_epochs'],
              batch_size=bs, validation_data=(X_te, Y_te), verbose=2,
              callbacks=[cb])

    if model_path:
        model.save(model_path)
    return model


def train_complementary(X_tr, Y_tr, X_te, Y_te, w_c, config, seed,
                        skip_existing=False, model_dir=None):
    """Train complementary AI: weighted toward low-confidence instances."""
    dnn = config['woofnette']['backbone']
    K = config['woofnette']['num_classes']
    bs = config['woofnette']['batch_size']
    acc_thres = config['woofnette']['acc_threshold']
    td = 'uniform_6_8'

    model_path = os.path.join(model_dir, f'm_c_{td}_{dnn}_{seed}.keras') if model_dir else None

    if skip_existing and model_path and os.path.exists(model_path):
        print(f"  Loading complementary from {os.path.basename(model_path)}")
        return tf.keras.models.load_model(model_path)

    print(f"  Training Complementary AI (seed={seed})")
    set_all_seeds(seed)
    model = build_classifier(K, backbone=dnn)
    cb = make_accuracy_threshold_callback(monitor='accuracy', threshold=acc_thres)
    model.fit(X_tr, Y_tr, epochs=config['woofnette']['max_epochs'],
              batch_size=bs, validation_data=(X_te, Y_te), verbose=2,
              sample_weight=w_c, callbacks=[cb])

    if model_path:
        model.save(model_path)
    return model


def train_aligned(X_tr, h_preds_tr, X_te, Y_te, w_a, config, seed,
                  skip_existing=False, model_dir=None):
    """Train aligned AI: trained on human predictions, weighted toward high-confidence."""
    dnn = config['woofnette']['backbone']
    K = config['woofnette']['num_classes']
    bs = config['woofnette']['batch_size']
    acc_thres = config['woofnette']['acc_threshold']
    td = 'uniform_6_8'

    model_path = os.path.join(model_dir, f'm_a_{td}_{dnn}_{seed}.keras') if model_dir else None

    if skip_existing and model_path and os.path.exists(model_path):
        print(f"  Loading aligned from {os.path.basename(model_path)}")
        return tf.keras.models.load_model(model_path)

    print(f"  Training Aligned AI (seed={seed})")
    set_all_seeds(seed)
    model = build_classifier(K, backbone=dnn)
    cb = make_accuracy_threshold_callback(monitor='accuracy', threshold=acc_thres)
    model.fit(X_tr, h_preds_tr, epochs=config['woofnette']['max_epochs'],
              batch_size=bs, validation_data=(X_te, Y_te), verbose=2,
              sample_weight=w_a, callbacks=[cb])

    if model_path:
        model.save(model_path)
    return model


def train_behavior_aware(X_tr, Y_pack_tr, X_te, Y_pack_te, L_c_h, config, seed,
                         skip_existing=False, model_dir=None):
    """Train behavior-aware single AI with the CGPR team loss."""
    dnn = config['woofnette']['backbone']
    K = config['woofnette']['num_classes']
    bs = config['woofnette']['batch_size']
    acc_thres = config['woofnette']['acc_threshold']
    td = 'uniform_6_8'

    loss_fn = team_loss_exact_cgpr(tf.constant(L_c_h, dtype=tf.float32), K)
    model_path = os.path.join(model_dir, f'm_s_{td}_{dnn}_{seed}.keras') if model_dir else None

    if skip_existing and model_path and os.path.exists(model_path):
        print(f"  Loading behavior-aware from {os.path.basename(model_path)}")
        keras_obj = {"team_loss": loss_fn, "true_top1_acc": true_top1_acc,
                     "human_agreement": human_agreement}
        return tf.keras.models.load_model(model_path, custom_objects=keras_obj)

    print(f"  Training Behavior-Aware AI (seed={seed})")
    set_all_seeds(seed)
    model = build_classifier(K, backbone=dnn, return_compiled=False)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss=loss_fn,
        metrics=[true_top1_acc, human_agreement])

    cb = make_accuracy_threshold_callback(monitor='true_top1_acc', threshold=acc_thres)
    model.fit(X_tr, Y_pack_tr, epochs=config['woofnette']['max_epochs'],
              batch_size=bs, validation_data=(X_te, Y_pack_te), verbose=2,
              callbacks=[cb])

    if model_path:
        model.save(model_path)
    return model


def run_simulation(seed_indices=None, skip_existing=False, results_dir=None):
    """Run the full WoofNette simulation."""
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.optimizer.set_jit(False)

    config = load_config()
    data_dir = require_directory(config['paths']['data'], 'data')
    ensure_woofnette_arrays(data_dir)
    model_dir = require_directory(
        config['paths']['models'],
        'models',
        create=True,
    )
    classes = config['classes']
    K = config['woofnette']['num_classes']
    sim_cfg = config['woofnette']['simulation']
    t_low = sim_cfg['threshold_low']
    t_high = sim_cfg['threshold_high']
    team_model = sim_cfg['team_model']
    bs = config['woofnette']['batch_size']
    results_dir = results_dir or RESULTS_DIR_DEFAULT

    seeds = sim_cfg['seeds']
    if seed_indices is not None:
        seeds = [seeds[i] for i in seed_indices]

    # Load data
    print("Loading WoofNette data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)
    Y_train_raw = np.load(os.path.join(data_dir, 'Y_train.npy'), allow_pickle=True)
    Y_test_raw = np.load(os.path.join(data_dir, 'Y_test.npy'), allow_pickle=True)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Convert string labels to integer indices
    label_dict = {c: i for i, c in enumerate(classes)}
    if Y_train_raw.dtype.kind in ('U', 'S', 'O'):
        ds_tr_info = np.array([label_dict[y] for y in Y_train_raw])
        ds_te_info = np.array([label_dict[y] for y in Y_test_raw])
        Y_train = ds_tr_info.copy()
        Y_test = ds_te_info.copy()
    else:
        Y_train = Y_train_raw.astype(int)
        Y_test = Y_test_raw.astype(int)
        ds_tr_info = Y_train.copy()
        ds_te_info = Y_test.copy()

    Y_tr_ohe = tf.keras.utils.to_categorical(Y_train, K)
    Y_te_ohe = tf.keras.utils.to_categorical(Y_test, K)

    # Load confusion matrix
    cm_path = os.path.join(PROJECT_DIR, config['paths']['pilot_cm'])
    conf_matrix = load_confusion_matrix(cm_path, classes)

    # Load the routing proxy used for instance weighting.
    proxy_model_path, proxy_target, proxy_source = resolve_proxy_checkpoint(
        config, data_dir)
    if not os.path.exists(proxy_model_path):
        raise FileNotFoundError(
            "Routing proxy checkpoint not found: "
            f"{proxy_model_path}. Update "
            "`woofnette.confidence_predictor.model_path` in config.yaml "
            "to the agreement-trained checkpoint you want to reuse."
        )
    print(
        "  Loading "
        f"{proxy_target}-trained routing proxy from "
        f"{os.path.basename(proxy_model_path)} ({proxy_source})..."
    )
    m_proxy = tf.keras.models.load_model(proxy_model_path)
    print("  Computing predicted confidence...")
    h_conf_tr = np.squeeze(m_proxy.predict(X_train, batch_size=bs))
    h_conf_te = np.squeeze(m_proxy.predict(X_test, batch_size=bs))
    del m_proxy
    gc.collect()
    tf.keras.backend.clear_session()

    # Instance weights
    w_c_tr = compute_instance_weights(h_conf_tr, t_low, t_high)
    w_c_te = compute_instance_weights(h_conf_te, t_low, t_high)
    w_a_tr = 1.0 - w_c_tr

    # Threshold distribution
    t_dist = get_threshold_distribution(t_low, t_high)

    # Team model selector
    cgpr_team = cgpr_team_online if team_model == 'online' else cgpr_team_static

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'simulation.csv')

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed_idx+1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")

        set_all_seeds(seed)

        # --- Simulate human ---
        h_preds_tr = simulate_human(Y_tr_ohe, ds_tr_info, conf_matrix)
        h_preds_te = simulate_human(Y_te_ohe, ds_te_info, conf_matrix)
        ind_human = acc_ohe(Y_te_ohe, h_preds_te)
        print(f"  Human accuracy: {ind_human*100:.2f}%")

        # --- Train all models ---
        # Baseline
        tf.keras.backend.clear_session()
        m_r = train_baseline(X_train, Y_tr_ohe, X_test, Y_te_ohe, config, seed,
                             skip_existing, model_dir)
        p_r_tr, c_r_tr = get_predictions_and_confidence(m_r, X_train, bs)
        p_r_te, c_r_te = get_predictions_and_confidence(m_r, X_test, bs)
        del m_r; gc.collect()

        # Complementary
        tf.keras.backend.clear_session()
        m_c = train_complementary(X_train, Y_tr_ohe, X_test, Y_te_ohe, w_c_tr,
                                  config, seed, skip_existing, model_dir)
        p_c_tr, c_c_tr = get_predictions_and_confidence(m_c, X_train, bs)
        p_c_te, c_c_te = get_predictions_and_confidence(m_c, X_test, bs)
        del m_c; gc.collect()

        # Aligned
        tf.keras.backend.clear_session()
        m_a = train_aligned(X_train, h_preds_tr, X_test, Y_te_ohe, w_a_tr,
                            config, seed, skip_existing, model_dir)
        p_a_tr, c_a_tr = get_predictions_and_confidence(m_a, X_train, bs)
        p_a_te, c_a_te = get_predictions_and_confidence(m_a, X_test, bs)
        del m_a; gc.collect()

        # Behavior-Aware
        eps = 1e-7
        mask_c_tr = w_c_tr
        mask_a_tr = w_a_tr
        mask_c_te = w_c_te
        mask_a_te = 1.0 - w_c_te

        L_c_h = np.sum((Y_tr_ohe.argmax(1) != h_preds_tr.argmax(1)) * mask_c_tr) \
                / (np.sum(mask_c_tr) + eps)

        Y_pack_tr = np.concatenate([Y_tr_ohe, h_preds_tr,
                                    mask_a_tr[:, None], mask_c_tr[:, None]], axis=1)
        Y_pack_te = np.concatenate([Y_te_ohe, h_preds_te,
                                    mask_a_te[:, None], mask_c_te[:, None]], axis=1)

        tf.keras.backend.clear_session()
        m_s = train_behavior_aware(X_train, Y_pack_tr, X_test, Y_pack_te, L_c_h,
                                   config, seed, skip_existing, model_dir)
        p_s_tr, c_s_tr = get_predictions_and_confidence(m_s, X_train, bs)
        p_s_te, c_s_te = get_predictions_and_confidence(m_s, X_test, bs)
        del m_s; gc.collect()

        # --- Adaptive AI routing ---
        tau_vec_tr = t_dist.rvs(len(h_conf_tr))
        tau_vec_te = t_dist.rvs(len(h_conf_te))

        # Oracle
        p_o_tr, c_o_tr, align_tr_o = oracle_route(
            p_a_tr, c_a_tr, p_c_tr, c_c_tr, h_conf_tr, tau_vec_tr)
        p_o_te, c_o_te, align_te_o = oracle_route(
            p_a_te, c_a_te, p_c_te, c_c_te, h_conf_te, tau_vec_te)

        # RRS
        p_rrs_tr, c_rrs_tr, _ = rrs_route(p_a_tr, c_a_tr, p_c_tr, c_c_tr)
        p_rrs_te, c_rrs_te, _ = rrs_route(p_a_te, c_a_te, p_c_te, c_c_te)

        # --- Individual accuracy ---
        ind_base = acc_ohe(Y_te_ohe, p_r_te)
        ind_aligned = acc_ohe(Y_te_ohe, p_a_te)
        ind_comp = acc_ohe(Y_te_ohe, p_c_te)
        ind_behav = acc_ohe(Y_te_ohe, p_s_te)
        ind_oracle = acc_ohe(Y_te_ohe, p_o_te)
        ind_rrs = acc_ohe(Y_te_ohe, p_rrs_te)

        print(f"\n  Individual Accuracy:")
        print(f"    Human:           {ind_human*100:.2f}%")
        print(f"    Baseline AI:     {ind_base*100:.2f}%")
        print(f"    Aligned AI:      {ind_aligned*100:.2f}%")
        print(f"    Complementary:   {ind_comp*100:.2f}%")
        print(f"    Behavior-Aware:  {ind_behav*100:.2f}%")
        print(f"    Adaptive Oracle: {ind_oracle*100:.2f}%")
        print(f"    Adaptive RRS:    {ind_rrs*100:.2f}%")

        # --- CGPR team evaluation ---
        align_mask = h_conf_te > tau_vec_te

        models_preds = {
            'baseline': p_r_te, 'aligned': p_a_te, 'complementary': p_c_te,
            'behavior_aware': p_s_te, 'adaptive_oracle': p_o_te,
            'adaptive_rrs': p_rrs_te,
        }
        team_results = {}
        for name, ai_pred in models_preds.items():
            team_ohe, _ = cgpr_team(h_preds_te, h_conf_te, ai_pred, align_mask)
            team_results[name] = acc_ohe(Y_te_ohe, team_ohe)

        print(f"\n  Team Accuracy (CGPR {team_model}):")
        for name, acc in team_results.items():
            print(f"    {name:20s}: {acc*100:.2f}%")

        # --- Log results ---
        row = {
            'seed': seed,
            'ind_human': ind_human,
            'ind_baseline_ai': ind_base,
            'ind_aligned_ai': ind_aligned,
            'ind_complementary_ai': ind_comp,
            'ind_behavior_aware_ai': ind_behav,
            'ind_adaptive_oracle': ind_oracle,
            'ind_adaptive_rrs': ind_rrs,
            'team_baseline_ai': team_results['baseline'],
            'team_aligned_ai': team_results['aligned'],
            'team_complementary_ai': team_results['complementary'],
            'team_behavior_aware_ai': team_results['behavior_aware'],
            'team_adaptive_oracle': team_results['adaptive_oracle'],
            'team_adaptive_rrs': team_results['adaptive_rrs'],
        }
        log_seed_results(row, csv_path)

        # --- Save predictions/confidences ---
        npz_path = os.path.join(results_dir, f'pred_and_conf_s{seed}.npz')
        np.savez_compressed(
            npz_path,
            y_p_tr_h=h_preds_tr, y_p_te_h=h_preds_te,
            y_c_tr_h=h_conf_tr, y_c_te_h=h_conf_te,
            y_p_tr_r=p_r_tr, y_p_te_r=p_r_te,
            y_c_tr_r=c_r_tr, y_c_te_r=c_r_te,
            y_p_tr_a=p_a_tr, y_p_te_a=p_a_te,
            y_c_tr_a=c_a_tr, y_c_te_a=c_a_te,
            y_p_tr_c=p_c_tr, y_p_te_c=p_c_te,
            y_c_tr_c=c_c_tr, y_c_te_c=c_c_te,
            y_p_tr_s=p_s_tr, y_p_te_s=p_s_te,
            y_c_tr_s=c_s_tr, y_c_te_s=c_s_te,
            y_p_tr_o=p_o_tr, y_p_te_o=p_o_te,
            y_c_tr_o=c_o_tr, y_c_te_o=c_o_te,
            y_p_tr_rrs=p_rrs_tr, y_p_te_rrs=p_rrs_te,
            y_c_tr_rrs=c_rrs_tr, y_c_te_rrs=c_rrs_te,
            seed=seed,
        )
        print(f"  Saved predictions to {os.path.basename(npz_path)}")

    print(f"\nAll results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WoofNette Adaptive AI simulation')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Seed indices to run (0-indexed), e.g. --seeds 0')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Load cached models instead of retraining')
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to write simulation.csv and pred_and_conf files to')
    args = parser.parse_args()

    run_simulation(
        seed_indices=args.seeds,
        skip_existing=args.skip_existing,
        results_dir=args.results_dir,
    )
