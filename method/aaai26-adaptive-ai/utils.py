import os
import random
import numpy as np
import scipy.stats as ss
import yaml

# TensorFlow is imported lazily to avoid slow CUDA init for CPU-only scripts.
tf = None

def _ensure_tf():
    global tf
    if tf is None:
        import tensorflow as _tf
        tf = _tf
    return tf


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_project_dir():
    return os.path.dirname(__file__)


def resolve_optional_path(path_value, base_dir=None):
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return path_value
    root = base_dir if base_dir is not None else get_project_dir()
    return os.path.join(root, path_value)


def require_directory(path_value, description, base_dir=None, create=False):
    path = resolve_optional_path(path_value, base_dir=base_dir)
    if path is None:
        raise ValueError(
            f"Missing configuration for {description}. "
            "Please set the corresponding path in config.yaml."
        )
    if create:
        os.makedirs(path, exist_ok=True)
    elif not os.path.isdir(path):
        raise FileNotFoundError(
            f"{description} directory not found: {path}. "
            "Update config.yaml to point to a valid directory."
        )
    return path


def set_all_seeds(seed, deterministic_tf=True):
    """Set seeds for full reproducibility across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _tf = _ensure_tf()
    _tf.keras.utils.set_random_seed(seed)
    if deterministic_tf and hasattr(_tf.config, "experimental"):
        _tf.config.experimental.enable_op_determinism()


def log_seed_results(row_dict, csv_path):
    """Append one row of results to a CSV file."""
    import pandas as pd
    df_row = pd.DataFrame([row_dict])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode='a', header=write_header, index=False)


def set_max_to_one(a):
    """Convert soft predictions to hard one-hot (argmax)."""
    idx = a.argmax(axis=1)
    return (idx[:, None] == np.arange(a.shape[1])).astype(float)


def acc_ohe(Y, Y_pred):
    """Accuracy for one-hot encoded arrays."""
    n = Y.shape[0]
    ohe_dim = Y.shape[1]
    return ((Y == Y_pred).sum(axis=1) == ohe_dim).sum() / n


def build_classifier(num_classes, input_shape=(128, 128, 3), backbone='resnet50',
                     train_full_model=True, optimizer=None, return_compiled=True):
    """Build a pretrained-backbone classifier for image classification.

    Args:
        num_classes: Number of output classes.
        input_shape: Input image shape (H, W, C).
        backbone: One of 'resnet50', 'resnet152'.
        train_full_model: If False, freeze backbone layers except BatchNorm.
        optimizer: Keras optimizer (defaults to SGD(lr=1e-3)).
        return_compiled: If False, return uncompiled model.
    """
    _tf = _ensure_tf()
    backbone_cls = {
        'resnet50': _tf.keras.applications.ResNet50,
        'resnet152': _tf.keras.applications.ResNet152,
    }
    if backbone not in backbone_cls:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(backbone_cls)}")

    base_model = backbone_cls[backbone](
        weights='imagenet', include_top=False, input_shape=input_shape)

    if not train_full_model:
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    if input_shape[0] != 224:
        model.add(tf.keras.layers.UpSampling2D(
            size=(224 // input_shape[0], 224 // input_shape[0]),
            interpolation='bilinear'))
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(num_classes * 4, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    if not return_compiled:
        return model

    if optimizer is None:
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def simulate_human(Y_ohe, group_info, confusion_matrix):
    """Simulate human predictions using a confusion matrix.

    Args:
        Y_ohe: One-hot encoded true labels (n, num_classes).
        group_info: Array mapping each instance to a class index (unused when
                    confusion_matrix is provided, but kept for API consistency).
        confusion_matrix: (num_classes, num_classes) normalized probability matrix.
            Row i gives the distribution over predicted classes when the true class is i.

    Returns:
        y_pred: One-hot predictions (n, num_classes).
    """
    n = Y_ohe.shape[0]
    num_classes = Y_ohe.shape[1]
    ohe_ixs = np.arange(num_classes)
    y_pred = np.zeros_like(Y_ohe)

    for i in range(n):
        true_ix = Y_ohe[i].argmax()
        chosen = np.random.choice(ohe_ixs, p=confusion_matrix[true_ix])
        y_pred[i, chosen] = 1

    return y_pred


def _unpack_team_loss_targets(y_true_pack, K):
    """Unpack packed behavior-aware labels and masks."""
    y_true = y_true_pack[:, :K]
    y_human = y_true_pack[:, K:2*K]
    m_a = y_true_pack[:, 2*K]
    m_c = y_true_pack[:, 2*K + 1]
    return y_true, y_human, m_a, m_c


def _masked_soft_disagreement(target, y_pred, mask, eps):
    """Mean soft disagreement against a one-hot target under a mask."""
    _tf = _ensure_tf()
    p_target = _tf.reduce_sum(target * y_pred, axis=1)
    loss = (1.0 - p_target) * mask
    return _tf.reduce_sum(loss) / (_tf.reduce_sum(mask) + eps)


def _masked_cross_entropy(target, y_pred, mask, ce, eps):
    """Mean categorical cross-entropy under a mask."""
    _tf = _ensure_tf()
    loss = ce(target, y_pred) * mask
    return _tf.reduce_sum(loss) / (_tf.reduce_sum(mask) + eps)


def _team_loss_terms(y_true_pack, y_pred, K, ce, eps):
    """Compute the two regional terms used by the CGPR team objective."""
    y_true, y_human, m_a, m_c = _unpack_team_loss_targets(y_true_pack, K)
    L_c = _masked_soft_disagreement(y_true, y_pred, m_c, eps)
    L_a = _masked_cross_entropy(y_human, y_pred, m_a, ce, eps)
    return L_c, L_a


def team_loss_exact_cgpr(L_c_h_const, K):
    """Behavior-aware training loss used in the public release.

    The coupling term is detached for optimization stability while preserving
    the intended dependence on the gap between human and model performance in
    the complementary region.
    """
    _tf = _ensure_tf()
    eps = 1e-7
    ce = _tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def team_loss(y_true_pack, y_pred):
        L_c, L_a = _team_loss_terms(y_true_pack, y_pred, K, ce, eps)
        weight = _tf.stop_gradient(L_c_h_const - L_c)
        weight = _tf.maximum(weight, 0.0)
        return L_c + weight * L_a

    return team_loss


def true_top1_acc(y_true_pack, y_pred):
    """Custom metric: accuracy against ground-truth labels from packed tensor."""
    _tf = _ensure_tf()
    K = _tf.shape(y_pred)[-1]
    y_true = y_true_pack[:, :K]
    return _tf.keras.metrics.categorical_accuracy(y_true, y_pred)


def human_agreement(y_true_pack, y_pred):
    """Custom metric: agreement with human predictions from packed tensor."""
    _tf = _ensure_tf()
    K = _tf.shape(y_pred)[-1]
    y_human = y_true_pack[:, K:2*K]
    return _tf.keras.metrics.categorical_accuracy(y_human, y_pred)


def compute_instance_weights(confidences, threshold_low, threshold_high):
    """Compute complementarity weights: w_i = 1 - F_T(C_i).

    Threshold distribution is U(threshold_low, threshold_high).
    """
    t_dist = ss.uniform(loc=threshold_low, scale=threshold_high - threshold_low)
    return 1.0 - t_dist.cdf(confidences)


def get_threshold_distribution(threshold_low, threshold_high):
    """Get a scipy distribution for sampling thresholds."""
    return ss.uniform(loc=threshold_low, scale=threshold_high - threshold_low)


def _argmax(ohe):
    """One-hot -> index."""
    return ohe.argmax(axis=1) if ohe.ndim == 2 else ohe


def oracle_route(y_p_a, y_c_a, y_p_c, y_c_c, y_c_h, tau):
    """Route based on human confidence: aligned if conf > tau, else complementary."""
    use_aligned = y_c_h > tau
    preds = np.where(use_aligned[:, None], y_p_a, y_p_c)
    confs = np.where(use_aligned, y_c_a, y_c_c)
    return preds, confs, use_aligned


def rrs_route(y_p_a, y_c_a, y_p_c, y_c_c):
    """Rational Routing Shortcut: pick whichever specialist is more confident."""
    use_aligned = y_c_a > y_c_c
    preds = np.where(use_aligned[:, None], y_p_a, y_p_c)
    confs = np.where(use_aligned, y_c_a, y_c_c)
    return preds, confs, use_aligned


def _disagreement_rate(h_idx, ai_idx):
    return np.mean(h_idx != ai_idx) if len(h_idx) else 0.0


def cgpr_team_static(y_p_h, y_c_h, ai_pred, align_mask):
    """One-shot CGPR: r = 1 - disagreement(alignment_region).

    Returns (team_pred_ohe, team_pred_idx).
    """
    h_idx = _argmax(y_p_h)
    ai_idx = _argmax(ai_pred)

    Da = align_mask
    Dc = ~Da

    r = 1.0 - _disagreement_rate(h_idx[Da], ai_idx[Da])

    team_idx = h_idx.copy()
    rely = np.random.rand(len(team_idx)) < r
    use_ai = Dc & rely
    team_idx[use_ai] = ai_idx[use_ai]

    team_pred = np.zeros_like(y_p_h)
    team_pred[np.arange(len(team_pred)), team_idx] = 1
    return team_pred, team_idx


def cgpr_team_online(y_p_h, y_c_h, ai_pred, align_mask):
    """Streaming CGPR: Beta(1,1) prior, updated per alignment-region observation.

    Returns (team_pred_ohe, team_pred_idx).
    """
    h_idx = _argmax(y_p_h)
    ai_idx = _argmax(ai_pred)
    Da = align_mask
    Dc = ~Da

    successes = 1
    trials = 2

    team_idx = h_idx.copy()
    for i in range(len(team_idx)):
        if Da[i]:
            agree = (h_idx[i] == ai_idx[i])
            trials += 1
            successes += 1 if agree else 0
        else:
            r = successes / trials
            if np.random.rand() < r:
                team_idx[i] = ai_idx[i]

    team_pred = np.zeros_like(y_p_h)
    team_pred[np.arange(len(team_pred)), team_idx] = 1
    return team_pred, team_idx


def make_accuracy_threshold_callback(monitor='accuracy', threshold=0.60):
    """Create a Keras callback that stops training when a metric reaches a threshold."""
    _tf = _ensure_tf()

    class AccuracyThresholdCallback(_tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.monitor = monitor
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor, 0)
            if current >= self.threshold:
                print(f"\n  Reached {self.monitor}={current:.4f} >= {self.threshold}, stopping.")
                self.model.stop_training = True

    return AccuracyThresholdCallback()
