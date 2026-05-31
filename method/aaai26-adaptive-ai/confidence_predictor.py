"""Train a confidence predictor for WoofNette images.

Trains a ResNet152 regression model to predict inter-annotator agreement
from images.  The resulting checkpoint is used by woofnette.py as a routing
proxy (stand-in for human confidence) during adaptive ensemble training.

Adapted from the IJCAI 2024 Behavior-Aware AI release:
https://github.com/shasanamin/ijcai24-behavior-aware-ai

Usage:
    python confidence_predictor.py
    python confidence_predictor.py --epochs 500
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.stats import pearsonr

from utils import load_config, set_all_seeds, require_directory

PROJECT_DIR = os.path.dirname(__file__)
SEED = 42


def load_pilot_data():
    """Load pilot study data and compute per-image agreement statistics."""
    config = load_config()
    classes = config['classes']

    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'pilot_study.csv'))
    df = df[df['imageName'] != 'attention.png'].copy()

    results = []
    for img_name, grp in df.groupby('imageName'):
        true_class = img_name.split('_')[0]
        preds = grp['prediction'].values

        majority = pd.Series(preds).mode().iloc[0]
        agreement = (preds == majority).mean()

        results.append({
            'imageName': img_name,
            'true_class': true_class,
            'agreement': agreement,
        })

    return pd.DataFrame(results)


def load_images(image_names, image_dir, target_size=(128, 128)):
    """Load and preprocess images."""
    images = []
    for name in image_names:
        path = os.path.join(image_dir, name)
        img = load_img(path, target_size=target_size)
        images.append(img_to_array(img))
    return np.array(images)


def build_model(input_shape=(128, 128, 3), backbone='resnet152'):
    """Build a ResNet-based regression model for agreement prediction."""
    backbone_cls = {
        'resnet50': tf.keras.applications.ResNet50,
        'resnet152': tf.keras.applications.ResNet152,
    }
    base = backbone_cls[backbone](
        weights='imagenet', include_top=False,
        input_shape=(224, 224, input_shape[2]))

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Resizing(224, 224, interpolation='bilinear'),
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.RootMeanSquaredError(),
                 'mean_absolute_error'])

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train agreement-based confidence predictor')
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    set_all_seeds(SEED)
    config = load_config()

    backbone = config['woofnette']['confidence_predictor']['backbone']
    data_dir = require_directory(
        config['paths'].get('data'), 'data', base_dir=PROJECT_DIR)
    image_dir = config.get('paths', {}).get('woofnette_images')
    if image_dir is None:
        image_dir = os.path.join(data_dir, 'test')
    elif not os.path.isabs(image_dir):
        image_dir = os.path.join(PROJECT_DIR, image_dir)
    image_dir = require_directory(image_dir, 'woofnette_images',
                                  base_dir=PROJECT_DIR)

    print("Loading pilot study data...")
    df_img = load_pilot_data()
    print(f"  {len(df_img)} images with annotations")

    # 90/10 train/test split
    np.random.seed(SEED)
    test_idx = np.random.choice(len(df_img), int(0.1 * len(df_img)),
                                replace=False)
    train_idx = np.array([i for i in range(len(df_img)) if i not in test_idx])
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    print("Loading images...")
    all_images = load_images(df_img['imageName'].values, image_dir)
    X_tr, X_te = all_images[train_idx], all_images[test_idx]

    y_tr = df_img['agreement'].values[train_idx]
    y_te = df_img['agreement'].values[test_idx]

    # Baseline: predicting the mean
    baseline_rmse = np.sqrt(np.mean((y_te - y_tr.mean())**2))
    print(f"Baseline (mean) RMSE: {baseline_rmse:.4f}")

    # Train
    print(f"\nTraining agreement predictor ({backbone}, {args.epochs} epochs)")
    model = build_model(backbone=backbone)
    model.fit(X_tr, y_tr, epochs=args.epochs, verbose=2)

    # Evaluate
    y_pred_te = model.predict(X_te).flatten()
    rmse = np.sqrt(np.mean((y_te - y_pred_te)**2))
    r, _ = pearsonr(y_te, y_pred_te)
    print(f"\nTest RMSE: {rmse:.4f}, Pearson R: {r:.3f}")

    save_path = os.path.join(data_dir, f'm_agr_{backbone}.keras')
    model.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    main()
