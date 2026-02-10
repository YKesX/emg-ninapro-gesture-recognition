import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from sklearn.utils import class_weight

from models import get_1d_cnn_model, get_cnn_bilstm_model, get_random_forest_model


class TrainingConfig:
    
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    INPUT_SHAPE = (20, 10)  
    NUM_CLASSES = 13        

def load_real_data():
    """Loads training data from disk. Returns (X, y_encoded, y_integers) or None on failure."""

    print("Loading data from disk...")

    x_path = 'data/processed/X_train.npy'
    y_path = 'data/processed/y_train.npy'

    if not os.path.exists(x_path):
        x_path = '../data/processed/X_train.npy'
        y_path = '../data/processed/y_train.npy'
        if not os.path.exists(x_path):
            print(f"ERROR: '{x_path}' not found!")
            return None

    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except Exception as e:
        print(f"ERROR: Failed to load data files: {e}")
        return None

    if X.shape[0] != y.shape[0]:
        print(f"ERROR: Sample count mismatch — X: {X.shape[0]}, y: {y.shape[0]}")
        return None

    y_encoded = tf.keras.utils.to_categorical(y, TrainingConfig.NUM_CLASSES)

    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    return X, y_encoded, y

def train_model(model_name="1d_cnn"):
    print(f"\n{'='*60}\nPreparing Model: {model_name}\n{'='*60}")
    
    # Fix #2: Safe unpacking — catch single return value first
    result = load_real_data()
    if result is None or not isinstance(result, tuple) or len(result) != 3:
        print("ERROR: Data loading failed. Aborting training.")
        return None, None

    X_data, y_data_encoded, y_integers = result

    # ── Random Forest (Scikit-Learn) ──────────────────────────────
    # Fix #3: No class_weight calculation here — RF handles it
    # internally via class_weight='balanced' in models.py
    if model_name == "random_forest":
        # Fix #1: Explicit keyword arguments
        model = get_random_forest_model(num_classes=TrainingConfig.NUM_CLASSES)

        # Flatten 3D -> 2D: (N, 20, 10) -> (N, 200)
        n_samples = X_data.shape[0]
        X_flat = X_data.reshape(n_samples, -1)
        print(f"Data reshaped for RF: {X_data.shape} -> {X_flat.shape}")

        print("\nStarting Random Forest Training...")
        model.fit(X_flat, y_integers, sample_weight=None)

        # Save with joblib (.pkl)
        save_path = f"best_model_{model_name}.pkl"
        joblib.dump(model, save_path)
        print(f"Random Forest saved to: {save_path}")

        return model, None  # No training history for RF

    # ── Deep Learning (Keras) ─────────────────────────────────────
    # Fix #1: Explicit keyword arguments for all cross-module calls
    if model_name == "1d_cnn":
        model = get_1d_cnn_model(
            input_shape=TrainingConfig.INPUT_SHAPE,
            num_classes=TrainingConfig.NUM_CLASSES
        )
    elif model_name == "cnn_bilstm":
        model = get_cnn_bilstm_model(
            input_shape=TrainingConfig.INPUT_SHAPE,
            num_classes=TrainingConfig.NUM_CLASSES
        )
    else:
        print(f"ERROR: Unknown model name '{model_name}'.")
        return None, None

    # Fix #3: Class weights only needed for Keras models
    print("Calculating Class Weights...")
    unique_classes = np.unique(y_integers)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_integers
    )
    # Fix #4: zip instead of enumerate — safe for non-contiguous labels
    class_weights_dict = dict(zip(unique_classes.astype(int), weights))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=TrainingConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'best_model_{model_name}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\nStarting Training... (Epochs: {TrainingConfig.EPOCHS})")
    history = model.fit(
        X_data, y_data_encoded,
        batch_size=TrainingConfig.BATCH_SIZE,
        epochs=TrainingConfig.EPOCHS,
        validation_split=TrainingConfig.VALIDATION_SPLIT,
        callbacks=callbacks,
        class_weight=class_weights_dict, 
        verbose=1
    )
    return model, history

def plot_training_history(history, model_name="1d_cnn"):
    if history is None:
        print(f"No training history to plot for '{model_name}' (non-DL model). Skipping.")
        return

    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title(f'{model_name} - Accuracy'); axes[0].legend()
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title(f'{model_name} - Loss'); axes[1].legend()
    plt.tight_layout(); 
    plt.savefig(f'reports/figures/training_history_{model_name}.png')
    plt.show()


if __name__ == "__main__":

    train_model("1d_cnn")
