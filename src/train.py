import os
import numpy as np
try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required to run training. "
        "Please install it, for example with:\n\n"
        "    pip install tensorflow\n"
    ) from exc
try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required to plot training history. "
        "Please install it, for example with:\n\n"
        "    pip install matplotlib\n"
    ) from exc
from sklearn.utils import class_weight

from models import get_1d_cnn_model, get_cnn_bilstm_model


class TrainingConfig:
    
    BATCH_SIZE = 32
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    INPUT_SHAPE = (20, 10)  
    NUM_CLASSES = 13        

def load_real_data():
    
    print("Loading data from disk...")
    
    # Try to load pre-split data first (preferred)
    x_train_path = 'data/processed/X_train_split.npy'
    y_train_path = 'data/processed/y_train_split.npy'
    x_val_path = 'data/processed/X_val_split.npy'
    y_val_path = 'data/processed/y_val_split.npy'
    
    if not os.path.exists(x_train_path):
        x_train_path = '../data/processed/X_train_split.npy'
        y_train_path = '../data/processed/y_train_split.npy'
        x_val_path = '../data/processed/X_val_split.npy'
        y_val_path = '../data/processed/y_val_split.npy'
        if not os.path.exists(x_train_path):
            print(f"ERROR: Split data not found at '{x_train_path}'!")
            print("Please run prepare_data_splits.py first to create the splits.")
            return None, None, None, None, None, None

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(x_val_path)
    y_val = np.load(y_val_path)
    
    y_train_encoded = tf.keras.utils.to_categorical(y_train, TrainingConfig.NUM_CLASSES)
    y_val_encoded = tf.keras.utils.to_categorical(y_val, TrainingConfig.NUM_CLASSES)
    
    print(f"Train Data Loaded: {X_train.shape}")
    print(f"Validation Data Loaded: {X_val.shape}")
    return X_train, y_train_encoded, y_train, X_val, y_val_encoded, y_val

def train_model(model_name="1d_cnn"):
    print(f"\n{'='*60}\nPreparing Model: {model_name}\n{'='*60}")
    
    X_train, y_train_encoded, y_train_integers, X_val, y_val_encoded, y_val_integers = load_real_data()
    if X_train is None: return None, None

    # Class Weights
    print("Calculating Class Weights...")
    unique_classes = np.unique(y_train_integers)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train_integers
    )
    class_weights_dict = dict(enumerate(class_weights))

    if model_name == "1d_cnn":
        model = get_1d_cnn_model(TrainingConfig.INPUT_SHAPE, TrainingConfig.NUM_CLASSES)
    elif model_name == "cnn_bilstm":
        model = get_cnn_bilstm_model(TrainingConfig.INPUT_SHAPE, TrainingConfig.NUM_CLASSES)
    else: return None, None

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=TrainingConfig.EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=f'best_model_{model_name}.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\nStarting Training... (Epochs: {TrainingConfig.EPOCHS})")
    history = model.fit(
        X_train, y_train_encoded,
        batch_size=TrainingConfig.BATCH_SIZE,
        epochs=TrainingConfig.EPOCHS,
        validation_data=(X_val, y_val_encoded),
        callbacks=callbacks,
        class_weight=class_weights_dict, 
        verbose=1
    )
    return model, history

def plot_training_history(history, model_name="1d_cnn"):
    
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
