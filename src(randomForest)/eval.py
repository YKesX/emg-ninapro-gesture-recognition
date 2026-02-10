import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import BaseEnsemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

class EvalConfig:
    MODEL_NAME = "1d_cnn"
    INPUT_SHAPE = (20, 10)
    NUM_CLASSES = 13

def load_test_data():
    """
    Loads test data from disk.

    Returns (X_test, y_test, y_test_encoded) or None on failure.

    TODO: This function currently loads X_train.npy and performs a
          train_test_split to carve out a test set. This is NOT ideal
          for production — the 'test' samples were part of the training
          pool, which risks data leakage if the same split is not
          guaranteed in train.py. Replace with a dedicated
          X_test.npy / y_test.npy generated during preprocessing.
    """
    print("Preparing test data...")

    x_path = 'data/processed/X_train.npy'
    y_path = 'data/processed/y_train.npy'

    if not os.path.exists(x_path):
        x_path = '../data/processed/X_train.npy'
        y_path = '../data/processed/y_train.npy'
        if not os.path.exists(x_path):
            print(f"ERROR: Data file not found at '{x_path}'.")
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

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, EvalConfig.NUM_CLASSES)

    print(f"Test Data Ready: X_test={X_test.shape}, y_test={y_test.shape}")
    return X_test, y_test, y_test_encoded

def load_model(model_path):
    """
    Loads a model from disk.
    Supports both Keras (.keras) and Scikit-Learn (.pkl) formats.
    Returns the model object or None on failure.
    """
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'.")
        return None

    try:
        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            print(f"Scikit-Learn model loaded from: {model_path}")
        else:
            model = tf.keras.models.load_model(model_path)
            print(f"Keras model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model '{model_path}': {e}")
        return None


def _is_sklearn_model(model):
    """Runtime type check — safer than relying on model_name strings."""
    return isinstance(model, BaseEnsemble)

def evaluate_model(model, X_test, y_test, y_test_encoded, model_name="1d_cnn"):
    """
    Evaluates a trained model on the test set.
    Handles both Keras (probabilities) and Scikit-Learn (class labels) transparently.
    """
    print(f"\n{'='*60}\nEVALUATION: {model_name}\n{'='*60}")

    # ── Predict ───────────────────────────────────────────────────
    if _is_sklearn_model(model):
        # Scikit-Learn expects 2D input: (N, features)
        X_eval = X_test.reshape(X_test.shape[0], -1)
        print(f"  Data reshaped for sklearn: {X_test.shape} -> {X_eval.shape}")
        y_pred = model.predict(X_eval)  # Returns class labels directly
    else:
        # Keras returns class probabilities
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

    # ── Metrics ───────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test': y_test,
    }

def plot_confusion_matrix(cm, model_name="1d_cnn"):
    """Plots and saves the confusion matrix as a heatmap."""
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures', exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'reports/figures/confusion_matrix_{model_name}.png')
    plt.show()
    print(f"  Saved to reports/figures/confusion_matrix_{model_name}.png")

def evaluate_single_model(model_name="1d_cnn"):
    print(f"\nEVALUATING: {model_name}")

    # Determine correct file extension
    if model_name == "random_forest":
        model_path = f"best_model_{model_name}.pkl"
    else:
        model_path = f"best_model_{model_name}.keras"

    model = load_model(model_path)
    if model is None:
        return None

    # Safe unpacking — same pattern as train.py
    result = load_test_data()
    if result is None or not isinstance(result, tuple) or len(result) != 3:
        print("ERROR: Test data loading failed. Aborting evaluation.")
        return None

    X_test, y_test, y_test_encoded = result

    results = evaluate_model(model, X_test, y_test, y_test_encoded, model_name=model_name)
    plot_confusion_matrix(results['confusion_matrix'], model_name=model_name)
    return results

if __name__ == "__main__":

    evaluate_single_model("1d_cnn")

