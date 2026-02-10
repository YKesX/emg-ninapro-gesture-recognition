import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class EvalConfig:
    MODEL_NAME = "1d_cnn"
    INPUT_SHAPE = (20, 10)
    NUM_CLASSES = 13

def load_test_data():
    print("Preparing test data...")
    
    x_path = 'data/processed/X_test_split.npy'
    y_path = 'data/processed/y_test_split.npy'
    
    if not os.path.exists(x_path):
         x_path = '../data/processed/X_test_split.npy'
         y_path = '../data/processed/y_test_split.npy'
         if not os.path.exists(x_path):
             print(f"ERROR: Test split not found at '{x_path}'!")
             print("Please run prepare_data_splits.py first to create the splits.")
             return None, None

    try:
        X_test = np.load(x_path)
        y_test = np.load(y_path)
        
        print(f"Test Data Ready: {X_test.shape}")
        return X_test, y_test
    except Exception as e:
        print(f"Failed to load data. {e}")
        return None, None

def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except:
        print("ERROR: Model file not found.")
        return None

def evaluate_model(model, X_test, y_test, model_name="1d_cnn"):
    print(f"\nEVALUATION: {model_name}")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   Accuracy: {acc*100:.2f}%")
    print(f"   Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    
    return {'accuracy': acc, 'macro_f1': macro_f1, 'confusion_matrix': cm, 'y_pred': y_pred, 'y_test': y_test}

def plot_confusion_matrix(cm, model_name="1d_cnn"):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.show()

def evaluate_single_model(model_name="1d_cnn"):
    print(f"\nEVALUATING: {model_name}")
    model = load_model(f"best_model_{model_name}.keras")
    if model is None: return None

    X_test, y_test = load_test_data()
    if X_test is None: return None

    results = evaluate_model(model, X_test, y_test, model_name)
    plot_confusion_matrix(results['confusion_matrix'], model_name)
    return results

if __name__ == "__main__":

    evaluate_single_model("1d_cnn")

