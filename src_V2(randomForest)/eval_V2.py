import numpy as np
import tensorflow as tf
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# models.py import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

NUM_CLASSES = 13

def evaluate_all():
    print("\n📊 DEĞERLENDİRME BAŞLIYOR...")
    
    # 1. Test Verisini Yükle (Train.py'nin kaydettiği)
    try:
        X_test = np.load('data/processed/X_test_final.npy')
        y_test = np.load('data/processed/y_test_final.npy')
    except:
        print("❌ Test verisi bulunamadı. Önce src/train.py çalıştırın.")
        return

    results = {}
    
    # --- DEEP LEARNING MODELLERİ ---
    dl_models = ["1D-CNN", "CNN-BiLSTM"]
    for name in dl_models:
        path = f'models/best_model_{name}.keras'
        if os.path.exists(path):
            print(f"🔍 Değerlendiriliyor: {name}")
            model = tf.keras.models.load_model(path)
            
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            results[name] = {'acc': acc, 'f1': f1}
            
            # Confusion Matrix Kaydet
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            os.makedirs('reports/figures', exist_ok=True)
            plt.savefig(f'reports/figures/cm_{name}.png')
            plt.close()

    # --- RANDOM FOREST ---
    if os.path.exists('models/best_model_rf.pkl'):
        print(f"🔍 Değerlendiriliyor: Random Forest")
        rf = joblib.load('models/best_model_rf.pkl')
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = rf.predict(X_test_flat)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results['Random Forest'] = {'acc': acc, 'f1': f1}

    # --- FİNAL TABLOSU ---
    print(f"\n{'='*45}\n🏆 PROJE SONUÇ RAPORU\n{'='*45}")
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Macro F1':<10}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<20} | {res['acc']*100:6.2f}%    | {res['f1']:.4f}")

if __name__ == "__main__":
    evaluate_all()