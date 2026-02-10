import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import class_weight
import joblib
import sys

# models.py dosyasını import etmek için
# Eğer aynı klasörde çalışıyorsan direkt import çalışır ama garanti olsun:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import get_1d_cnn_model, get_cnn_bilstm_model, get_rf_model

# AYARLAR
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 13

# Klasörleri Garantiye Al
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_data_stratified():
    """
    VERİ SIZINTISI (LEAKAGE) ÇÖZÜMÜ:
    Veriyi karıştırmadan (Shuffle: False), her sınıfın 
    ilk %80'ini Train, son %20'sini Test yapar.
    """
    print("📥 Veri Yükleniyor (Stratified Block Split)...")
    
    # Dosya yollarını ana dizine göre ayarlıyoruz
    try:
        X_all = np.load('data/processed/X_train.npy')
        y_all = np.load('data/processed/y_train.npy')
    except FileNotFoundError:
        print("❌ HATA: 'data/processed/' altında .npy dosyaları yok.")
        print("   Lütfen önce preprocess.py çalıştırın veya verileri yükleyin.")
        sys.exit()

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Her sınıfı kendi içinde zamana göre böl
    for cls in np.unique(y_all):
        indices = np.where(y_all == cls)[0]
        X_cls, y_cls = X_all[indices], y_all[indices]
        
        split_idx = int(len(X_cls) * 0.8) # %80 Train
        
        X_train_list.append(X_cls[:split_idx])
        y_train_list.append(y_cls[:split_idx])
        X_test_list.append(X_cls[split_idx:])
        y_test_list.append(y_cls[split_idx:])

    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.concatenate(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    # Sadece EĞİTİM verisini karıştır (Test verisinin sırası bozulmasın)
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]
    
    print(f"✅ Veri Hazır: Train {X_train.shape}, Test {X_test.shape}")
    
    # Test verisini Eval.py için kaydet
    np.save('data/processed/X_test_final.npy', X_test)
    np.save('data/processed/y_test_final.npy', y_test)
    
    return X_train, y_train, X_test, y_test

def train_all():
    # 1. Veriyi Yükle
    X_train, y_train, X_test, y_test = load_data_stratified()
    
    # One-Hot Encoding (DL Modelleri için)
    y_train_enc = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_enc = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    # Class Weights (Dengesizlik için)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), weights))

    # --- A. RANDOM FOREST ---
    print("\n🌲 Random Forest Eğitiliyor...")
    rf = get_rf_model()
    # 3D -> 2D Reshape
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    rf.fit(X_train_flat, y_train)
    joblib.dump(rf, 'models/best_model_rf.pkl')
    print("✅ RF Kaydedildi.")

    # --- B. DEEP LEARNING ---
    models_dict = {
        "1D-CNN": get_1d_cnn_model(),
        "CNN-BiLSTM": get_cnn_bilstm_model()
    }
    
    for name, model in models_dict.items():
        print(f"\n🚀 {name} Eğitiliyor...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(f'models/best_model_{name}.keras', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train_enc,
            validation_data=(X_test, y_test_enc),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=weight_dict,
            verbose=1
        )
        print(f"✅ {name} Kaydedildi.")

if __name__ == "__main__":
    train_all()