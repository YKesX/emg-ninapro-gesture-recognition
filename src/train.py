import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# DİKKAT: models.py dosyasından senin fonksiyonları çağırıyoruz
from models import get_1d_cnn_model, get_cnn_bilstm_model


class TrainingConfig:
    """Eğitim ayarları (GERÇEK NINAPRO VERİSİNE GÖRE)"""
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    INPUT_SHAPE = (20, 10)  
    NUM_CLASSES = 13        

def load_real_data():
    """Veriyi yükler ve etiketleri hazırlar"""
    print("📂 Gerçek veri diskten okunuyor...")
    
    # Dosya yolları (Eğer hata verirse başına ../ ekle)
    x_path = 'data/processed/X_train.npy'
    y_path = 'data/processed/y_train.npy'
    
    if not os.path.exists(x_path):
        # GitHub yapısında bazen bir üst klasörde olabilir
        x_path = '../data/processed/X_train.npy'
        y_path = '../data/processed/y_train.npy'
        if not os.path.exists(x_path):
             print(f"❌ HATA: '{x_path}' bulunamadı!"); return None, None, None

    X = np.load(x_path)
    y = np.load(y_path) 
    
    y_encoded = tf.keras.utils.to_categorical(y, TrainingConfig.NUM_CLASSES)
    
    print(f"✅ Veri Yüklendi: {X.shape}")
    return X, y_encoded, y

def train_model(model_name="1d_cnn"):
    print(f"\n{'='*60}\n🚀 Model Hazırlanıyor: {model_name}\n{'='*60}")
    
    X_data, y_data_encoded, y_integers = load_real_data()
    if X_data is None: return None, None

    # Sınıf Ağırlıkları
    print("⚖️ Sınıf Ağırlıkları Hesaplanıyor...")
    unique_classes = np.unique(y_integers)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_integers
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Modeli Oluştur (models.py'dan çekiyoruz)
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

    # EĞİTİMİ BAŞLAT
    print(f"\n🔥 Eğitim Başlıyor... (Epochs: {TrainingConfig.EPOCHS})")
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
    # Klasör yoksa oluştur
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures', exist_ok=True) # Hata vermesin diye exist_ok ekledim

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

# Bu dosya direkt çalıştırılırsa eğitimi başlat
if __name__ == "__main__":
    train_model("1d_cnn")