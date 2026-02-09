import tensorflow as tf
from tensorflow.keras import layers, models

def get_1d_cnn_model(input_shape, num_classes):
    """
    Görev Listesi Task 3: 'Implement deep baseline 1D CNN'
    """
    model = models.Sequential([
        # --- Özellik Çıkarma (Feature Extraction) ---
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(), # Eğitimi hızlandırır ve dengeler
        layers.MaxPooling1D(pool_size=2), # Veriyi yarıya indirir, önemliyi tutar

        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(), # Tüm zaman adımlarını tek bir vektöre indirger

        # --- Sınıflandırma (Classification) ---
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Ezberlemeyi (overfitting) önlemek için %50 unutma
        layers.Dense(num_classes, activation='softmax') # Sonuç: Hangi hareket?
    ])

    # Compile: Model eğitime hazırlanıyor
    model.compile(
        optimizer='adam',           # Hızlı öğrenme algoritması
        loss='categorical_crossentropy', # Hata hesaplama yöntemi
        metrics=['accuracy']        # Başarı ölçümü
    )
    return model

def get_cnn_bilstm_model(input_shape, num_classes):
    """
    Görev Listesi Task 3: 'Intermediate deep variant: CNN + BiLSTM head'
    """
    model = models.Sequential([
        # 1. Kısım: CNN Blokları (Deseni anla)
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),

        # 2. Kısım: BiLSTM (Zamanı ve sırayı anla)
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),

        # 3. Kısım: Karar Ver
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model