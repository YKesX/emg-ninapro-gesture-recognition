from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier

# Yapılandırma
INPUT_SHAPE = (20, 10)
NUM_CLASSES = 13

def get_1d_cnn_model():
    """
    Overfitting'i önlemek için optimize edilmiş 1D-CNN.
    Özellikler: L2 Regularization, BatchNormalization, Dropout.
    """
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=INPUT_SHAPE, 
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_cnn_bilstm_model():
    """
    CNN ve Bi-LSTM Hibrit Modeli.
    Zamana bağlı örüntüleri yakalar.
    """
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=INPUT_SHAPE,
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling1D(2),
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3)),
        
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_rf_model():
    """Random Forest Classifier"""
    return RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1, 
        max_depth=15
    )