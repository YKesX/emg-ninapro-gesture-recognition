import tensorflow as tf
from tensorflow.keras import layers, models

def get_1d_cnn_model(input_shape, num_classes):

    model = models.Sequential([
        # --- (Feature Extraction) ---
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),

        # --- (Classification) ---
        layers.Dense(128, activation='relu'),
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

def get_cnn_bilstm_model(input_shape, num_classes):

    model = models.Sequential([
        # Feature Extraction (CNN)
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),

        # Temporal Learning (BiLSTM)
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),

        # Classification Head
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

