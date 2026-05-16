import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D

# 1. Model Configuration
# 10 frames per sequence (since app runs AI every 5 frames = ~1.6 real seconds)
SEQUENCE_LENGTH = 10
# 33 MediaPipe landmarks * 4 values (x, y, z, visibility) = 132 features
FEATURES = 132 
# 3 Classes: Normal (0), Brawl (1), Fall (2)
NUM_CLASSES = 3

def build_model():
    model = Sequential([
        # 1. Spatial Feature Extraction (Understands how joints relate to each other in a single frame)
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(SEQUENCE_LENGTH, FEATURES)),
        
        # 2. Temporal Sequence Analysis (Reads motion forwards AND backwards in time)
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=False, activation='relu')),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("Building LSTM Action Recognition Model...")
    model = build_model()
    model.summary()
    
    print("\nLoading dataset...")
    data_x_path = 'X_data.npy'
    data_y_path = 'y_data.npy'
    
    if os.path.exists(data_x_path) and os.path.exists(data_y_path):
        X_train = np.load(data_x_path)
        y_train = np.load(data_y_path)
        print(f" -> Found real data! Loaded {len(X_train)} samples.")
    else:
        print(" -> Real data not found. Falling back to dummy data...")
        X_train = np.random.rand(100, SEQUENCE_LENGTH, FEATURES)
        y_train = np.random.randint(0, NUM_CLASSES, 100)
    
    print("Training Advanced Model...")
    
    # Callbacks to prevent overfitting and save the absolute best version
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('brawl_lstm_model.h5', monitor='loss', save_best_only=True)
    ]
    
    model.fit(X_train, y_train, epochs=150, batch_size=16, callbacks=callbacks)
    
    print("\nBest model automatically saved to: brawl_lstm_model.h5")
    print("Restart your Flask app to load the new AI model!")