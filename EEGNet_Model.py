import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D, Input

# STEP 1: File Read
file_name = "0_0.csv"
df = pd.read_csv(file_name, sep=",")   # ✅ FIXED

print("Columns in file:", df.columns)

channels = ["TP9", "AF7", "AF8", "TP10"]   # ✅ ab ye exist karenge
X = df[channels].values  # shape (samples, channels)

# Reshape for EEGNet: (trials, channels, samples, 1)
X = X.T[np.newaxis, :, :, np.newaxis]  # dummy ek trial
y = np.array([0])  # dummy label

# STEP 2: EEGNet
model = Sequential([
    Input(shape=(len(channels), X.shape[2], 1)),
    Conv2D(8, (1, 32), padding='same', use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D((len(channels), 1), use_bias=False, depth_multiplier=2),
    BatchNormalization(),
    AveragePooling2D((1, 4)),
    Dropout(0.5),
    SeparableConv2D(16, (1, 16), padding='same', use_bias=False),
    BatchNormalization(),
    AveragePooling2D((1, 8)),
    Dropout(0.5),
    Flatten(),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=10, verbose=1)
