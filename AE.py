# AE.py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load preprocessed CIC-IDS-2018 dataset (normal data)
df = pd.read_csv("cic_ids_2018_normal.csv")  # Replace with your normal data file

# Drop non-numeric/logical fields (e.g., IPs, ports if present)
X = df.select_dtypes(include=[np.number])

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and validation sets
X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=42)

# Build autoencoder model
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(X_train, X_train,
                          epochs=50,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(X_val, X_val),
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Save the model
autoencoder.save("models/silent_sentinel_autoencoder.h5")