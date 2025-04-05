# SHAP.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your trained model (autoencoder or other models)
from tensorflow.keras.models import load_model
autoencoder = load_model("models/silent_sentinel_autoencoder.h5")

# Load dataset for explainability
df = pd.read_csv("cic_ids_2018_normal.csv")  # Example dataset for explainability
X = df.select_dtypes(include=[np.number])

# Normalize and preprocess the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Use SHAP for explainability
explainer = shap.KernelExplainer(autoencoder.predict, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# Visualize SHAP values (Feature Importance)
shap.summary_plot(shap_values, X_scaled)