# Entrenamiento del modelo
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Carga de modelo guardado y datos de prueba
loaded_model = joblib.load('mejor_modelo.pkl')
X_test_scaled, y_test = joblib.load('datos_prueba.pkl')

print(f"Modelo cargado exitosamente.")

# 2. Evaluación del modelo en los datos de prueba
y_pred = loaded_model.predict(X_test_scaled)
