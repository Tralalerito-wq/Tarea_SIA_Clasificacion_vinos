# Análisis exploratorio de datos (EDA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
data = pd.read_csv("WineQT.csv")
data = data.drop("Id", axis=1)

# Limpieza de datos: Verificación de nulos y duplicados
print("Valores nulos por columna:")
print(data.isnull().sum())

print(f"\nCantidad de filas duplicadas eliminadas: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Estadísticas descriptivas del dataset
print("Estadísticas descriptivas:")
display(data.describe())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Preparación de variables predictoras (X) y objetivo (y)
X = data.drop("quality", axis=1)
y = data["quality"]

# División de datos en conjunto de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajuste del escalador y transformación del conjunto de entrenamiento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el escalador para usarlo en la fase de pruebas
joblib.dump(scaler, 'scaler.pkl')
print("Conjuntos de datos listos y escalador guardado exitosamente.")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Inicialización de modelos
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(max_depth=5, random_state=42)
}

# Entrenamiento y evaluación rápida con los datos de entrenamiento escalados
best_model_name = None
best_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_train_scaled)
    acc = accuracy_score(y_train, preds)
    print(f"{name} (Train Accuracy): {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_model = model

print(f"\nEl mejor modelo en el set de entrenamiento es: {best_model_name} con {best_accuracy:.4f} de Accuracy.")

# Guardado del mejor modelo
joblib.dump(best_model, 'mejor_modelo.pkl')
# También guardamos el set de prueba final para usarlo en el script de pruebas de forma aislada
joblib.dump((X_test_scaled, y_test), 'datos_prueba.pkl')

print(f"Modelo '{best_model_name}' y datos de prueba exportados exitosamente en archivos .pkl")
