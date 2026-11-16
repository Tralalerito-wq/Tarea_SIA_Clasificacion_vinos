# Pruebas del modelo
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "datos_preparados.pkl"
MODEL_PATH = OUTPUT_DIR / "mejor_modelo.pkl"


def main():
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        print("Faltan archivos generados en los pasos anteriores.", file=sys.stderr)
        print("Ejecuta eda.py y entrenamiento.py primero.", file=sys.stderr)
        return 1

    prepared = joblib.load(DATA_PATH)
    saved_model = joblib.load(MODEL_PATH)

    y_test = prepared["y_test"]
    predictions = saved_model["model"].predict(prepared["X_test"])

    print(f"Modelo evaluado: {saved_model['name']}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Precision: {precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1 Score:  {f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}"
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    labels = sorted(pd.unique(y_test))
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matriz_confusion.png")
    plt.close()

    data = prepared["data"]
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="quality", color="steelblue")
    plt.title("Distribución de calidad")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribucion_calidad.png")
    plt.close()

    print(f"\nGráficos guardados en: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
