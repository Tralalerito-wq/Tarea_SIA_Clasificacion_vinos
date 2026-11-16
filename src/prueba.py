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
PREPARED_DATA_PATH = OUTPUT_DIR / "datos_preparados.pkl"
MODEL_PATH = OUTPUT_DIR / "mejor_modelo.pkl"


def main():
    missing = [str(path) for path in (PREPARED_DATA_PATH, MODEL_PATH) if not path.is_file()]
    if missing:
        print(
            "Faltan archivos generados por los pasos anteriores: " + ", ".join(missing) + "\n"
            "Ejecuta primero python src/eda.py y luego python src/entrenamiento.py.",
            file=sys.stderr,
        )
        return 1

    prepared = joblib.load(PREPARED_DATA_PATH)
    saved_model = joblib.load(MODEL_PATH)
    model = saved_model["model"]
    y_test = prepared["y_test"]
    predictions = model.predict(prepared["X_test_scaled"])

    print(f"Modelo evaluado: {saved_model['name']}")
    print("=== Métricas en el conjunto de prueba ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = sorted(pd.unique(y_test))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusión - conjunto de prueba")
    plt.tight_layout()
    confusion_path = OUTPUT_DIR / "matriz_confusion.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close()

    # Importancias cuando el estimador las ofrece (p. ej., árboles y Random Forest).
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = saved_model["feature_names"]
        ranking = pd.DataFrame({"variable": features, "importancia": importances})
        ranking = ranking.sort_values("importancia", ascending=False)
        print("\nImportancia de variables:")
        print(ranking.to_string(index=False))
        plt.figure(figsize=(9, 6))
        sns.barplot(data=ranking, x="importancia", y="variable", color="steelblue")
        plt.title("Importancia de variables")
        plt.tight_layout()
        importance_path = OUTPUT_DIR / "importancia_variables.png"
        plt.savefig(importance_path, dpi=150)
        plt.close()

    data = prepared["data"]
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x="quality", hue="quality", palette="viridis", legend=False)
    plt.title("Distribución de la calidad del vino")
    plt.xlabel("Calidad")
    plt.ylabel("Cantidad de muestras")
    plt.tight_layout()
    distribution_path = OUTPUT_DIR / "distribucion_calidad.png"
    plt.savefig(distribution_path, dpi=150)
    plt.close()

    histogram_path = OUTPUT_DIR / "distribucion_caracteristicas.png"
    columns = [column for column in ["alcohol", "sulphates", "volatile acidity", "citric acid"] if column in data]
    if columns:
        data[columns].hist(bins=15, figsize=(12, 8))
        plt.suptitle("Distribución de características seleccionadas")
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150)
        plt.close("all")

    print(f"\nGráficos guardados en: {OUTPUT_DIR}")
    print(f"- {confusion_path.name}")
    print(f"- {distribution_path.name}")
    if columns:
        print(f"- {histogram_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
