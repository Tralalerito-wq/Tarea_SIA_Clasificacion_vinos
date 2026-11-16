# Análisis exploratorio de datos (EDA)
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "WineQT.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PREPARED_DATA_PATH = OUTPUT_DIR / "datos_preparados.pkl"


def main():
    if not DATA_PATH.is_file():
        print(
            f"No se encontró el archivo de datos: {DATA_PATH}\n"
            "Coloca WineQT.csv en la carpeta raíz del proyecto (al mismo nivel que README.md) y vuelve a ejecutar.",
            file=sys.stderr,
        )
        return 1

    data = pd.read_csv(DATA_PATH)
    if "quality" not in data.columns:
        print("El CSV debe contener una columna 'quality'.", file=sys.stderr)
        return 1

    # Id es un identificador, no una característica útil para predecir la calidad.
    data = data.drop(columns=["Id"], errors="ignore")
    print("Valores nulos por columna:")
    print(data.isnull().sum().to_string())

    duplicated_count = int(data.duplicated().sum())
    print(f"\nFilas duplicadas eliminadas: {duplicated_count}")
    data = data.drop_duplicates().dropna().reset_index(drop=True)

    print("\nEstadísticas descriptivas:")
    print(data.describe().to_string())

    X = data.drop(columns=["quality"])
    y = data["quality"]
    if X.empty:
        print("No se encontraron columnas predictoras en el CSV.", file=sys.stderr)
        return 1

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as exc:
        print(f"No se pudo dividir el dataset en entrenamiento y prueba: {exc}", file=sys.stderr)
        return 1

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "feature_names": list(X.columns),
            "data": data,
            "scaler": scaler,
        },
        PREPARED_DATA_PATH,
    )
    print(f"\nDatos preparados guardados en: {PREPARED_DATA_PATH}")
    print("Siguiente paso: python src/entrenamiento.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
