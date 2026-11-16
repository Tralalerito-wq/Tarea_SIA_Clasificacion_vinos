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


def main():
    if not DATA_PATH.exists():
        print(f"No se encontró el archivo: {DATA_PATH}", file=sys.stderr)
        print("Coloca WineQT.csv en la carpeta raíz del proyecto.", file=sys.stderr)
        return 1

    data = pd.read_csv(DATA_PATH)

    if "quality" not in data.columns:
        print("El CSV debe contener una columna llamada 'quality'.", file=sys.stderr)
        return 1

    data = data.drop(columns=["Id"], errors="ignore")

    print("Valores nulos por columna:")
    print(data.isnull().sum().to_string())

    print(f"\nFilas duplicadas eliminadas: {data.duplicated().sum()}")
    data = data.drop_duplicates().dropna().reset_index(drop=True)

    print("\nEstadísticas descriptivas:")
    print(data.describe().to_string())

    X = data.drop(columns=["quality"])
    y = data["quality"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as error:
        print(f"No se pudo dividir el dataset: {error}", file=sys.stderr)
        return 1

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "feature_names": list(X.columns),
            "data": data,
        },
        OUTPUT_DIR / "datos_preparados.pkl",
    )

    print(f"\nDatos preparados en: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
