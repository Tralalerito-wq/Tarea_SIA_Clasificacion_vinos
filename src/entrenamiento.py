# Entrenamiento del modelo
from pathlib import Path
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_PATH = OUTPUT_DIR / "datos_preparados.pkl"


def main():
    if not DATA_PATH.exists():
        print("No se encontraron datos preparados.", file=sys.stderr)
        print("Ejecuta primero: python src/eda.py", file=sys.stderr)
        return 1

    prepared = joblib.load(DATA_PATH)
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]

    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        print("Se necesitan al menos dos ejemplos por clase.", file=sys.stderr)
        return 1

    models = {
        "Regresión logística": LogisticRegression(max_iter=2000),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Árbol de decisión": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(max_depth=5, random_state=42),
    }

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    best_name, best_score, best_model = None, -1, None

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="accuracy"
        )
        score = scores.mean()
        print(f"{name}: accuracy promedio = {score:.4f}")

        if score > best_score:
            best_name, best_score, best_model = name, score, model

    best_model.fit(X_train, y_train)
    joblib.dump(
        {
            "model": best_model,
            "name": best_name,
            "feature_names": prepared["feature_names"],
        },
        OUTPUT_DIR / "mejor_modelo.pkl",
    )

    print(f"\nMejor modelo: {best_name} (accuracy CV: {best_score:.4f})")
    print("Modelo guardado en outputs/mejor_modelo.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
