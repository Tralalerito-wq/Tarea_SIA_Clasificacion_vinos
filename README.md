# Clasificación de vinos

## Requisitos

Se necesita Python 3.9 o posterior. Coloca el archivo `WineQT.csv` en la carpeta raíz del proyecto (junto a este README). El archivo de datos no está incluido en el repositorio.

## Ejecución desde Git Bash (Windows)

Desde la carpeta del repositorio:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python src/eda.py
python src/entrenamiento.py
python src/prueba.py
