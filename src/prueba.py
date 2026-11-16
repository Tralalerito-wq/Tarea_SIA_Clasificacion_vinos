# Pruebas del modelo
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
accuracy= accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred, average='weighted')
recall= recall_score(y_test, y_pred, average='weighted')
f1= f1_score(y_test, y_pred, average='weighted')
conf_matrix= confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)

# Analizar la distribución de la variable objetivo 'quality'
distribucion = y.value_counts().sort_index()
print("Distribución de clases de calidad:")
print(distribucion)

# Graficar la distribución
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(x=y, palette='viridis')
plt.title('Distribución de las clases de Calidad (Quality)')
plt.xlabel('Calidad')
plt.ylabel('Cantidad de muestras')
plt.show()
