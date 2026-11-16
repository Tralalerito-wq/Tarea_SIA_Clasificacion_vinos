# Pruebas del modelo
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("=== Métricas Finales de Evaluación en Prueba ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Gráfico de la Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Real')
plt.title('Matriz de Confusión - Conjunto de Prueba')
plt.show()


# Extraemos el Random Forest entrenado desde nuestro diccionario de modelos
rf_model = models['Random Forest']

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

display(feature_importances)

# Visualización de la distribución de las variables y la variable objetivo
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=data, hue='quality', palette='viridis', legend=False)
plt.title('Distribución de las clases de Calidad (Quality)')
plt.xlabel('Calidad')
plt.ylabel('Cantidad de muestras')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Histograma de algunas características clave
data[['alcohol', 'sulphates', 'volatile acidity', 'citric acid']].hist(bins=15, figsize=(12, 8))
plt.suptitle('Distribución de Características Clave')
plt.show()
