# Análisis exploratorio de datos (EDA)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data= pd.read_csv("WineQT.csv")
data= data.drop("Id", axis=1)

X= data.drop("quality", axis=1)
y= data["quality"]

#EDA
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
