# Entrenamiento del modelo
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


lr= LogisticRegression()
knn= KNeighborsClassifier()
svm= SVC()
dt= DecisionTreeClassifier()
rf= RandomForestClassifier(max_depth= 5)

dt.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

y_pred= svm.predict(X_test_scaled)
lr_pred= lr.predict(X_test_scaled)
knn_pred= knn.predict(X_test_scaled)
dt_pred= dt.predict(X_test_scaled)
rf_pred= rf.predict(X_test_scaled)
