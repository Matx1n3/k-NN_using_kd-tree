from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
import time

from classifiers import KNearestNeighbours
from metrics import accuracy_score

digits = load_digits()

# Obtener las características (X) y las etiquetas (y)
X = digits.data
y = digits.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Medir el tiempo de ejecución del modelo personalizado
start_custom = time.time()
model_custom = KNearestNeighbours()
print("Training Custom KNN...")
model_custom.fit(train_x, train_y)
print("Testing Custom KNN...")
prediction_custom = model_custom.predict(test_x, 300)
finish_custom = time.time()
print("Done with Custom KNN!")
print("Time (Custom KNN):", finish_custom - start_custom)
print("Accuracy score (Custom KNN):", accuracy_score(prediction_custom, test_y))

# Medir el tiempo de ejecución del modelo de Scikit-Learn
start_sklearn = time.time()
model_sklearn = KNeighborsClassifier(300)
print("Training Scikit-Learn KNN...")
model_sklearn.fit(train_x, train_y)
print("Testing Scikit-Learn KNN...")
prediction_sklearn = model_sklearn.predict(test_x)
finish_sklearn = time.time()
print("Done with Scikit-Learn KNN!")
print("Time (Scikit-Learn KNN):", finish_sklearn - start_sklearn)
print("Accuracy score (Scikit-Learn KNN):", accuracy_score(prediction_sklearn, test_y))
