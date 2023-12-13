from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np

from classifiers import KNearestNeighbours
from metrics import accuracy_score
from parameter_tuning import choose_k

digits = load_digits()

# Obtener las características (X) y las etiquetas (y)
X = digits.data
y = digits.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model = KNearestNeighbours()

print("Training KNN...")
model.fit(train_x, train_y)

print("Choosing k...")
k_array = np.array([1, 2, 3, 4, 5, 6, 7])
k_array = np.array([[x] for x in k_array])
best_k = choose_k(knn_model=model, validate_x=val_x, validate_y=val_y, k_array=k_array, args=('distance weight', 'manhattan'))
print("Best k:", best_k)

print("Testing KNN...")
prediction = model.predict(test_x, distance_type="manhattan")

print("Accuracy score:", accuracy_score(prediction, test_y))
