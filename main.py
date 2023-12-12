from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from classifiers import KNearestNeighbours
from metrics import accuracy_score

digits = load_digits()

# Obtener las características (X) y las etiquetas (y)
X = digits.data
y = digits.target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNearestNeighbours()
print("Training KNN...")
model.fit(train_x, train_y)
print("Testing KNN...")
prediction = model.predict(test_x)
print("Accuracy score:", accuracy_score(prediction, test_y))