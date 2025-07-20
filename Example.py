
# import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load and prepare data (only 2 classes)
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 2]
y = y[y != 2]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes")
plt.grid(False)
plt.show()
