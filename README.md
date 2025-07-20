# Naive Bayes Classifier – Machine Learning

This project demonstrates the implementation of the **Naive Bayes algorithm** using Scikit-Learn on a simplified version of the Iris dataset (binary classification).


## 📌 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on **Bayes’ Theorem**, with the “naive” assumption that features are **independent** of each other.

Despite its simplicity, it works surprisingly well in many real-world situations, especially in **Natural Language Processing (NLP)** and **text classification**.


## ✅ Why Use It?

| Feature | Benefit |
|--------|---------|
| ⚡ Very fast | Great for large datasets and real-time systems |
| 📬 Works well for text | Spam detection, sentiment analysis, etc. |
| 📊 Requires less data | Performs well even with small training sets |
| 🧠 Easy to understand | Based on simple probability calculations |


## 📂 Dataset Used

- [Scikit-learn's Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- Binary classification only: Class 0 vs Class 1 (we removed Class 2 for simplicity)


## 🚀 Implementation Steps

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load and prepare the data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binary classification: remove class 2
X = X[y != 2]
y = y[y != 2]

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3. Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Greens')
plt.title("Confusion Matrix - Naive Bayes")
plt.grid(False)
plt.show()

📊 Output

Accuracy: 1.0
And a confusion matrix will be displayed with correct predictions for Class 0 and Class 1.


​
 
Naive assumption:
All features are conditionally independent given the class.

GaussianNB:
Assumes the features follow a normal (Gaussian) distribution.

💡 Real-World Applications
Email Spam Filtering

Sentiment Analysis (positive/negative review)

News categorization

Medical diagnosis

Social media comment moderation

✅ Conclusion
Naive Bayes is one of the simplest yet most effective machine learning algorithms. It's extremely fast, works well with high-dimensional data, and is a perfect choice for text-based tasks.
