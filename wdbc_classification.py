
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = load_breast_cancer()

print("Data shape:", data.data.shape)
print("Target shape:", data.target.shape)
print("Target names:", data.target_names)
print("First 5 feature names:", data.feature_names[:5])

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))

    if accuracy_score(y_test, pred) > best_accuracy:
        best_accuracy = accuracy_score(y_test, pred)
        best_name = name
        best_pred = pred

ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred)).plot()
plt.savefig("wdbc_classification_matrix.png")

plt.figure()
plt.scatter(data.data[:,0], data.data[:,1], c=data.target)
plt.savefig("wdbc_classification_scatter.png")
