import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='Test')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Binarni klasifikacijski problem')
plt.legend()
plt.show()

#b)
model = LogisticRegression()
model.fit(X_train, y_train)

#c)
theta0 = model.intercept_[0]
theta1 = model.coef_[0][0]
theta2 = model.coef_[0][1]

print(f"θ₀ = {theta0:.4f}")
print(f"θ₁ = {theta1:.4f}")
print(f"θ₂ = {theta2:.4f}")

x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2 = -(theta0 + theta1 * x1) / theta2

# Zasebno klasa 0 i klasa 1
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
            color='blue', label='Klasa 0')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
            color='red', label='Klasa 1')

plt.plot(x1, x2, color='black', label='Granica odluke')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke')
plt.legend()
plt.show()


#d)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)

# Točnost, preciznost, odziv
print(f"Točnost:    {accuracy_score(y_test, y_pred):.4f}")
print(f"Preciznost: {precision_score(y_test, y_pred):.4f}")
print(f"Odziv:      {recall_score(y_test, y_pred):.4f}")

# Matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


# e)
correct = (y_pred == y_test)
colors = np.where(correct, 'green', 'black')

plt.scatter(X_test[:, 0], X_test[:, 1], c=colors)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Testni skup - točno/pogrešno klasificirani')
plt.show()