import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()



# 6.5.1, 1)

# KNN model (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_n, y_train)

# predikcije
y_train_knn = knn.predict(X_train_n)
y_test_knn = knn.predict(X_test_n)

print("\nKNN (K=5):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn)))

# granica odluke
plot_decision_regions(X_train_n, y_train, classifier=knn)
plt.title("KNN (K=5)")
plt.show()


# 6.5.1, 2)
# K = 1
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train_n, y_train)

# predikcija i točnost
y_train_pred1 = knn1.predict(X_train_n)
y_test_pred1 = knn1.predict(X_test_n)
train_acc1 = accuracy_score(y_train, y_train_pred1)
test_acc1 = accuracy_score(y_test, y_test_pred1)

plot_decision_regions(X_train_n, y_train, classifier=knn1)
plt.title(f"KNN (K=1) - Train acc: {train_acc1:.3f}, Test acc: {test_acc1:.3f}")
plt.show()


# K = 100
knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train_n, y_train)

# predikcija i točnost
y_train_pred100 = knn100.predict(X_train_n)
y_test_pred100 = knn100.predict(X_test_n)
train_acc100 = accuracy_score(y_train, y_train_pred100)
test_acc100 = accuracy_score(y_test, y_test_pred100)

plot_decision_regions(X_train_n, y_train, classifier=knn100)
plt.title(f"KNN (K=100) - Train acc: {train_acc100:.3f}, Test acc: {test_acc100:.3f}")
plt.show()


# Zadatak 6.5.2
# definiraj pipeline (skaliranje + KNN)
pipe_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# raspon K vrijednosti
param_grid = {
    'knn__n_neighbors': list(range(1, 51))
}

# GridSearchCV (5-fold cross-validation)
grid = GridSearchCV(pipe_knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# najbolji K
print("Najbolji K:", grid.best_params_['knn__n_neighbors'])
print("Najbolja CV točnost:", grid.best_score_)

# evaluacija na test skupu
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

print("Tocnost na test skupu:", accuracy_score(y_test, y_test_pred))


# Zadatak 6.5.3
# SVM s RBF kernelom
svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_n, y_train)

# predikcije
y_train_svm = svm_rbf.predict(X_train_n)
y_test_svm = svm_rbf.predict(X_test_n)

print("\nSVM (RBF):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_svm)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_svm)))

# granica odluke
plot_decision_regions(X_train_n, y_train, classifier=svm_rbf)
plt.title("SVM (RBF)")
plt.show()

# Promjena C

for C in [0.1, 1, 10, 100]:
    model = svm.SVC(kernel='rbf', C=C, gamma='scale')
    model.fit(X_train_n, y_train)

    print(f"C={C}, test acc={accuracy_score(y_test, model.predict(X_test_n)):.3f}")

# Promjena gamme

for gamma in [0.01, 0.1, 1, 10]:
    model = svm.SVC(kernel='rbf', C=1, gamma=gamma)
    model.fit(X_train_n, y_train)

    print(f"gamma={gamma}, test acc={accuracy_score(y_test, model.predict(X_test_n)):.3f}")

kernels = ['linear', 'poly']

# Promjena kernela

for k in kernels:
    model = svm.SVC(kernel=k)
    model.fit(X_train_n, y_train)

    print(f"{k} kernel test acc: {accuracy_score(y_test, model.predict(X_test_n)):.3f}")

    plot_decision_regions(X_train_n, y_train, classifier=model)
    plt.title(f"SVM ({k})")
    plt.show()
    

# Zadatak 6.5.4
# pipeline (skaliranje + SVM)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel='rbf'))
])

# mreža hiperparametara
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.01, 0.1, 1, 10]
}

# GridSearchCV (5-fold)
grid = GridSearchCV(pipe_svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# najbolji parametri
print("Najbolji parametri:", grid.best_params_)
print("Najbolja CV točnost:", grid.best_score_)

# evaluacija na test skupu
best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

print("Tocnost na test skupu:", accuracy_score(y_test, y_test_pred))