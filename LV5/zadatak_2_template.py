import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z.astype(float), alpha=0.3, cmap=cmap)  # IZMJENA: dodano .astype(float)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

print(df.isnull().sum())

df = df.drop(columns=['sex'])

df.dropna(axis=0, inplace=True)

df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

print(df.info())

output_variable = ['species']

input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

y_train = y_train.astype(int)  
y_test = y_test.astype(int)    

# a)
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

x = np.arange(len(unique_train))
width = 0.35

plt.bar(x - width/2, counts_train, width, label='Skup za učenje')
plt.bar(x + width/2, counts_test, width, label='Skup za testiranje')
plt.xticks(x, ['Adelie', 'Chinstrap', 'Gentoo'])
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasi')
plt.legend()
plt.show()

# b)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.ravel())

# c)
print(f"Intercept: {model.intercept_}")
print(f"Koeficijenti: {model.coef_}")

# d)
plot_decision_regions(X_train, y_train.ravel(), model)
plt.xlabel('Duljina kljuna (mm)')
plt.ylabel('Duljina peraje (mm)')
plt.title('Granica odluke - skup za učenje')
plt.legend()
plt.show()

# e)
y_pred = model.predict(X_test)
print(f"Točnost: {accuracy_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.show()

print(classification_report(y_test, y_pred))


# f)
input_variables_f = ['bill_length_mm',
                     'flipper_length_mm',
                     'bill_depth_mm',
                     'body_mass_g']

X_f = df[input_variables_f].to_numpy()
y_f = df[output_variable].to_numpy()

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_f, y_f, test_size=0.2, random_state=123
)

y_train_f = y_train_f.astype(int)
y_test_f = y_test_f.astype(int)

model_f = LogisticRegression(max_iter=1000)
model_f.fit(X_train_f, y_train_f.ravel())

y_pred_f = model_f.predict(X_test_f)

print(f"Točnost s 4 veličine: {accuracy_score(y_test_f, y_pred_f):.4f}")
print(classification_report(y_test_f, y_pred_f))