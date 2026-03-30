import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data_C02_emission.csv')

# print(data.columns.tolist())
# print(data.head())

# a)
input_columns = ['Engine Size (L)', 
                 'Cylinders',
                 'Fuel Consumption City (L/100km)',
                 'Fuel Consumption Hwy (L/100km)',
                 'Fuel Consumption Comb (L/100km)']

x = data[input_columns].values
y = data['CO2 Emissions (g/km)'].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# b)
feature = 'Engine Size (L)'
feature_index = 0

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, feature_index], y_train, color='blue',alpha=0.3, label='Training Data')
plt.scatter(X_test[:, feature_index], y_test, color='red', alpha=0.3, label='Test Data')
plt.xlabel(feature)
plt.ylabel('CO2 Emissions (g/km)')
plt.title(f'Ovisnost CO2 o {feature}')
plt.legend()
plt.show()


# c)
sc = MinMaxScaler()
X_train_n = sc.fit_transform ( X_train )
X_test_n = sc.transform ( X_test )
plt.hist(X_train[:, feature_index], bins=20, color='red', alpha=0.7, label='Prije skaliranja')
plt.title('Prije skaliranja')
plt.show()
plt.hist(X_train_n[:, feature_index], bins=20, color='blue', alpha=0.7, label='Nakon skaliranja')
plt.title('Nakon skaliranja')
plt.show()

#d)
import sklearn . linear_model as lm
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)  # X_train_n, ne X_train, jer želimo koristiti skalirane podatke
print('Koeficijenti: ', linearModel.coef_)
print('Presjek: ', linearModel.intercept_)

#e)
# Predviđanje na testnom skupu
y_pred = linearModel.predict(X_test_n)

# Scatter plot stvarno vs predviđeno
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.3)
plt.xlabel('Stvarne vrijednosti (g/km)')
plt.ylabel('Predviđene vrijednosti (g/km)')
plt.title('Stvarne vs Predviđene vrijednosti CO2')
plt.show()

#f)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np

MSE  = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
MAE  = mean_absolute_error(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
R2   = r2_score(y_test, y_pred)

print(f"MSE:  {MSE:.2f}")
print(f"RMSE: {RMSE:.2f}")
print(f"MAE:  {MAE:.2f}")
print(f"MAPE: {MAPE:.2f}")
print(f"R2:   {R2:.4f}")

#g)
# Testiraj s različitim brojem ulaznih veličina
input_columns_list = [
    ['Engine Size (L)'],                                        # 1 veličina
    ['Engine Size (L)', 'Cylinders'],                          # 2 veličine
    ['Engine Size (L)', 'Cylinders',
     'Fuel Consumption City (L/100km)'],                       # 3 veličine
    ['Engine Size (L)', 'Cylinders',
     'Fuel Consumption City (L/100km)',
     'Fuel Consumption Hwy (L/100km)'],                        # 4 veličine
    ['Engine Size (L)', 'Cylinders',
     'Fuel Consumption City (L/100km)',
     'Fuel Consumption Hwy (L/100km)',
     'Fuel Consumption Comb (L/100km)']                        # 5 veličina
]

for cols in input_columns_list:
    X = data[cols].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    sc = MinMaxScaler()
    X_train_n = sc.fit_transform(X_train)
    X_test_n = sc.transform(X_test)

    model = lm.LinearRegression()
    model.fit(X_train_n, y_train)
    y_pred = model.predict(X_test_n)

    MAE = mean_absolute_error(y_test, y_pred)
    R2  = r2_score(y_test, y_pred)
    print(f"Broj veličina: {len(cols):1d} | MAE: {MAE:.2f} | R²: {R2:.4f}")