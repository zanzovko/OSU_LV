import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

input_columns = [
    'Engine Size (L)',
    'Cylinders',
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)'
]

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

X_num = data[input_columns].values
X = np.hstack((X_num, X_encoded))

y = data['CO2 Emissions (g/km)'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

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

errors = np.abs(y_test - y_pred)
max_error_index = np.argmax(errors)
print(f"\nMaksimalna pogreška: {errors[max_error_index]:.2f} g/km")

print(f"Stvarna vrijednost:    {y_test[max_error_index]:.2f} g/km")
print(f"Predviđena vrijednost: {y_pred[max_error_index]:.2f} g/km")

test_indices = X_test
original_index = y_test
print(data.iloc[max_error_index][['Make', 'Model', 'Fuel Type']])