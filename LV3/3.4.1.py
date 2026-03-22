import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print("\nPod a):\n")
# Koliko mjerenja sadrži DataFrame?
print("Broj mjerenja:", data.shape[0])
print("Broj veličina:", data.shape[1])

# Kojeg je tipa svaka veličina?
print("\nInformacije o DataFrameu:")
print(data.info())

# Postoje li izostale vrijednosti?
print("\nBroj izostalih vrijednosti po stupcu:")
print(data.isnull().sum())

# Postoje li duplicirane vrijednosti?
print("\nBroj dupliciranih vrijednosti:", data.duplicated().sum())

# Brisanje izostalih i dupliciranih vrijednosti
data = data.dropna(axis=0)
data = data.drop_duplicates()
data = data.reset_index(drop=True)

print("\nBroj redaka nakon brisanja:", data.shape[0])
category_cols = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
for col in category_cols:
    data[col] = data[col].astype('category')

print("\nTipovi veličina nakon konverzije:")
print(data.dtypes)

print("\nPod b):\n")
print("3 automobila s najvećom gradskom potrošnjom:")
vehicles_max_consumption = data.nlargest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
print(vehicles_max_consumption)

print("\n3 automobila s najmanjom gradskom potrošnjom:")
vehicles_min_consumption = data.nsmallest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']]
print(vehicles_min_consumption)

print("\nPod c):\n")
vehicles_filter = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print("Broj vozila s motorom između 2.5 i 3.5 L:", len(vehicles_filter))
print("Prosječna CO2 emisija:", vehicles_filter['CO2 Emissions (g/km)'].mean())

print("\nPod d):\n")
audi = data[data['Make'] == 'Audi']
print("Broj mjerenja za Audi:", len(audi))
audi_4cyl = audi[audi['Cylinders'] == 4]
print("Prosječna CO2 emisija Audi vozila s 4 cilindra:", audi_4cyl['CO2 Emissions (g/km)'].mean())

print("\nPod e):\n")
grouped = data.groupby('Cylinders')
print("Broj vozila po broju cilindara:")
print(grouped.size())
print("\nProsječna CO2 emisija po broju cilindara:")
print(grouped['CO2 Emissions (g/km)'].mean())

print("\nPod f):\n")
diesel = data[data['Fuel Type'] == 'D']
gasoline = data[data['Fuel Type'] == 'X']
print("Prosječna gradska potrošnja - dizel:", diesel['Fuel Consumption City (L/100km)'].mean())
print("Prosječna gradska potrošnja - regularni benzin:", gasoline['Fuel Consumption City (L/100km)'].mean())
print("\nMediana gradske potrošnje - dizel:", diesel['Fuel Consumption City (L/100km)'].median())
print("Mediana gradske potrošnje - regularni benzin:", gasoline['Fuel Consumption City (L/100km)'].median())

print("\nPod g):\n")
filter_4cyl = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
max_consumption_4cyl = filter_4cyl['Fuel Consumption City (L/100km)'].max()
result = filter_4cyl[filter_4cyl['Fuel Consumption City (L/100km)'] == max_consumption_4cyl][['Make', 'Model', 'Fuel Consumption City (L/100km)']]
print("Vozilo s 4 cilindra na dizel s najvećom gradskom potrošnjom:")
print(result)

print("\nPod h):\n")
manual = data[data['Transmission'].str.startswith('M')]
print("Broj vozila s ručnim mjenjačem:", len(manual))

print("\nPod i):\n")
correlation = data.corr(numeric_only=True)
print("Korelacija između numeričkih veličina:")
print(correlation)