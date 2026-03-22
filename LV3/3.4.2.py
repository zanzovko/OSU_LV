import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

# a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
plt.xlabel('CO2 Emisije (g/km)')
plt.ylabel('Broj vozila')
plt.title('Histogram emisije CO2 plinova')
plt.show()

# b)
plt.figure()
fuel_types = data['Fuel Type'].unique()
colors = ['blue', 'red', 'green', 'orange', 'purple']

for fuel, color in zip(fuel_types, colors):
    subset = data[data['Fuel Type'] == fuel]
    plt.scatter(subset['Fuel Consumption City (L/100km)'],
                subset['CO2 Emissions (g/km)'],
                label=fuel, color=color, s=10)

plt.xlabel('Gradska potrošnja goriva (L/100km)')
plt.ylabel('CO2 Emisije (g/km)')
plt.title('Odnos gradske potrošnje i emisije CO2')
plt.legend()
plt.show()

# c)
data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.xlabel('Tip goriva')
plt.ylabel('Izvangradska potrošnja goriva (L/100km)')
plt.title('Razdioba izvangradske potrošnje po tipu goriva')
plt.suptitle('')
plt.show()

# d) 
plt.figure()
grouped_fuel = data.groupby('Fuel Type')
grouped_fuel.size().plot(kind='bar')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.title('Broj vozila po tipu goriva')
plt.xticks(rotation=0)
plt.show()

# e)
plt.figure()
grouped_cyl = data.groupby('Cylinders')
grouped_cyl['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.xlabel('Broj cilindara')
plt.ylabel('CO2 Emisije (g/km)')
plt.title('Prosječna CO2 emisija po broju cilindara')
plt.xticks(rotation=0)
plt.show()