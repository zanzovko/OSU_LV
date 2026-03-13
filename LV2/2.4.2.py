import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)

print("Broj osoba:", len(data))


plt.scatter(data[:, 1], data[:, 2])
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Visina vs Masa')
plt.show()


plt.scatter(data[::50, 1], data[::50, 2])
plt.xlabel('Visina (cm)')
plt.ylabel('Masa (kg)')
plt.title('Visina vs Masa')
plt.show()


min = data[:, 1].min()
max = data[:, 1].max()
mean = data[:, 1].mean()
print(f"Visina - Min: {min}m, Max: {max}m, Prosjek: {mean}m")


indexMale = (data[:,0] == 1)
indexFemale = (data[:,0] == 0)
heightMale = data[indexMale, 1]
heightFemale = data[indexFemale, 1]

print(f"Muškarci - Min: {heightMale.min()}m, Max: {heightMale.max()}m, Prosjek: {heightMale.mean()}m")
print(f"Žene - Min: {heightFemale.min()}m, Max: {heightFemale.max()}m, Prosjek: {heightFemale.mean()}m")