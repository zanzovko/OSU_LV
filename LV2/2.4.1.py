import numpy as np
import matplotlib.pyplot as plt

# try:
#     slika = plt.imread('road.jpg')
# except FileNotFoundError:
#     print("Slika 'road.jpg' nije pronađena. Provjerite da se nalazi u istom direktoriju kao i ovaj skript.")
#     exit()

# img = slika [ :,:,0].copy()
# print ( img . shape )
# print ( img . dtype )
# plt . figure ()
# plt . imshow ( img , cmap ="gray")
# plt . show ()
plt.close('all') 

x = [1, 2, 3, 3, 1]
y = [1, 2, 2, 1, 1]

plt . plot (x , y , 'b', linewidth =1 , marker =".", markersize =10 )
plt . axis ([0 ,4 ,0 , 4])
plt . xlabel ('x os')
plt . ylabel ('y os')
plt . title ('Zadatak 2.4.1')
plt . show ()