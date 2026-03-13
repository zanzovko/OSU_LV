import numpy as np
import matplotlib.pyplot as plt

crno = np.zeros((50, 50))  
bijelo = np.ones((50, 50))  

gornji_red = np.hstack((crno, bijelo))   
donji_red  = np.hstack((bijelo, crno))   

slika = np.vstack((gornji_red, donji_red))  

plt.imshow(slika, cmap="gray")
plt.title("Četiri kvadrata")
plt.show()