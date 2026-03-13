import numpy as np
import matplotlib.pyplot as plt

try:
    slika = plt.imread('road.jpg')
except FileNotFoundError:
    print("Slika 'road.jpg' nije pronađena. Provjerite da se nalazi u istom direktoriju kao i ovaj skript.")
    exit()


svijetla = slika + 50
svijetla = np.clip(svijetla, 0, 255)

plt.subplot(1, 2, 1)
plt.imshow(slika)
plt.title("Originalna")

plt.subplot(1, 2, 2)
plt.imshow(svijetla)
plt.title("Posvijetljena")

plt.tight_layout()
plt.show()



sirina = slika.shape[1]

cetvrtina = sirina // 4
druga_cetvrtina = slika[:, cetvrtina:cetvrtina*2, :]

plt.imshow(druga_cetvrtina)
plt.title("Druga četvrtina po širini")
plt.show()


rotirana = np.rot90(slika, k=-1)  # k=-1 = smjer kazaljke na satu

plt.subplot(1, 2, 1)
plt.imshow(slika)
plt.title("Originalna")

plt.subplot(1, 2, 2)
plt.imshow(rotirana)
plt.title("Rotirana 90°")

plt.tight_layout()
plt.show()



zrcaljena = np.fliplr(slika)

plt.subplot(1, 2, 1)
plt.imshow(slika)
plt.title("Originalna")

plt.subplot(1, 2, 2)
plt.imshow(zrcaljena)
plt.title("Zrcaljena")

plt.tight_layout()
plt.show()