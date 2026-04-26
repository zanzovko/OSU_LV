import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
# img = Image.imread("imgs/imgs/test_1.jpg")
# img = Image.imread("imgs/imgs/test_2.jpg")
# img = Image.imread("imgs/imgs/test_3.jpg")
# img = Image.imread("imgs/imgs/test_4.jpg")
# img = Image.imread("imgs/imgs/test_5.jpg")
img = Image.imread("imgs/imgs/test_6.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


# 1. BROJ ORIGINALNIH BOJA
unique_colors = len(np.unique(img_array, axis=0))
print(f"Broj originalnih boja: {unique_colors}")

# 2. K-MEANS GRUPIRANJE NA RGB VRIJEDNOSTIMA
K = 5  # broj boja
km = KMeans(n_clusters=K, init='k-means++', n_init=5, random_state=0)
km.fit(img_array)

# 3. ZAMJENA SVAKOG PIKSELA S NAJBLIŽIM CENTROM
labels = km.predict(img_array)
img_array_aprox = km.cluster_centers_[labels]

# Vrati u 3D oblik
img_quantized = np.reshape(img_array_aprox, (w, h, d))

# 4. USPOREDBA ORIGINALA I KVANTIZIRANE SLIKE
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img)
axes[0].set_title(f'Originalna slika ({unique_colors} boja)')
axes[1].imshow(img_quantized)
axes[1].set_title(f'Kvantizirana slika (K={K} boja)')
plt.tight_layout()
plt.show()

# 6. LAKAT METODA
inertias = []
K_range = range(1, 11)

for k in K_range:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=0)
    km_temp.fit(img_array)
    inertias.append(km_temp.inertia_)

plt.figure()
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Broj grupa K')
plt.ylabel('Inertia (J)')
plt.title('Lakat metoda')
plt.grid()
plt.show()

# 7. BINARNE SLIKE ZA SVAKU GRUPU
fig, axes = plt.subplots(1, K, figsize=(15, 4))
for k in range(K):
    binary_img = (labels == k).reshape(w, h)
    axes[k].imshow(binary_img, cmap='gray')
    axes[k].set_title(f'Grupa {k}')
    axes[k].axis('off')
plt.tight_layout()
plt.show()
