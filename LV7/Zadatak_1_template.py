import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# # generiranje podatkovnih primjera
# X = generate_data(500, 1)

# # prikazi primjere u obliku dijagrama rasprsenja
# plt.figure()
# plt.scatter(X[:,0],X[:,1])
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# plt.title('podatkovni primjeri')
# plt.show()


# Generiranje podataka
flagc = 1 
X = generate_data(500, flagc)

# 1. PRIKAZ ORIGINALNIH PODATAKA
plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Podatkovni primjeri')
plt.show()

# 2. K-MEANS GRUPIRANJE
K = 3  # broj grupa
km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
            marker='X', s=200, c='red', label='Centri')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'K-means grupiranje (K={K})')
plt.legend()
plt.show()

# 3. LAKAT METODA - optimalni K
inertias = []
K_range = range(1, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Broj grupa K')
plt.ylabel('Inertia (J)')
plt.title('Lakat metoda')
plt.grid()
plt.show()


# ZAKLJUČCI:
# 
# flagc=1: 3 sferične grupe → K=3 daje savršen rezultat
# flagc=2: 3 razvučene grupe → K=3 radi solidno
# flagc=3: 4 grupe različitih veličina → K=4 optimalno
# flagc=4: 2 koncentrična kruga → K-means NE RADI dobro!
# flagc=5: 2 polumjeseca → K-means NE RADI dobro!
#
# K-means dobro radi za sferične/konveksne grupe.
# Za nelinearne oblike (krugovi, polumjeseci) potrebni su drugi algoritmi.