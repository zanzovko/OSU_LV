import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# 1. UČITAJ MODEL
model = keras.models.load_model("FCN.keras")
model.summary()

# 2. UČITAJ MNIST PODATKE
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Pripremi test podatke (isto kao u zadatku 1)
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# 3. PREDIKCIJA
predictions = model.predict(x_test_s)
y_pred = np.argmax(predictions, axis=1)

# 4. PRONAĐI LOŠE KLASIFICIRANE SLIKE
wrong_indices = np.where(y_pred != y_test)[0]
print(f"Broj loše klasificiranih slika: {len(wrong_indices)}")
print(f"Točnost: {(1 - len(wrong_indices)/len(y_test))*100:.2f}%")

# 5. PRIKAŽI 10 LOŠE KLASIFICIRANIH SLIKA
plt.figure(figsize=(15, 6))
for i in range(10):
    idx = wrong_indices[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f'Stvarna: {y_test[idx]}\nPredviđena: {y_pred[idx]}')
    plt.axis('off')

plt.tight_layout()
plt.show()