import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.image as Image

# 1. UČITAJ MODEL
model = keras.models.load_model("FCN.keras")
print("Model učitan!")

# 2. UČITAJ SLIKU S DISKA
img = Image.imread("test.png")
print(f"Originalna dimenzija slike: {img.shape}")

# 3. PRIKAŽI ORIGINALNU SLIKU
plt.figure()
plt.imshow(img)
plt.title("Originalna slika test.png")
plt.show()

# 4. PRIPREMI SLIKU ZA MREŽU
# Ako je slika u boji (RGB), pretvori u grayscale
if len(img.shape) == 3:
    # uzmi samo jedan kanal ili prosjek
    img = np.mean(img, axis=2)

# Promijeni dimenzije na 28x28 ako nisu
from PIL import Image as PILImage
img_pil = PILImage.open("test.png").convert('L')  # 'L' = grayscale
img_pil = img_pil.resize((28, 28))
img_array = np.array(img_pil)

# Invertiraj boje ako je crna brojka na bijeloj pozadini
# MNIST ima BIJELU brojku na CRNOJ pozadini!
img_array = 255 - img_array  # invertiraj

# Skaliraj na [0,1]
img_array = img_array.astype("float32") / 255.0

# Prikaži pripremljenu sliku
plt.figure()
plt.imshow(img_array, cmap='gray')
plt.title("Pripremljena slika (28x28, invertirano)")
plt.show()

# 5. PRILAGODI OBLIK ZA MREŽU (1, 28, 28, 1)
img_array = np.expand_dims(img_array, axis=0)   # batch dimenzija
img_array = np.expand_dims(img_array, axis=-1)  # kanal dimenzija
print(f"Oblik za mrežu: {img_array.shape}")

# 6. KLASIFIKACIJA
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class] * 100

print(f"\nPredviđena znamenka: {predicted_class}")
print(f"Sigurnost: {confidence:.2f}%")

# Ispis svih vjerojatnosti
print("\nVjerojatnosti za sve klase:")
for i, prob in enumerate(prediction[0]):
    print(f"  Klasa {i}: {prob*100:.2f}%")