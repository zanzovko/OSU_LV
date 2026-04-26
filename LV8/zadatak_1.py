import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# prikaz karakteristika train i test podataka
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f'Oznaka: {y_train[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Ispis prve oznake
print(f"Oznaka prve slike: {y_train[0]}")

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")


# pretvori labele
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)


# TODO: kreiraj model pomocu keras.Sequential(); prikazi njegovu strukturu
model = keras.Sequential()
model.add(layers.Input(shape=(28, 28, 1)))
model.add(layers.Flatten())                      # 28×28 → 784
model.add(layers.Dense(100, activation="relu"))  # 1. skriveni sloj
model.add(layers.Dense(50, activation="relu"))   # 2. skriveni sloj
model.add(layers.Dense(10, activation="softmax"))# izlazni sloj
model.summary()


# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


# TODO: provedi ucenje mreze
batch_size = 32
epochs = 10

history = model.fit(
    x_train_s,
    y_train_s,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# 6. EVALUACIJA NA TESTNOM SKUPU
score = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"\nTest loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")

# TODO: Prikazi test accuracy i matricu zabune
predictions = model.predict(x_test_s)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Matrica zabune')
plt.show()


# TODO: spremi model
model.save("FCN.keras")
print("\nModel spremljen u FCN.keras")
