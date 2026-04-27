import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Učitaj CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Skaliraj na [0,1]
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN s DROPOUT slojevima
model = keras.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_n, y_train,
          epochs=40,
          batch_size=64,
          validation_split=0.1)

score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Točnost na testnom skupu: {100.0*score[1]:.2f}%')