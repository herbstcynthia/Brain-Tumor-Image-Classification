import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib

train_dir = pathlib.Path('Data/Training')
test_dir = pathlib.Path('Data/Testing')
 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=None,
    seed=123,
    image_size=(180,180)
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  validation_split=None,
  seed=123,
  image_size=(180, 180),
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    tf.keras.layers.Rescaling(1./255),
    Conv2D(filters = 32,kernel_size = (3, 3), input_shape = (512, 512, 3), activation = "sigmoid"),
    MaxPooling2D(pool_size = (6,6)),
    Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"),
    MaxPooling2D(pool_size = (6,6)),
    Flatten(),
    Dense(units = 128, activation = "sigmoid"),
    Dense(units = 4, activation = "sigmoid")
])

model.compile(optimizer = "adam", loss = "CategoricalCrossentropy", metrics = ["accuracy"])

train_datagen = ImageDataGenerator(
                                   rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_dataset = train_datagen.flow_from_directory('Data/Training',
                                                     target_size=(512, 512),
                                                     batch_size=64,
                                                     class_mode='categorical')



testing_dataset = test_datagen.flow_from_directory('Data/Testing',
                                                   target_size=(512, 512),
                                                   batch_size=64,
                                                   class_mode='categorical')


history = model.fit(
    training_dataset,
    epochs=5,
    verbose=True,
    validation_data=testing_dataset
)

print(history)





