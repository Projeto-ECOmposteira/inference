import os
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers

DATASET_PATH = '/dataset'

test_datagen = ImageDataGenerator(rescale = 1./255)
train_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'TRAIN'),
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

test_set = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'TEST'),
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary'
)

classifier = Sequential(
    [
        Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        Conv2D(32, (3, 3), activation = 'relu'),
        MaxPooling2D(pool_size = (2, 2)),
        Flatten(),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 1, activation = 'sigmoid'),
    ]
)

callback_checkpoint = model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

classifier.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy'],
)

classifier.load_weights('checkpoints')

classifier.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=10,
    callbacks=[callback_checkpoint],
)

classifier.save('model.tf')
