import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('model.tf')
generator = ImageDataGenerator(rescale=1./255)

