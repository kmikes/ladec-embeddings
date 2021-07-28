import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

end_to_end_model = keras.models.load_model('classifer.mod')

probabilities = end_to_end_model.predict(
    [["this message is about computer graphics and 3D modeling"], ['this message is about why god does not exist']]
)

print( probabilities )