import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def predict(image):
    model = tf.keras.models.load_model('./model.h5')
    #test_img = mpimg.imread(image)
    test_img = np.asarray(image)
    class_types = ['healthy','sepotria','stripe_rust']

    img = np.resize(test_img, (400,600,3))
    fin = np.reshape(img, (1,400,600,3))
    print(fin.shape)
    pred = model.predict(fin)
    return class_types[np.argmax(pred)]
