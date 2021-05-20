import numpy as np
import tensorflow as tf
import cv2
from keras.models import model_from_json
from PIL import Image

from tensorflow.keras.models import Model
import numpy as np
from keras.models import model_from_json
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import sys


def getPrediction(image_path):
    image_path='static/'+image_path
    confedence=float(0.0)
    res=""
    model = tf.keras.models.load_model("model_0.997.h5")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, (224, 224))
    data=[image]
    data=np.array(data) / 255.0
    classes=model.predict(data,batch_size=1)
    confedence=float(max(classes[0]))
    pred = np.argmax(classes, axis=1)
    if pred==[0]:
        res='Predicted as Normal'
    else:
        res='Predicted as Covid-19'
    return res, confedence



