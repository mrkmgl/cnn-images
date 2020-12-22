import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

#Opening dataset from the other file
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
y = np.array(y)

X = X/255.0

#Determing layers for our model
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential() #simple sequential model

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:])) #convolutional 1st layer
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2))) #pooling

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3))) #convolutional 2nd layer
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2))) #pooling

            model.add(Flatten())

            for _ in range(dense_layer):       #hidden layers
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))                #output layer
            model.add(Activation('sigmoid')) 

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME)) #used for analysis


            #Parameters for the model
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )
            #Training the model
            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      callbacks=[tensorboard])

#Saving the model
model.save('64x3-CNN.model')

#Model prediction
import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]


def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('doggo.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
