import csv
import mlflow
import requests
import numpy as np
import tensorflow as tf
import os 
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()
data_dir = args.data_dir

# data_dir = '../blobstore/cars'
img_width, img_height = 224, 224
# num_train_samples = 8144
# num_validation_samples = 8041
num_train_samples = 100
num_validation_samples = 10
epochs = 1
batch_size = 8
num_classes = 196

def build_model():

    weights_file = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    if not os.path.exists(weights_file):
        url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        r = requests.get(url, allow_redirects=True)
        open(weights_file, 'wb').write(r.content)

    base_model = densenet.DenseNet121(input_shape=(img_width, img_height, 3),
                                      weights=weights_file,
                                      include_top=False,
                                      pooling='avg')
    
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

class_names = []
with open(data_dir + '/names.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    for row in csv_reader:
        class_names.append(row[0])

train_data_dir = data_dir + '/car_data/train'
validation_data_dir = data_dir + '/car_data/test'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

mlflow.tensorflow.autolog()

model_history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch = num_train_samples // batch_size,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // batch_size,
    callbacks=callbacks_list)

mlflow.keras.save_model(model, 'model')