
import tensorflow as tf
import keras
import numpy as np
import glob
import csv
import cv2
import matplotlib.pyplot as plt


lines = []
with(open('./data/driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #if line.split("/")[1].split("_")[0] == 'center':
        lines.append(line)

# Removing the column title row
lines = lines[1:]


# Initiating a blank dictionary for data
data = {'train':[], 'validation':[]}
for line in lines:
    # Reading the steering measurement
    measurement = float(line[3])
    # Removing about 95% of the images with measurement = 0
    if measurement != 0 or np.random.rand() > 0.95:
        # loop 3 times for three different positions of left, center, right
        for i in range(3):
            datapoint = {}
            datapoint['path'] = './data/' + line[i].lstrip()
            datapoint['position'] = line[i].split('/')[1].split('_')[0]
            datapoint['flipped'] = False
            # Adding or subtracting 5.0 (can be tuned) to the steering angle based on the camera position
            if datapoint['position'] == 'right':                
                datapoint['measurement'] = measurement - 5.0
            elif datapoint['position'] == 'left':
                datapoint['measurement'] = measurement + 5.0
            else:
                datapoint['measurement'] = measurement
            if datapoint['position'] == 'center' and np.random.rand() > 0.8:
                data['validation'].append(datapoint)
            else:
                data['train'].append(datapoint)
            # Creating a copy of image dictionary to avoid pointing to the previous one
            datapoint = dict(datapoint)    
            # Augmenting with the flipped image
            datapoint['flipped'] = True
            datapoint['measurement'] = -datapoint['measurement']
            data['train'].append(datapoint)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, samples, batch_size=128, dim=(160,320), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.samples = samples
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_datapoints_temp = [self.samples[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_datapoints_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_datapoints_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=float)

        # Generate data
        for i, datapoint in enumerate(list_datapoints_temp):
            image = plt.imread(datapoint['path'])
            # flipping horizontal if necessary
            if datapoint['flipped']:
                image = cv2.flip(image, 1)
            # Store sample
            X[i,] = image

            # Store value
            y[i] = datapoint['measurement']

        return X, y


params = {'dim': (160,320), 
          'batch_size': 128, 
          'n_channels': 3,
          'shuffle': True}


training_generator = DataGenerator(data['train'], **params)
validation_generator = DataGenerator(data['validation'], **params)


training_generator.__getitem__(0)[0][0].shape


def image_preprocess(images):
    images = tf.image.rgb_to_yuv(images)
    images = tf.image.crop_to_bounding_box(images, 60, 0, 80, 320)
    images = tf.image.resize(images, [66,200])
    images = tf.math.add(tf.math.divide(images,128), -1)
    return images


# CNN Architecture
## Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Lambda
from keras.layers import Input


# Building DAVE2 Architecture in Keras 
model = Sequential()

# preprocessing
#model.add(Lambda(image_preprocess, input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/128.0 - 1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))

model.add(Conv2D(24, (5,5), strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(36, (5,5), strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5,5), strides=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('tanh'))


print(model.summary())


model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator, 
                    epochs=10,
                    use_multiprocessing=True,
                    workers=6)
model.save('model.h5')



