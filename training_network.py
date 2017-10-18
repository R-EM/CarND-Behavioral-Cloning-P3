import csv
import os
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from random import shuffle

samples = []
with open('./data3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

sample_size = 32
#batch_size = 192

def generator(samples, sample_size):
	num_samples = len(samples)

	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, sample_size):
			batch_samples = samples[offset:offset+sample_size]


			images = []
			angles = []

			for batch_sample in batch_samples:
				for i in range(3):
					name = './data3/IMG/' + batch_sample[i].split('/')[-1]
					image = cv2.imread(name)
					angle = float(batch_sample[3])
					images.append(image)
					angles.append(angle)

			X_train = np.array(images)
			y_train = np.array(angles)

			augmented_images, augmented_angles = [], []
			
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle*-1.0)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)

			X_train, y_train = shuffle(X_train, y_train)

			yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, sample_size)
validation_generator = generator(validation_samples, sample_size)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

# Pre-processing of data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Dropout(0.5))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples*6), validation_data = validation_generator, nb_val_samples = len(validation_samples*6), nb_epoch=4)
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')
