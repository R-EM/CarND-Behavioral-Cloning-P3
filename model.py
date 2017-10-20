import csv
import os
import cv2
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from random import shuffle

my_samples = []
udacity_samples = []

with open('./data3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		my_samples.append(line)

with open('./data_udacity/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		udacity_samples.append(line)

sample_size = 32

# Function to resize image
def resize_img(img):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(img, (64,64))

# Function for data augentation
def data_augmentation(batch_samples, folder_name):
	images = []
	angles = []

	for batch_sample in batch_samples:
		for i in range(3):
			name = folder_name + batch_sample[i].split('/')[-1]
			image = cv2.imread(name)
			angle = float(batch_sample[3])
			correction = 0.4

			if i == 1:
				angle += correction
			if i == 2:
				angle -= correction
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

	return X_train, y_train

# Function to generate data
def generator(samples, sample_size, folder_name):
	num_samples = len(samples)

	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, sample_size):
			batch_samples = samples[offset:offset+sample_size]

			X_train, y_train = data_augmentation(batch_samples, folder_name)

			yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def NVidia_model(train_gen, train_samp, validation_gen, validation_samp, epochs):
	

	# Pre-processing of data
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	model.add(Lambda(resize_img))

	dropout_rate = 0.5

	model.add(Convolution2D(24,5,5, subsample=(2,2), activation = "elu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation = "elu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation = "elu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(64,3,3, activation = "elu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(64,3,3, activation = "elu"))

	model.add(Dropout(dropout_rate))


	model.add(Flatten())
	#model.add(Dropout(dropout_rate))
	model.add(Dense(100))
	#model.add(Dropout(dropout_rate))
	model.add(Dense(50))
	#model.add(Dropout(dropout_rate))
	model.add(Dense(10))
	#model.add(Dropout(dropout_rate))
	model.add(Dense(1))
	#model.add(Dropout(dropout_rate))

	model.compile(loss = 'mse', optimizer = 'adam')
	model.fit_generator(train_gen, samples_per_epoch = len(train_samp*6), validation_data = validation_gen, nb_val_samples = len(validation_samp*6), nb_epoch=epochs)
	#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

	
model = Sequential()

# Udacity training samples
folder_name = './data_udacity/IMG/'
train_samples, validation_samples = train_test_split(udacity_samples, test_size=0.2)
train_generator = generator(train_samples, sample_size, folder_name)
validation_generator = generator(validation_samples, sample_size, folder_name)
NVidia_model(train_generator, train_samples, validation_generator, validation_samples, 2)


# My training samples
folder_name = './data3/IMG/'
train_samples, validation_samples = train_test_split(my_samples, test_size=0.2)
train_generator = generator(train_samples, sample_size)
validation_generator = generator(validation_samples, sample_size, folder_name)
NVidia_model(train_generator, train_samples, validation_generator, validation_samples, 2)


model.save('model.h5')