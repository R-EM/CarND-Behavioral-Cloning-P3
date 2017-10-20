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
middle_driving_lap1_samples = []

folder_names = ['data3', 'data_udacity', 'middle_driving_lap1']
data_samples = []


for i in range(len(folder_names)):
	with open('./', folder_names[i], '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			data_samples.append(line)

sample_size = 32
sample_multiplier = 2

# Function to resize image
def resize_img(img):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(img, (64,64))

# Function for data augentation
def data_augmentation(batch_samples, folder_name):
	images = []
	angles = []

	for batch_sample in batch_samples:
		for i in range(1):
			name = folder_name + batch_sample[i].split('/')[-1]
			image = cv2.imread(name)
			angle = float(batch_sample[3])
			#correction = 0.4

			#if i == 1:
			#	angle += correction
			#if i == 2:
			#	angle -= correction
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

# Pre-processing of data
def model_preprocessing(model):
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	model.add(Lambda(resize_img))
	return model

def NVidia_model(model, dropout_rate):

	model.add(Convolution2D(24,5,5, subsample=(2,2), activation = "relu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation = "relu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation = "relu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(64,3,3, activation = "relu"))
	#model.add(Dropout(dropout_rate))
	model.add(Convolution2D(64,3,3, activation = "relu"))

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
	return model
	#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

def train_model(model, samples, folder_name, epochs):
	#train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	train_samples = samples
	validation_folder_name = './middle_driving_lap1/IMG/'
	validation_samples = middle_driving_lap1_samples
	train_generator = generator(train_samples, sample_size, folder_name)
	validation_generator = generator(validation_samples, sample_size, validation_folder_name)

	model.fit_generator(train_generator, samples_per_epoch = len(train_samples*sample_multiplier), validation_data = validation_generator, nb_val_samples = len(validation_samples*sample_multiplier), nb_epoch=epochs)
	return model

dropout_rate = 0.5	
epochs = 1

model = Sequential()
model = model_preprocessing(model)
model = NVidia_model(model, dropout_rate)


for i in range(3):
	print("For loop: ", i+1)
	# Use Udacity's training samples
	folder_name = './data_udacity/IMG/'
	model = train_model(model, udacity_samples, folder_name, epochs = 1)

	# Use my training samples
	folder_name = './data3/IMG/'
	model = train_model(model, my_samples, folder_name, epochs = 1)


model.save('model.h5')
