import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv

# Data augmentation - flip images
center = cv2.imread('center.jpg')
left = cv2.imread('left.jpg')
right = cv2.imread('right.jpg')

center_flip = cv2.flip(center,1)
left_flip = cv2.flip(left,1)
right_flip = cv2.flip(right,1)

cv2.imwrite('center_flip.jpg', center_flip)
cv2.imwrite('left_flip.jpg', left_flip)
cv2.imwrite('right_flip.jpg', right_flip)


# Training loss plot
train_loss = 	[0.0153, 0.0109, 0.0093, 0.0082, 0.0074, 0.0065, 0.0061, 0.0057, 0.0053, 0.0050, 0.0047, 0.0046, 0.0043, 0.0042, 0.0042, 0.0041, 0.0039, 0.0038, 0.0037, 0.0036]
val_loss = 		[0.0113, 0.0097, 0.0087, 0.0087, 0.0075, 0.0069, 0.0070, 0.0068, 0.0069, 0.0066, 0.0068, 0.0068, 0.0070, 0.0066, 0.0063, 0.0059, 0.0062, 0.0063, 0.0063, 0.0063]

#plt.plot(train_loss, label="Training loss")
#plt.plot(val_loss, label="Validation loss")

#plt.ylabel('Loss')
#plt.xlabel('EPOCHS')

#plt.legend(loc='upper right')
#plt.show()


# steering angle histogram
def get_angles(folder_name, file_names):
	data_samples = []

	for i in range(len(file_names)):
		with open(folder_name + file_names[i] + '_driving_log.csv') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				data_samples.append(line)

	angles = []
	for sample in data_samples:
		angles.append(float(sample[3]))
	return angles

folder_name = '../data_driving/'

# Udacity's angles
file_names = ['data_udacity']
udacity_angles = get_angles(folder_name, file_names)

# My angles
file_names = ['middle_driving_lap1', 'middle_driving_lap2', 'middle_driving_lap3', 'middle_driving_reverse_lap1', 'middle_driving_reverse_lap2']
angles = get_angles(folder_name, file_names)

plt.hist([angles, udacity_angles], bins = 100)
plt.legend(["My angles","Udacity's angles"])
#bins = np.arange(-100, 100, 5) # fixed bin size
#angles.hist(bins=bins)
#plt.hist(angles, bins = bins)
plt.show()