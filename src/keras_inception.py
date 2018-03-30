import os,sys
import time
import keras
import numpy as np

from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from data_helper import load_data
print ("Libraries loaded successfully")

IMG_SIZE=256
IV3_LR=0
nb_classes = 14
FC_SIZE=1024

#loading the saved numpy training arrays
X, XF, Y = load_data(typ="train")
print("Training data shapes: ")
print(X.shape, Y.shape, XF.shape)

dev_sample_percentage=0.30
np.random.seed(17111)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
xf_shuffled = XF[shuffle_indices]
y_shuffled = Y[shuffle_indices]

dev_sample_index = -1 * int(dev_sample_percentage * float(len(Y)))
x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
xf_train, xf_val = xf_shuffled[:dev_sample_index], xf_shuffled[dev_sample_index:]
y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Randomly chosen %02d % data for training...", (1-dev_sample_percentage)*100)
print(x_train.shape, xf_train.shape, y_train.shape)

print("Randomly chosen %02d % data for validation...", (dev_sample_percentage)*100)
print(x_val.shape, xf_val.shape, y_val.shape)

print("deleting the redundent variables to free up the memory...")
del X, Y, XF, xf_shuffled, x_shuffled, y_shuffled

'''
borrowing the Google's inception V3 architecture except the densely connected layers with
the pretrained weights on Stanford's imagenet dataset.
'''
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))


'''
method to add the new last layers based on number of classes in new data
'''
def add_new_last_layer(base_model, nb_classes):
	'''
	Args:
		base_model: Inception V3 without last layer
		nb_classes: #classes
	'''
	X = base_model.output
	X = GlobalAveragePooling2D()(X)
	X = Dense(FC_SIZE, activation='relu')(X)
	predictions = Dense(nb_classes, activation='softmax')(X)
	model = Model(input=base_model.input, output=predictions)
	return model

'''
method to disable the weight updation (training) for pretrained layers
'''
def transfer_learn(model, base_model):
	'''Train only the last layers '''
	for layer in base_model.layers:
		layer.trainable = False
	model.compile( optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

'''
method to enable the 
'''
def finetune_learn(model):
	''' Train all layers'''
	for layer in model.layers[:]:
		layer.trainable = True
	model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model = add_new_last_layer(base_model, nb_classes)

batch_size = 32
epochs = 2

train_datagen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True)
train_datagen.fit(x_train)

test_datagen = ImageDataGenerator(
	rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
test_datagen.fit(x_val)

transfer_learn(model, base_model)
'''Training begins'''
history = model.fit_generator(
	train_datagen.flow(x_train, y_train, batch_size=batch_size),
	steps_per_epoch=x_train.shape[0] // batch_size,
	epochs=epochs,
	validation_data=test_datagen.flow(x_val, y_val, batch_size=batch_size))
#model.save('InceptionV3_transfer.h5')
epochs = 10
finetune_learn(model)
history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=test_datagen.flow(x_val, y_val, batch_size=batch_size))
model.save('InceptionV3_finetune.h5')


