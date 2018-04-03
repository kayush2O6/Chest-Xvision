import os,sys
import time
import keras
import numpy as np
import _pickle as cPickle

from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical, generic_utils
from keras.preprocessing.image import ImageDataGenerator

from data_helper import load_data
print ("Libraries loaded successfully")


#----------------helper function to generate the batches-----------
def batch_gen(data, batch_size):
    num_batches = int((len(data)-1)/batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        yield data[start_index:end_index]
#-----------------------------------------------------------------


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
    Google's inception V3 architecture except the densely connected layers with
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
method to enable all the layers for weights updation
'''
def finetune_learn(model):
	'''freeze the bottom IV3_LR and retrain the remaining top layers'''
	for layer in model.layers[:IV3_LR]:
		layer.trainable = False
	for layer in model.layers[IV3_LR:]:
		layer.trainable = True
	model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])



'''
adding the last classificaion layers in base model
'''
model = add_new_last_layer(base_model, nb_classes)
tolerance = 1e-6
batch_size = 50
epochs = 2

'''
helping method to generate the images for training 
'''
train_datagen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True)
train_datagen.fit(x_train)

'''
helping method to generate the images for validation
'''
test_datagen = ImageDataGenerator(
	rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
test_datagen.fit(x_val)


'''Transfer learning learns only newly added last layers of model'''
transfer_learn(model, base_model)


'''feed the training and validation data into model'''
history = model.fit_generator(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        validation_data=test_datagen.flow(x_val, y_val, batch_size=batch_size))

'''Update the epoch value for fine tuning the model trained after transfer learning'''
epochs = 8

'''fine tuning learns all the layers of the model'''
finetune_learn(model)
max_acc = 0.0
for e in range(epochs):
	batches = batch_gen(list(zip(x_train, y_train)), batch_size)
	print('Epoch ',e)
	print('Training...')
	progbar = generic_utils.Progbar(x_train.shape[0])
	for batch in batches:
		x_batch, y_batch = zip(*batch)
		#print(np.array(x_batch).shape, np.array(y_batch).shape)
		'''a single training step'''
		#tr_loss = 0.0
		#tr_accuracy = 0.0
		tr_loss, tr_accuracy = model.train_on_batch(np.array(x_batch), np.array(y_batch))
		progbar.add(batch_size, values=[("train loss", tr_loss), ("train accuracy", tr_accuracy)])
		# If train accuracy is greater than 0.40 then validation
		print("trAcc:", tr_accuracy)
		if tr_accuracy >= 0.40:
			print ("\nValidation...")
			val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=50)
			print (val_acc)
			if max_acc < val_acc:
				max_acc=val_acc
				print(max_acc)
				if val_acc> 0.42:
					model.save('my_model.h5')
	
		
	
