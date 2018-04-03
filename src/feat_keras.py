import os,sys
import time
import keras
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from keras.utils.vis_utils import plot_model
from keras.utils.training_utils import multi_gpu_model
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Merge, Dropout, Flatten, Reshape, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from data_helper import load_data
print ("Libraries loaded successfully")

IMG_SIZE=256
IV3_LR=0
nb_classes = 14
FC_SIZE_1=512
FC_SIZE_2=256

#loading the saved numpy training arrays
X, XF, Y = load_data(typ="train")
print("Training data shapes: ")
print(X.shape, Y.shape, XF.shape)

dev_sample_percentage=0.20
np.random.seed(47)
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
	model_base = Sequential()
	model_base.add(Flatten(input_shape=base_model.output_shape[1:])) 
#	model_base.add(GlobalAveragePooling2D(input_shape=(8, 8, 512)))	
	model_conv = Model(inputs=base_model.input, outputs=model_base(base_model.output)) 
	
	model_feat = Sequential()
	model_feat.add(Reshape((3,), input_shape=(3,)) )
	merged = Merge([model_conv, model_feat], mode='concat')	
	
	final_model = Sequential()
	final_model.add(merged)
	final_model.add(Dropout(0.3))
	final_model.add(Dense(int(FC_SIZE_1*2), activation='relu'))
	final_model.add(Dropout(0.3))
#	final_model.add(Dense(FC_SIZE_1, activation='relu'))
	final_model.add(Dense(FC_SIZE_1, activation='relu'))
	final_model.add(Dense(FC_SIZE_2, activation='relu'))
#	final_model.add(Dense(FC_SIZE_2, activation='relu'))
#	final_model.add(Dense(FC_SIZE_2, activation='relu'))
	final_model.add(Dropout(0.3))
	final_model.add(Dense(int(FC_SIZE_2/2), activation='relu'))
	final_model.add(Dense(int(FC_SIZE_2/4), activation='relu'))
	final_model.add(Dense(nb_classes, activation='softmax'))
	#base_model.summary()
	#model_conv.summary()
	#model_feat.summary()
	#final_model.summary()
	
	model = Model(input=[model_conv.input, model_feat.input], output=final_model.output)
	model.summary()
	return model

'''
method to disable the weight updation (training) for pretrained layers
'''
def transfer_learn(model, base_model):
	'''Train only the last layers '''
	for layer in base_model.layers:
		layer.trainable = False
	model.compile( optimizer=optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


'''
method to enable the weight updation for all layers
'''
def finetune_learn(model, base_model):
	for layer in base_model.layers:
		layer.trainable = True
	model.compile(optimizer=optimizers.Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])


'''
Added the custom callback class in order to save intermediate model in case good performance on validation set.
'''
class BestModel(keras.callbacks.Callback):
	#initialize all the variables
	def on_train_begin(self, logs={}):
		self.max_f1 = 0.0
		self.f1s = []
		self.flag = 0
	#methos to enable the validation after each 5 step of batch 
	def on_epoch_end(self, epoch, logs={}):
		ep_acc = logs.get('acc')
		if ep_acc >=0.41 :
			self.flag=1
		elif ep_acc >=0.80:
			 self.flag=0
		return	
	#method to validate the current model after 5 batch
	def on_batch_end(self, batch, logs={}):
		if self.flag ==1 and (int(batch)%5)==0:
			print ("validating...")
			p = model.predict([x_val, xf_val], batch_size=100)
			yt = np.argmax(y_val, axis=1)
			yp = np.argmax(p, axis=1)
			v_f1 = f1_score(yt, yp, average='weighted')
			print ("F1_SCORE",v_f1)
			self.f1s.append(v_f1)
			if self.max_f1 < v_f1:
				self.max_f1=v_f1
				if v_f1 >= 0.42:
					self.model.save(str(v_f1)+'mera_model_v8.h5')
		return

'''
adding the last classificaion layers in base model
'''
model = add_new_last_layer(base_model, nb_classes)
#model = multi_gpu_model(model, gpus=GPU)

batch_size = 48
epochs = 4
best_model = BestModel()

'''
helping method to generate the images  
'''
gen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True)
'''
method to generate teh flow for both inputs: images + additional features
'''
def gen_flow_for_two_inputs(XB, XFB, YB):
	genX1 = gen.flow(XB,YB,  batch_size=batch_size,seed=666)
	genX2 = gen.flow(XB,XFB, batch_size=batch_size,seed=666)
	while True:
		X1i = genX1.next()
		X2i = genX2.next()
		#Assert arrays are equal - this was for peace of mind, but slows down training
		#np.testing.assert_array_equal(X1i[0],X2i[0])
		#print (X1i[0].shape, X2i[1].shape, X1i[1].shape)
		yield [X1i[0], X2i[1]], X1i[1]

'''
generate the combined data for model
'''
gen_flow = gen_flow_for_two_inputs(x_train, xf_train, y_train)

#--------------------------------------------------------------------------------------------#

'''Transfer learning learns only newly added last layers of model'''
transfer_learn(model, base_model)

#print the model summary for debugging
model.summary()
#plot_model(model, to_file='model_plot_1.png', show_shapes=True, show_layer_names=True)

'''
history = model.fit(
	[x_train, xf_train], y_train,
	epochs=epochs,
	batch_size=batch_size)
'''
'''feed the training and validation data into model'''
history = model.fit_generator(
	gen_flow,
	steps_per_epoch=x_train.shape[0] // (batch_size),
	epochs=epochs,
	shuffle=True)
#	callbacks=[best_model])

'''Update the epoch value for fine tuning the model trained after transfer learning'''
epochs = 50

'''fine tuning learns all the layers of the model'''
finetune_learn(model, base_model)
model.summary()
#plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)
'''
history = model.fit(
        [x_train, xf_train], y_train,
	epochs=epochs,
	callbacks=[best_model])
'''
'''feed the training and validation data into model'''
history = model.fit_generator(
	gen_flow,
	steps_per_epoch=x_train.shape[0] // (batch_size),
	epochs=epochs,
	shuffle=True,
	callbacks=[best_model])

print (best_model.max_f1)
print ("Validation accuracy:\n", best_model.f1s)


