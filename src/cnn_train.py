######################################
# Author: Ayush Kumar                #
# Email: kayush206@gmail.com         #
######################################

from pylab import *
import cv2
import os
import time
import numpy as np
import pandas as pd
from skimage import exposure
import tensorflow as tf
from tensorflow.contrib import learn
from chest_cnn import ChestCNN
from data_helper import load_data
print ("Libraries loaded successfully")

def batch_gen(data, batch_size):
    num_batches = int((len(data)-1)/batch_size) + 1
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        yield data[start_index:end_index]
#-----------------
#loading the saved numpy training arrays
X, XF, Y = load_data(typ="train")
print(X.shape, Y.shape, XF.shape)

#Divide the data into train and validation split
dev_sample_percentage=0.25
np.random.seed(1117)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
xf_shuffled = XF[shuffle_indices]
y_shuffled = Y[shuffle_indices]

dev_sample_index = -1 * int(dev_sample_percentage * float(len(Y)))
x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
xf_train, xf_val = xf_shuffled[:dev_sample_index], xf_shuffled[dev_sample_index:]
y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print(x_train.shape, xf_train.shape, y_train.shape)
print(x_val.shape, xf_val.shape, y_val.shape)
del X, Y, XF, xf_shuffled, x_shuffled, y_shuffled

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement = True,
		log_device_placement = False)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = ChestCNN()
	
		global_step = tf.Variable(0, name='global_step', trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-5)
		grads_n_vars = optimizer.compute_gradients(cnn.loss)
		train_step = optimizer.apply_gradients(grads_n_vars, global_step=global_step)
		
		sess.run(tf.global_variables_intializer())
		
		max_acc=0
		#batches = batch_gen(list(zip(x_train, xf_train, y_train)), 64)
		#val_batches = batch_gen(list(zip(x_val, xf_val, y_val)), 64)
		for epoch in range(40):
			batches = batch_gen(list(zip(x_train, xf_train, y_train)), 80)
			i=0
			for batch in batches:
				x_batch, xf_batch, y_batch = zip(*batch)
				'''a single training step'''
				feed_dict = {
					cnn.x_input:x_batch,
					cnn.x_feature:xf_batch,
					cnn.y_input:y_batch,
					cnn.keep_prob:0.6
				}
				_, step, tr_loss, tr_acc = sess.run([train_step, global_step, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
				if tr_acc>=0.37:
					print ("Training Acc: ", tr_acc)
					val_batches = batch_gen(list(zip(x_val, xf_val, y_val)), 150)
					pred = []
					ydev = []
					accu = []
					for vb in val_batches:
						xv, xfv, yv = zip(*vb)
						'''evaluate model on dev set'''
						feed_dict = {
							cnn.x_input:xv,
							cnn.x_feature:xfv,
							cnn.y_input:yv,
							cnn.keep_prob:1
						}
						val_acc, val_pred, val_loss = sess.run([cnn.accuracy, cnn.y_conv, cnn.loss], feed_dict=feed_dict)
						accu.append(val_acc)
						pred.extend(val_pred)
						ydev.extend(yv)
						#print(np.array(ydev).shape)
					y_true = np.argmax(np.array(ydev), axis=1)
					accuracy_val = np.mean(accu)
					if max_acc < accuracy_val:
						max_acc=accuracy_val
						print (max_acc)
			i=i+1
		print("max_val_Acc {:g}".format(max_acc))
		print('Done!')
