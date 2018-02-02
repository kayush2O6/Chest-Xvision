######################################
# Author: Ayush Kumar                #
# Email: kayush206@gmail.com         #
######################################

import numpy as np
import tensorflow as tf
print(tf.__version__)

class ChestCNN(object):
	def __init__(self):
		#each training image:[256, 256], features:[3] and labels:[14]
		self.x_input = tf.placeholder(tf.float32, shape=[None, 256, 256], name="input_x")
		self.y_input = tf.placeholder(tf.float32, shape=[None, 14], name="input_y")
		self.x_feature = tf.placeholder(tf.float32, shape=[None, 2], name="x_features")
		self.keep_prob = tf.placeholder(tf.float32)

		x_re = tf.reshape(self.x_input, [-1, 256, 256, 1])
		print ('input_x_reshaped : ', x_re.get_shape())

		#filter_size = 5x5, num_filters = 50

		#first conv layer
		W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 100], stddev=0.1))
		b_conv1 = tf.Variable(tf.constant(0.1, shape=[100]))

		h_conv1 = tf.nn.relu(tf.nn.conv2d(x_re, W_conv1, strides=[1, 1, 1, 1], padding='SAME'))
		h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		#second conv layer : size=7x7 num_filter = 50 
		W_conv2 = tf.Variable(tf.truncated_normal([7, 7, 100, 100], stddev=0.1))
		b_conv2 = tf.Variable(tf.constant(0.1, shape=[100]))

		h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME'))
		h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		#Third conv layer : size=5x5 num_filter = 64 
		W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 100, 64], stddev=0.1))
		b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]))

		h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME'))
		h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

		#Densaly connected layer with 1024 neurons
		W_fc1 = tf.Variable(tf.truncated_normal([(32*32*64)+2, 1024], stddev=0.1))
		b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
		h_pool_flat = tf.reshape(h_pool3, [-1, 32*32*64])
		h_pool_flat_cat = tf.concat([h_pool_flat, self.x_feature], 1)
		h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat_cat, W_fc1)+b_fc1)

		#To reduce the oerfitting DROPOUT is added
		h_fc1_do = tf.nn.dropout(h_fc1, self.keep_prob)

		#Second Densly connected layer
		W_fc2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
		b_fc2 = tf.Variable(tf.constant(0.1, shape=[512]))
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_do, W_fc2)+b_fc2)

		#To reduce the oerfitting DROPOUT is added
		h_fc2_do = tf.nn.dropout(h_fc2, self.keep_prob)

		#Readout layer
		W_fc3 = tf.Variable(tf.truncated_normal([512, 14], stddev=0.1))
		b_fc3 = tf.Variable(tf.constant(0.1, shape=[14]))
		self.y_conv = tf.matmul(h_fc2_do, W_fc3)+b_fc3	

		#Calculating the loss and backpropogation
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.y_conv))
		#regularizer = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)
		self.loss = tf.reduce_mean(loss)#+(0.01)* regularizer)

		#Calculating accuracy and prediction of model
		self.corr_predictions = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_input, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.corr_predictions, tf.float32))

