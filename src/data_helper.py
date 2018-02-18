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


def load_data(typ, img_size=256):
	if typ=="train":
		if os.path.exists("Xtrain.npy") and os.path.exists("XFtrain.npy") and os.path.exists("Ytrain.npy"):
    			return np.load("Xtrain.npy"), np.load("XFtrain.npy"), np.load("Ytrain.npy")
	elif typ=="test":
		if os.path.exists("Xtest.npy") and os.path.exists("XFtest.npy"):
			return np.load("Xtest.npy"), np.load("XFtest.npy")

	imgDir = '../data/'+typ+'_/'
	labels = pd.read_csv('../data/'+typ+'.csv')
	listImages = [l for l in labels['image_name']]
	posList = [int(l) for l in  labels['view_position']]

	X = np.array ([np.array (cv2.resize(cv2.imread (imgDir + img, cv2.IMREAD_COLOR), (img_size, img_size)), np.float32)/255. for img in listImages])
	if typ == "train":
		listClasses = [int(l.split('_')[1]) for l in labels['detected']]
		Y = []
		for cls in listClasses:
			temp = [0]*14
			temp[cls-1]=1
			Y.append(temp)
		Y = np.array(Y)

	age_raw=[]
	for ag in labels['age']:
		if ag>=100:
#			print (ag)
			ag = float(ag/12.)
		age_raw.append(ag)
	age = np.array(age_raw, np.float32)/100.
	gender_map = {'M':[0, 1],'F':[1, 0]}
	position_map = {0:[0, 1], 1:[1, 0]}
	gender = np.array([gender_map[str(g)] for g in labels['gender']], np.float32)
	position = np.array([position_map[int(l)] for l in labels['view_position']], np.float32)
	X_feature = np.column_stack((age, gender, position))
	if typ=="train":
		np.save("Xtrain.npy", X)
		np.save("XFtrain.npy", X_feature)
		np.save("Ytrain.npy", Y)
		return X, X_feature, Y
	elif typ=="test":
		np.save("Xtest.npy", X)
		np.save("XFtest.npy", X_feature)
		return X, X_feature

if __name__ == '__main__':
	X, XF, Y = load_data(typ="train")
	print (X.shape, XF.shape, Y.shape)
	print ("----")
	X, XF = load_data(typ="test")
	print (X.shape, XF.shape)		
	
