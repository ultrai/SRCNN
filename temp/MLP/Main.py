import scipy.misc 
from scipy.spatial.distance import cdist
import pickle 
import os
import time
import scipy.io as sio
import numpy as np
from skimage.util.shape import view_as_windows
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction import image
dir = "/home/mict/OCT_SR_MLP/"
type = ".tif"
path1 = dir + "Images for Dictionaries and Mapping leraning/"
path2 = dir + "For synthetic experiments/"
W=8
window_shape = (W, W)

x=1#<---- as indexing images start from 1
im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
#im[:,range(1,im.shape[1],2)] = 0
im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
#im = np.lib.pad(im, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
#im2 = np.lib.pad(hh, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='con


stant', constant_values=0)
#I1 = view_as_windows(im, window_shape)
#I2 = view_as_windows(hh, window_shape)
#I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
#I2 = np.ndarray.reshape(I2,(I2.shape[0],I2.shape[1],I2.shape[2]*I2.shape[3]))
#Feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2],I1.shape[3]))#Feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
#Feat2 = np.ndarray.reshape(I2,(I2.shape[0]*I2.shape[1],I2.shape[2],I2.shape[3]))#Feat2 = np.ndarray.reshape(I2,(I2.shape[0]*I2.shape[1],I2.shape[2]))
Feat = image.extract_patches_2d(im, window_shape)
Feat2 = image.extract_patches_2d(hh, window_shape)
LL = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
HH = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
for x in range(2, 11):
    print(x)
    im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
    hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
    #im[:,range(1,im.shape[1],2)] = 0
    im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
    im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
    #im = np.lib.pad(im, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
    #im2 = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
    #I1 = view_as_windows(im, window_shape)
    #I2 = view_as_windows(hh, window_shape)
    #I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
    #I2 = np.ndarray.reshape(I2,(I2.shape[0],I2.shape[1],I2.shape[2]*I2.shape[3]))
    #feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2],I1.shape[3]))#feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
    #feat2 = np.ndarray.reshape(I2,(I2.shape[0]*I2.shape[1],I2.shape[2],I2.shape[3]))#feat2 = np.ndarray.reshape(I2,(I2.shape[0]*I2.shape[1],I2.shape[2]))
    feat = image.extract_patches_2d(im, window_shape)
    feat2 = image.extract_patches_2d(hh, window_shape)
    Feat = np.concatenate((Feat, feat), axis=0)
    Feat2 = np.concatenate((Feat2, feat2), axis=0)
    ll = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
    hh = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
    LL = np.concatenate((LL, ll), axis=0)
    HH = np.concatenate((HH, hh), axis=0)

print('Data Prepared')
"""
sio.savemat(dir+"Data.mat", {'Feat':Feat,'Feat2':Feat2})
import h5py
h5f = h5py.File('Data.h5', 'w')
h5f.create_dataset('Feat', data=Feat)
h5f.create_dataset('Feat2', data=Feat2)
h5f.close()
print('Data saved')
#
#I = image.reconstruct_from_patches_2d(patches, (width, height )))
"""
X = Feat.reshape((Feat.shape[0],Feat.shape[1]*Feat.shape[2])) +1
Y = Feat2.reshape((Feat2.shape[0],Feat2.shape[1]*Feat2.shape[2]))+1
X = X.astype("float32")
Y = Y.astype("float32")
X /= 255
Y /= 255
np.random.seed(123)  # for reproducibility

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#from __future__ import absolute_import
#from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adagrad, Adam, RMSprop


batch_size = 100000
nb_epoch = 100

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(W*W, 1500))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1500, 1500))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1500, W*W))
model.add(Activation('sigmoid'))

opt = RMSprop()
model.compile(loss='MSE', optimizer=opt)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

for x in range(1, 11):
    print(x)
    im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
    target = np.array(Image.open(path1 + "HH" + str(x) + type)) 
    #im[:,range(1,im.shape[1],2)] = 0
    im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
    im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
    feat = image.extract_patches_2d(im, window_shape)
    hh = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
    feat = feat.reshape((feat.shape[0],feat.shape[1]*feat.shape[2])) +1
    feat.astype("float32")
    feat /= 255
    patch_est = model.predict(feat,batch_size=batch_size) 
    patch_est *= 255
    patch_est -= 1
    estimate = image.reconstruct_from_patches_2d(patch_est-1, (900, 450 ))
    err = (estimate-target).pow(2).mean()
    print(err)

    