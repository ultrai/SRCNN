import scipy.misc 
from scipy.spatial.distance import cdist
import pickle 
import os
import time
import scipy.io as sio
import numpy as np
from skimage.util.shape import view_as_windows
from PIL import Image
from sklearn import preprocessing
from sklearn.cluster import KMeans
dir = "/home/mict/OCT_SR_Kmeans/"
type = ".tif"
path1 = dir + "Images for Dictionaries and Mapping leraning/"
path2 = dir + "For synthetic experiments/"
W=9
window_shape = (W, W)

x=1
im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
im2 = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
im = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
I1 = view_as_windows(im, window_shape)
I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
Feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
LL = np.ndarray.reshape(im2,(1,1,im2.shape[0],im2.shape[1]))
HH = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
for x in range(2, 11):
    im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
    hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
    im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
    im2 = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
    im = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
    I1 = view_as_windows(im, window_shape)
    I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
    feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
    Feat = np.concatenate((Feat, feat), axis=0)
    ll = np.ndarray.reshape(im2,(1,1,im2.shape[0],im2.shape[1]))
    hh = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
    LL = np.concatenate((LL, ll), axis=0)
    HH = np.concatenate((HH, hh), axis=0)

print('Data Prepared')
scaler = preprocessing.StandardScaler().fit(Feat)
X = scaler.transform(Feat)
tic = time.clock()
clusters = 10
est = KMeans(init='k-means++', n_clusters=clusters, n_init=10,max_iter=100,n_jobs=-1)
est.fit(X)
print('Data clustered')
toc = time.clock()
print(toc-tic)
#pickle.dump( est, open( "save.p", "wb" ) )
#pickle.dump( toc-tic, open( "t.p", "wb" ) )
cen = est.cluster_centers_

D = cdist(X,cen)
L = np.argmin(D,axis=1)

masks = np.ndarray.reshape(L,(1,10,450,900))
Masks = masks==0
for Idx in range(1,clusters):
    masks_temp = masks==Idx
    Masks = np.concatenate((Masks, masks_temp), axis=0)
sio.savemat(dir+"Masks.mat", {'Masks':Masks})
sio.savemat(dir+"Data.mat", {'LL':LL,'HH':HH})

print('Data train Masks')

x=1
im = np.array(Image.open(path2 + str(x) + "/test" + type)) 
hh = np.array(Image.open(path2 + str(x) + "/average" + type)) 
im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
im2 = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
im = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
I1 = view_as_windows(im, window_shape)
I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
Feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
LL = np.ndarray.reshape(im2,(1,1,im2.shape[0],im2.shape[1]))
HH = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
for x in range(2, 19):
    print(x)
    if x!=9: # <----- 9th folder image is 401X900
        im = np.array(Image.open(path2 + str(x) + "/test" + type)) 
        hh = np.array(Image.open(path2 + str(x) + "/average" + type)) 
        im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
        im2 = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
        im = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
        I1 = view_as_windows(im, window_shape)
        I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
        feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
        Feat = np.concatenate((Feat, feat), axis=0)
        ll = np.ndarray.reshape(im2,(1,1,im2.shape[0],im2.shape[1]))
        hh = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
        LL = np.concatenate((LL, ll), axis=0)
        print(LL.shape[0])
        HH = np.concatenate((HH, hh), axis=0)

print('Prepared Test Data')

X = scaler.transform(Feat)

D = cdist(X,cen)
L = np.argmin(D,axis=1)

masks = np.ndarray.reshape(L,(1,17,450,900))
Masks = masks==0
for Idx in range(1,clusters):
    masks_temp = masks==Idx
    Masks = np.concatenate((Masks, masks_temp), axis=0)

sio.savemat(dir+"Masks_test.mat", {'Masks':Masks})
sio.savemat(dir+"Data_test.mat", {'LL':LL,'HH':HH})

x=9
im = np.array(Image.open(path2 + str(x) + "/test" + type)) 
hh = np.array(Image.open(path2 + str(x) + "/average" + type)) 
im2 = im[:,range(0,im.shape[1],2)]   #<------------ removal of intermediate A-scans
im2 = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution size through Spline
im = np.lib.pad(im2, (((W-1)*0.5, (W-1)*0.5), ((W-1)*0.5, (W-1)*0.5)), mode='constant', constant_values=0)
I1 = view_as_windows(im, window_shape)
I1 = np.ndarray.reshape(I1,(I1.shape[0],I1.shape[1],I1.shape[2]*I1.shape[3]))
Feat = np.ndarray.reshape(I1,(I1.shape[0]*I1.shape[1],I1.shape[2]))
LL = np.ndarray.reshape(im2,(1,1,im2.shape[0],im2.shape[1]))
HH = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
X = scaler.transform(Feat)

D = cdist(X,cen)
L = np.argmin(D,axis=1)

masks = np.ndarray.reshape(L,(1,1,401,900))
Masks = masks==0
for Idx in range(1,clusters):
    masks_temp = masks==Idx
    Masks = np.concatenate((Masks, masks_temp), axis=0)


sio.savemat(dir+"Masks_test2.mat", {'Masks':Masks})
sio.savemat(dir+"Data_test2.mat", {'LL':LL,'HH':HH})
