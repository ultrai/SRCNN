import scipy.misc  
import scipy.io as sio
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
type = ".tif"
path1 = "Images for Dictionaries and Mapping leraning/"
path2 = "For synthetic experiments/"
for x in range(1, 11):#<---- as indexing images start from 1
    #print(x)
    im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
    im2 = im[:,range(0,im.shape[1],2)]   #<-removal of intermediate A-scans
    im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution
    hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
    if x==1:
       LL = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
       HH = np.ndarray.reshape(hh,(1,hh.shape[0],hh.shape[1]))
    else:
      ll = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
      hh = np.ndarray.reshape(hh,(1,hh.shape[0],hh.shape[1]))
      LL = np.concatenate((LL, ll), axis=0)
      HH = np.concatenate((HH, hh), axis=0)
print('Train Data Prepared')
import h5py
LL = LL.astype(('float32'))
HH = HH.astype(('float32'))

h5f = h5py.File("Data.h5", 'w')
h5f.create_dataset('x', data=LL)
h5f.create_dataset('y', data=HH)
h5f.close()

print('Train Data saved')
for x in range(1, 19):
    if x!=9:
       im = np.array(Image.open(path2 +  str(x) +"/test"+type))
       hh = np.array(Image.open(path2 +  str(x) +"/average"+type)) 
       im2 = im[:,range(0,im.shape[1],2)]   #<-removal of intermediate A-scans
       ll = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution
       if x==1:
          LL = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
          HH = np.ndarray.reshape(hh,(1,hh.shape[0],hh.shape[1]))
       else:
          ll = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
          hh = np.ndarray.reshape(hh,(1,hh.shape[0],hh.shape[1]))
          LL = np.concatenate((LL, ll), axis=0)
          HH = np.concatenate((HH, hh), axis=0)
print('Test Data Prepared')

LL = LL.astype(('float32'))
HH = HH.astype(('float32'))
h5f = h5py.File("Data_test.h5", 'w')
h5f.create_dataset('x', data=LL)
h5f.create_dataset('y', data=HH)
h5f.close()
print('Test Data saved')

