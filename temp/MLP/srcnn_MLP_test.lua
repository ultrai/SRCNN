-----------------------------------------------------------------------

require 'sys'
require 'torch'
--require 'cunn'
require 'nn' 
--require 'cudnn'
matio = require 'matio'
require 'optim'
--require 'cutorch'
require 'math'
im = require 'image'
py = require('fb.python')
--cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

local AE = torch.load('Model.t7')
AE:evaluate()
local W = 16
py.exec([=[
import scipy.misc  
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
def foo2(dir,x,W):
    x = np.array(x)
    x = x.astype("int") 
    W = np.array(W)
    W = W.astype("int") 
    type = ".tif"
    path1 = dir + "Images for Dictionaries and Mapping leraning/"
    path2 = dir + "For synthetic experiments/"
    window_shape = (W, W)
    im = np.array(Image.open(path2 +  str(x) +"/test"+type))
    hh = np.array(Image.open(path2 +  str(x) +"/average"+type)) 
    im = np.array(Image.open(path1 + "LL" + str(x) + type))
    hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
    #im[:,range(1,im.shape[1],2)] = 0
    im2 = im[:,range(0,im.shape[1],2)]   #<-removal of intermediate A-scans
    im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution
    Feat = image.extract_patches_2d(im, window_shape)
    return (Feat,hh)
]=])
py.exec([=[
import numpy as np
from sklearn.feature_extraction import image
def foo3(patches):
    #patches = np.array(patches)
    #patches.astype("uint8")
    I = image.reconstruct_from_patches_2d(patches, (450, 900))
    return (I)
]=])

criterion = nn.MSECriterion()
local batch= 100000
for i = 1,1 do
  local temp = py.eval('foo2(d,l,w)', {d = "/home/mict/OCT_SR_MLP/",l = i,w=W})
  local input = temp[1]:clone()
  input = input:reshape(input:size(1),input:size(2)*input:size(3))
  local target = temp[2]:clone()
  local Out = torch.zeros(input:size(1),input:size(2))
  for Idx = 1,input:size(1),batch do
        print(Idx)
        if Idx+batch<input:size(1) then
           Out[{{Idx,Idx+batch},{}}] = AE:forward(input[{{Idx,Idx+batch},{}}]:float())
        else
          Out[{{Idx,input:size(1)},{}}] = AE:forward(input[{{Idx,input:size(1)},{}}]:float())
        end
  end
  Out = Out:reshape(Out:size(1),W,W)
  local estimated = py.eval('foo3(p)', {p =Out})
  err = criterion:forward(estimated:float(),target:float())
  print(err)
  temp_save = estimated:clone()
  im.save('temp.jpg',temp_save:div(255))
end
