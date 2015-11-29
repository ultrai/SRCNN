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

py.exec([=[
import scipy.misc  
import scipy.io as sio
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
def foo(dir):
    type = ".tif"
    path1 = dir + "Images for Dictionaries and Mapping leraning/"
    path2 = dir + "For synthetic experiments/"
    W=16
    window_shape = (W, W)
    for x in range(1, 8):#<---- as indexing images start from 1
        print(x)
        im = np.array(Image.open(path1 + "LL" + str(x) + type)) 
        #im[:,range(1,im.shape[1],2)] = 0
        im2 = im[:,range(0,im.shape[1],2)]   #<-removal of intermediate A-scans
        im = scipy.misc.imresize(im2,np.shape(im)) #<-----  Interpolation to High resolution
        hh = np.array(Image.open(path1 + "HH" + str(x) + type)) 
        if x==1:
           Feat = image.extract_patches_2d(im, window_shape)
           Feat2 = image.extract_patches_2d(hh, window_shape)
           LL = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
           HH = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
        else:
           feat = image.extract_patches_2d(im, window_shape)
           feat2 = image.extract_patches_2d(hh, window_shape)
           Feat = np.concatenate((Feat, feat), axis=0)
           Feat2 = np.concatenate((Feat2, feat2), axis=0)
           ll = np.ndarray.reshape(im,(1,1,im.shape[0],im.shape[1]))
           hh = np.ndarray.reshape(hh,(1,1,hh.shape[0],hh.shape[1]))
           LL = np.concatenate((LL, ll), axis=0)
           HH = np.concatenate((HH, hh), axis=0)
    #Feat = Feat[:,:,range(0,W,2)]
    Feat = Feat[::W,:,:]
    Feat2 = Feat2[::W,:,:]
    print('Data Prepared')
    #sio.savemat(dir+"Data.mat", {'Feat':Feat,'Feat2':Feat2})
    import h5py
    h5f = h5py.File("Data.h5", 'w')
    h5f.create_dataset('Feat', data=Feat)
    h5f.create_dataset('Feat2', data=Feat2)
    h5f.close()
    print('Data saved')
    done = 1
    return (done)
]=])



print( py.eval('foo(d)', {d = "/home/mict/OCT_SR_MLP/"}) )



require 'hdf5'
myFile = hdf5.open('Data.h5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.Feat
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp)
inputs = inputs:reshape(inputs:size(1),inputs:size(2)*inputs:size(3))

temp=Temp.Feat2
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp)
targets = targets:reshape(targets:size(1),targets:size(2)*targets:size(3))

--[[
complete = matio.load('Data.mat')
temp = complete.Feat
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp)
temp = complete.Feat2
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3)):copy(temp)
inputs = inputs:reshape(inputs:size(1),inputs:size(2)*inputs:size(3))
targets = targets:reshape(targets:size(1),targets:size(2)*targets:size(3))
]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-coefL1',0.0,'L1 norm Coefficient')
cmd:option('-coefL2',0.0,'L2 norm Coefficient')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-h1',3000,'Number of filters to be considered')
cmd:option('-h2',3000,'Number of filters to be considered')
cmd:option('-drop',0.0,'Dropout')
cmd:option('-batch',1e4,'Number of filters to be considered')
cmd:option('-feat_sz',13,'Each filter size')
cmd:option('-feat_sz2',13,'Each filter size')
cmd:option('-iterations',110,'total no of iterations')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)
criterion = nn.MSECriterion()
Cost = 999999

func1 = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        neval = neval + 1
        count=0
        for i = 1,inputs:size(1),opt.batch do
          --print(count)
          count = count+1
            if i+opt.batch<inputs:size(1) then
                input = Inputs[{{i,i+opt.batch},{}}]:float()
                target = Inputs[{{i,i+opt.batch},{}}]:float()
            else       
                input = Inputs[{{i,inputs:size(1)},{}}]:float()
                target = Inputs[{{i,targets:size(1)},{}}]:float()
           end
            output = ae:forward(input)
            err = criterion:forward(output,target)
            f = f + err
            df_do = criterion:backward(output,target)
            ae:backward(input, df_do)
 	          collectgarbage()
        end
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f,  sys:toc(),gradParameters[1]))
        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
          norm,sign= torch.norm,torch.sign
          -- Loss:
          f = f + opt.coefL1 * norm(parameters,1)
          f = f + opt.coefL2 * norm(parameters,2)^2/2
           -- Gradients:
          gradParameters:add( sign(parameters):mul(opt.coefL1) +        parameters:clone():mul(opt.coefL2) )
        end
        --torch.save('Model.t7',AE)
        --dofile('srcnn_MLP_test.lua')
        --AE:training()
        return f/count,gradParameters:div(count)
end

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        neval = neval + 1
        count=0
        for i = 1,inputs:size(1),opt.batch do
          --print(count)
          count = count+1
            if i+opt.batch<inputs:size(1) then
                input = inputs[{{i,i+opt.batch},{}}]:float()
                target = targets[{{i,i+opt.batch},{}}]:float()
            else       
                input = inputs[{{i,inputs:size(1)},{}}]:float()
                target = targets[{{i,targets:size(1)},{}}]:float()
           end
            output = AE:forward(input)
            err = criterion:forward(output,target)
            f = f + err
            df_do = criterion:backward(output,target)
            AE:backward(input, df_do)
 	          collectgarbage()
        end
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f,  sys:toc(),gradParameters[1]))
        if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
          norm,sign= torch.norm,torch.sign
          -- Loss:
          f = f + opt.coefL1 * norm(parameters,1)
          f = f + opt.coefL2 * norm(parameters,2)^2/2
           -- Gradients:
          gradParameters:add( sign(parameters):mul(opt.coefL1) +        parameters:clone():mul(opt.coefL2) )
        end
        --torch.save('Model.t7',AE)
        --dofile('srcnn_MLP_test.lua')
        --AE:training()
        return f/count,gradParameters:div(count)
end
----------------------------------------------------------------------
Inputs = inputs:clone()

encoder1 = nn.Sequential()
encoder1:add(nn.Linear(Inputs:size(2),opt.h1))
encoder1:add(nn.ReLU())
encoder1:add(nn.L1Penalty(1e-3,true))
encoder1:add(nn.Linear(opt.h1, Inputs:size(2)))
ae = encoder1:clone()
parameters,gradParameters = ae:getParameters()
sys:tic()
neval = 0

optimState = {maxIter = opt.iterations}
optimMethod = optim.cg--adam --rmsprop --lbfgs
optimMethod(func1, parameters, optimState)--, optimState)-- 
encoder1 = ae:clone()

-----------------------------------------------------------------
Inputs = targets:clone()

encoder2 = nn.Sequential()
encoder2:add(nn.Linear(Inputs:size(2),opt.h1))
encoder2:add(nn.ReLU())
encoder2:add(nn.L1Penalty(1e-3,true))
encoder2:add(nn.Linear(opt.h1, Inputs:size(2)))
ae = encoder2:clone()
parameters,gradParameters = ae:getParameters()
sys:tic()
neval = 0

optimState = {maxIter = opt.iterations}
optimMethod = optim.cg--adam --rmsprop --lbfgs
optimMethod(func1, parameters, optimState)--, optimState)-- 
encoder2 = ae:clone()
-----------------------------------------------------------------

--[[AE = nn.Sequential()
AE:add(nn.Linear(inputs:size(2),opt.h1))
AE:add(nn.ReLU())
AE:add(nn.Dropout(opt.drop))
AE:add(nn.Linear(opt.h1,opt.h2))
AE:add(nn.ReLU())
AE:add(nn.Dropout(opt.drop))
AE:add(nn.Linear(opt.h2,targets:size(2)))
AE:add(nn.ReLU())
AE:training()
parameters,gradParameters = AE:getParameters()
sys:tic()
neval = 0
optimState = {maxIter = opt.iterations}
optimMethod = optim.cg--adam --rmsprop --lbfgs
--for lol = 1,100 do
AE:training()
optimMethod(func, parameters, optimState)--, optimState)-- <------------------- optimization
--end
]]--
model = AE:clone('weight','bias')
AE:evaluate()
torch.save('Model.t7',model)
AE = nil
dofile('srcnn_MLP_test.lua')
