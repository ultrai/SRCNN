dofile ('SRCNN.lua') -- CNN architecture for super resoltuion (http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
dofile ('SRCNN_modified.lua') -- CNN with seperable filters 
--dofile ('SRCNN_modified2') -- http://lxu.me/projects/dcnn/  requires more number of parameters 
train1 = torch.load('train.t7') -- MSE on train set with CNN 
train2 = torch.load('train_modified.t7') -- MSE on train set with modified CNN 
--train3 = torch.load('train_modified2.t7')
test1 = torch.load('test.t7') -- MSE on test set and test set with CNN
test2 = torch.load('test_modified.t7')  -- MSE on test set and test set with modified CNN
--test3 = torch.load('test_modified2.t7')
torch.setdefaulttensortype('torch.FloatTensor')
train1_psnr = train1:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)  -- converting MSE to PSNR
train2_psnr = train2:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)
--train3_psnr = train3:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)
test1_psnr = test1:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)
test2_psnr = test2:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)
--test3_psnr = test3:clone():pow(-1):mul(255*255):log():div(torch.log(10)):mul(10)
t = torch.linspace(1,1250,1250):float() -- epochs
t = torch.cat(t,train1_psnr,2)          -- concatenation of all values  
t = torch.cat(t,train2_psnr,2)
--t = torch.cat(t,train3_psnr,2)
t = torch.cat(t,test1_psnr,2)
t = torch.cat(t,test2_psnr,2)
--t = torch.cat(t,test3_psnr,2)
py = require('fb.python')
py.exec([=[
import numpy as np
def foo(Data,target):
    a = np.asarray(Data)
    #kk=["foo_"+ str(target)+".csv"]
    np.savetxt(target, a, delimiter=",")
    done = 1
    return (done)
]=])
print( py.eval('foo(d,l)', {d = t,l="Data.csv"})) -- To create a CSV file
--[[
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/test.jpg" alt = "Test image" width="100" >
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_SRCNN.jpg" width="200">
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_Proposed.jpg" width="200">
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_truth.jpg" width="200">
a)Test image b)Fully connected CNN c)Modified CNN with 20% less parameters d)Anticipated super resolved image

![image](https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Data_plot.png )
PSNR profiles of CNN and modified CNN across training and testing datasets
]]--

--[[
qlua
require 'sys'
require 'torch'
require 'image'
require 'cunn'
require 'nn' 
require 'cudnn'
require 'cutorch'
gfx = require 'gfx.js'
CNN = torch.load('Model.t7')
--CNN = torch.load('Model_modified.t7')
parameters,gradParameters = CNN:getParameters()
parameters:size()
--itorch.image(CNN:get(2).weight)
ll = CNN:get(7).weight
pp = ll:clone()
pp = torch.reshape(pp,80,15,15)
torch.save('parameters.t7',pp)
pp = pp:double()
pp = torch.exp(pp:mul(15))
win_w1 = image.display{image=pp, zoom=4, nrow=10,
                                min=pp:min(), max=pp:max(),
                                win=win_w1, legend='stage 1: weights', padding=1}
image.save('weights2.jpg',win_w1.image)
]]--
