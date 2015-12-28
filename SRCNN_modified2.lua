--Deep Convolutional Neural Network for Image Deconvolution  
--I might be wrong
--http://lxu.me/
require 'sys'
require 'torch'
require 'cunn'
require 'nn' 
require 'cudnn'
matio = require 'matio'
require 'optim'
require 'cutorch'
require 'math'
im = require 'image'
--cutorch.setDevice(2)
torch.setdefaulttensortype('torch.FloatTensor')

require 'hdf5'
myFile = hdf5.open('Data.h5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.x
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.y
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

myFile = hdf5.open('Data_test.h5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.x
temp = temp:type('torch.FloatTensor')
inputs_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.y
temp = temp:type('torch.FloatTensor')
targets_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-coefL1',0,'L1 norm Coefficient')
cmd:option('-coefL2',0,'L2 norm Coefficient')
cmd:option('-P_L1',0,'L1 penality on activation')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-nfeat',60,'Number of filters to be considered')
cmd:option('-nfeat2',60,'Number of filters to be considered')
cmd:option('-feat_sz',11,'Each filter size')
cmd:option('-feat_sz2',11,'Each filter size')
cmd:option('-iterations',1000,'total no of iterations')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

Model = {}

criterion = nn.MSECriterion()
criterion:cuda()


Cost = 999999

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        f_test = 0
        neval = neval + 1
        for i = 1,10 do
            output = AE:forward(inputs[i]:cuda())
            err = criterion:forward(output,targets[i]:cuda())
            f = f + err
            df_do = criterion:backward(output,targets[i]:cuda())
            AE:backward(inputs[i]:cuda(), df_do:cuda())
 	    collectgarbage()
        end
        table.insert(train,f/10)
        AE:evaluate()
        for i = 1,17 do
            output = AE:forward(inputs_test[i]:cuda())
            err = criterion:forward(output,targets_test[i]:cuda())
            f_test = f_test + err
            collectgarbage()
        end
        AE:training()
        table.insert(test,f_test/17)
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f/10,  sys:toc(),gradParameters[1]))
      return f/10,gradParameters/10
end

optimState = {maxIter = opt.iterations}
optimMethod = optim.cg

AE = nn.Sequential()
AE:add(nn.SpatialZeroPadding((opt.feat_sz-1)/2, (opt.feat_sz-1)/2,  0,0))
AE:add(cudnn.SpatialConvolution(1, opt.nfeat, opt.feat_sz,1))
AE:add(nn.SpatialZeroPadding(0,0,(opt.feat_sz-1)/2, (opt.feat_sz-1)/2))
AE:add(cudnn.SpatialConvolution( opt.nfeat, 2*opt.nfeat, 1,opt.feat_sz))
AE:add(cudnn.ReLU())
AE:add(cudnn.SpatialConvolution( 2*opt.nfeat, opt.nfeat2,1, 1))
AE:add(cudnn.ReLU())
AE:add(nn.SpatialZeroPadding((opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2))
AE:add(cudnn.SpatialConvolution(opt.nfeat2, 1,opt.feat_sz2, opt.feat_sz2))
AE:add(cudnn.ReLU())
AE:cuda()
AE:training()
parameters,gradParameters = AE:getParameters()
sys:tic()
train = {}
test = {}
--trainLogger = optim.Logger(('Results/train_normal.log'))
--testLogger = optim.Logger(('Results/test_normal.log'))
neval = 0
optimMethod(func, parameters, optimState)-- <------------------- optimization
AE:evaluate()
for i = 1,10 do
      output = AE:forward(inputs[i]:cuda())
      im.save('/home/mict/Desktop/OCT_SR/Results/Train_' .. i .. '_Xu.jpg', output:float():div(255))
end
for i = 1,17 do
      output = AE:forward(inputs_test[i]:cuda())
      im.save('/home/mict/Desktop/OCT_SR/Results/Test_' .. i .. '_Xu.jpg', output:float():div(255))
end
torch.save('Model_modified2.t7',AE)
torch.save('train_modified2.t7',torch.Tensor(train))
torch.save('test_modified2.t7',torch.Tensor(test))
AE = nil
collectgarbage()

