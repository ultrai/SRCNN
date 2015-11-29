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
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

complete = matio.load('data.mat')
temp = complete.data.x
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

temp = complete.data.y
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

complete = matio.load('data_test.mat')
temp = complete.data_test.x
temp = temp:type('torch.FloatTensor')
inputs2 = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

temp = complete.data_test.y
temp = temp:type('torch.FloatTensor')
targets2 = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-coefL1',0,'L1 norm Coefficient')
cmd:option('-coefL2',0,'L2 norm Coefficient')
cmd:option('-P_L1',0,'L1 penality on activation')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-nfeat',120,'Number of filters to be considered')
cmd:option('-nfeat2',80,'Number of filters to be considered')
cmd:option('-feat_sz',13,'Each filter size')
cmd:option('-feat_sz2',13,'Each filter size')
cmd:option('-iterations',250,'total no of iterations')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

Mask = {}
Mask[1] = torch.Tensor(targets:size(1),targets:size(2),targets:size(3),targets:size(4)):zero():add(1)
Model = {}

criterion = nn.MSECriterion()

criterion:cuda()


noise = function (Image)
	resample = Image[{{},{},{1}}]
	for Idx = 2,900 do
	if Idx%2~=0 then
	resample = torch.cat(resample,Image[{{},{},{Idx}}],3)
	end
	end
	resample = im.scale(resample,900,450)
	return resample
end	
Cost = 999999

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f2 = 0
        neval = neval + 1
        for i = 1,10 do
            output = AE:forward(noise(inputs[i]):cuda())
            err = criterion:forward(output:cuda(), targets[i]:cuda())
            f2 = f2 + err
            df_do = criterion:backward(output:cuda(), targets[i]:cuda())
            AE:backward(noise(inputs[i]):cuda(), df_do)
 	    collectgarbage()
        end
	print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f2,  sys:toc(),gradParameters[1]))
      --return f2,gradParameters
f1 = 0
        for i = 1,10 do
            output = AE:forward(noise(inputs[i]):cuda())
            err = criterion:forward(output, targets[i]:cuda())
            f1 = f1 + err
        end
        table.insert(train,f1/10)
        f1 = 0
        for i = 1,18 do
            output = AE:forward(noise(inputs2[i]):cuda())
            err = criterion:forward(output, targets2[i]:cuda())
            f1 = f1 + err
        end
        table.insert(test,f1/18)
end

func2 = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f2 = 0
        neval = neval + 1
        for i = 1,10 do
            output = AE:forward(noise(inputs[i]):cuda())
            err = criterion:forward(torch.cmul(output,Mask1[i]:cuda()), torch.cmul(targets[i]:cuda(),Mask1[i]:cuda()))
            f2 = f2 + err
            df_do = criterion:backward(torch.cmul(output:cuda(),Mask1[i]:cuda()), torch.cmul(targets[i]:cuda(),Mask1[i]:cuda()))
            AE:backward(noise(inputs[i]):cuda(), df_do)
 	    collectgarbage()
        end
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f2,  sys:toc(),gradParameters[1]))
      return f2,gradParameters
end

optimState = {maxIter = opt.iterations}
optimMethod = optim.cg

for t = 1,1 do
    train = {}
    test = {}
    trainLogger = optim.Logger(('Results/train' .. t ..  '.log'))
    testLogger = optim.Logger(('Results/test' .. t ..  '.log'))
    Model = torch.load('Model.t7')
    if t==1 then Model = {} end
    AE = nn.Sequential()
    AE:add(nn.SpatialZeroPadding((opt.feat_sz-1)/2, (opt.feat_sz-1)/2, (opt.feat_sz-1)/2, (opt.feat_sz-1)/2))
    AE:add(cudnn.SpatialConvolution(1, opt.nfeat, opt.feat_sz, opt.feat_sz))
    AE:add(cudnn.ReLU())
    AE:add(cudnn.SpatialConvolution( opt.nfeat, opt.nfeat2,1, 1))
    AE:add(cudnn.ReLU())
    AE:add(nn.SpatialZeroPadding((opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2, (opt.feat_sz2-1)/2))
    AE:add(cudnn.SpatialConvolution(opt.nfeat2, 1,opt.feat_sz2, opt.feat_sz2))
    AE:add(cudnn.ReLU())
    AE:cuda()
    AE:training()
    Mask1 = Mask[t]
    parameters,gradParameters = AE:getParameters()
    sys:tic()
    neval = 0
    optimMethod(func2, parameters, optimState)-- <------------------- optimization
    AE:evaluate()
    Mask2 = torch.Tensor(targets:size(1),targets:size(2),targets:size(3),targets:size(4)):zero()
    for i = 1,10 do
        Mask2[{{i},{},{},{}}] = torch.cmul(torch.gt((AE:forward((noise(inputs[i])):cuda()):float()-(targets[i]):float()):abs(),5),Mask1[{{i},{},{},{}}]:byte())
    end
    AE:float()
    Model[t] = AE:clone('weight', 'bias')
    torch.save('Model.t7',Model)
    Model = nil
    AE = nil
    collectgarbage()
    Mask[t+1] =  Mask2
    torch.save('train' .. t .. '.txt', train,'ascii')
    torch.save('test' .. t .. '.txt', test,'ascii')
    torch.save('train' .. t .. '.t7',train)
    torch.save('test' .. t .. '.t7',test)
    torch.save('Mask.t7',Mask)
end

------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
-- Testing
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
cutorch.setDevice(2)

Model = torch.load('Model.t7')


noise = function (Image)
	resample = Image[{{},{},{1}}]
	for Idx = 2,900 do
	if Idx%2~=0 then
	resample = torch.cat(resample,Image[{{},{},{Idx}}],3)
	end
	end
	resample = im.scale(resample,900,450)
	return resample
end
criterion = nn.MSECriterion()


complete = matio.load('data.mat')
temp = complete.data.x
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)
Inputs = torch.Tensor(temp:size(1),temp:size(2)+2,temp:size(3),temp:size(4))

temp = complete.data.y
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

for i = 1,10 do
    Out = torch.Tensor(targets:size(2),targets:size(3),targets:size(4)):zero()
    for t = 1,#Model do
        AE = Model[t]
        AE:cuda()
        O1 = AE:forward(noise(inputs[i]):cuda()):float()
        AE:float()
       -- AE = nil
        collectgarbage()
        if t==1 then print( criterion:float():forward(O1,targets[i]:float()) .. " O1") end
        Out = Out:float()+O1
     end
     print( criterion:float():forward(Out:div(#Model),targets[i]:float()) .. " O")
end
 
