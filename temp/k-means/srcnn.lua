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
torch.setdefaulttensortype('torch.FloatTensor')

complete = matio.load('Data.mat')
temp = complete.LL
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

temp = complete.HH
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

complete = matio.load('Masks.mat')
temp = complete.Masks
temp = temp:type('torch.FloatTensor')
masks = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

complete = matio.load('Data_test.mat')
temp = complete.LL
temp = temp:type('torch.FloatTensor')
inputs2 = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

temp = complete.HH
temp = temp:type('torch.FloatTensor')
targets2 = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

complete = matio.load('Masks_test.mat')
temp = complete.Masks
temp = temp:type('torch.FloatTensor')
masks2 = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-coefL1',0,'L1 norm Coefficient')
cmd:option('-coefL2',0,'L2 norm Coefficient')
cmd:option('-P_L1',0,'L1 penality on activation')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-nfeat',150,'Number of filters to be considered')
cmd:option('-nfeat2',90,'Number of filters to be considered')
cmd:option('-feat_sz',13,'Each filter size')
cmd:option('-feat_sz2',11,'Each filter size')
cmd:option('-iterations',100,'total no of iterations')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

Model = {}
Mask = {}

criterion = nn.MSECriterion()
criterion:cuda()

Cost = 999999

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f2 = 0
        neval = neval + 1
        for i = 1,10 do
            output = AE:forward(inputs[i]:cuda())
            err = criterion:forward(torch.cmul(output,Mask1[i]:cuda()), torch.cmul(targets[i]:cuda(),Mask1[i]:cuda()))
            f2 = f2 + err
            df_do = criterion:backward(torch.cmul(output:cuda(),Mask1[i]:cuda()), torch.cmul(targets[i]:cuda(),Mask1[i]:cuda()))
            AE:backward(inputs[i]:cuda(), df_do)
 	    collectgarbage()
        end
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f2,  sys:toc(),gradParameters[1]))
      return f2,gradParameters
end

optimState = {maxIter = opt.iterations}
optimMethod = optim.cg

for t = 1,masks:size(1) do
    train = {}
    test = {}
    trainLogger = optim.Logger(('Results/train' .. t ..  '.log'))
    testLogger = optim.Logger(('Results/test' .. t ..  '.log'))
    if t==1 then Model = {} else Model = torch.load('Model.t7') end
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
    Mask1 = masks[t]
    parameters,gradParameters = AE:getParameters()
    sys:tic()
    neval = 0
    optimMethod(func, parameters, optimState)-- <------------------- optimization
    AE:evaluate()
    AE:float()
    Model[t] = AE:clone('weight', 'bias')
    --Mask[t] = Mask1:clone()
    torch.save('Model.t7',Model)
    Model = nil
    AE = nil
    collectgarbage()
    Mask[t+1] =  Mask2
    torch.save('train' .. t .. '.txt', train,'ascii')
    torch.save('test' .. t .. '.txt', test,'ascii')
    torch.save('train' .. t .. '.t7',train)
    torch.save('test' .. t .. '.t7',test)
    --torch.save('Mask.t7',Mask)
end

