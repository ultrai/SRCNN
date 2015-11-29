
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

inputs_test = inputs2
targets_test = targets2
masks_test = masks
--inputs_test = inputs2
--targets_test = targets2
--masks_test = masks2

criterion = nn.MSECriterion()
criterion:float()
Out = torch.Tensor(#Model,inputs_test:size(1),inputs_test:size(3),inputs_test:size(4)):zero()
    
for t = 1,#Model do
    AE = Model[t]
    AE:cuda()
    --Mask1 = (masks_test[t]):float()
    for i = 1,inputs_test:size(1) do
        O1 = AE:forward(inputs_test[i]:cuda()):float()
        Out[{{t},{i},{},{}}] = O1 --torch.cmul(O1,Mask1[i])--
    end
     AE:float()
     AE = nil
     collectgarbage()
end
--Out = torch.cmul(Out,masks_test):sum(1)
Out = Out:mul(1/#Model):sum(1)
for i = 1,inputs:size(1) do
    e = criterion:forward(Out[{{},{i},{},{}}],targets_test[i])
    print(e)
end

e = criterion:forward(Out,targets_test)
print(e)


