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
Out = torch.Tensor(targets:size(1),#Model,targets:size(3),targets:size(4)):zero()

for i = 1,inputs:size(1) do
    for t = 1,#Model do
        AE = Model[t]
        AE:cuda()
        Out[{{i},{t},{},{}}] = AE:forward(noise(inputs[i]):cuda()):float()
        AE:float()
        AE = nil
        collectgarbage()
     end
end


Out_final2 = Out:sum(2):div(#Model)
print(criterion:forward(Out[{{},{1},{},{}}],targets))
print(criterion:forward(Out_final2,targets))

--[[
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
 ]]--
