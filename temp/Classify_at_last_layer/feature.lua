-- Features
require 'sys'
require 'torch'
require 'cunn'
require 'nn' 
require 'cudnn'
matio = require 'matio'
require 'optim'
require 'cutorch'
require 'math'
py = require('fb.python')
im = require 'image'
cutorch.setDevice(1)
criterion = nn.MSECriterion()
Model = torch.load('Model.t7')
Mask = torch.load('Mask.t7')
complete = matio.load('data.mat')
temp = complete.data.x
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)
Inputs = torch.Tensor(temp:size(1),temp:size(2)+2,temp:size(3),temp:size(4))
temp = complete.data.y
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:mul(0.004)

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

stack_model = function(AE)
              AE_feat = nn.Sequential()
              for lay = 1,5 do 
                   AE_feat = AE_feat:add(AE:get(lay))
              end
              return AE_feat
end
Features = {}
Lable = {}

for i = 1,10 do
    Out = torch.zeros(#Model,inputs:size(3),inputs:size(4)):float()
    AE = Model[1]    
    AE_feat = stack_model(AE)
    AE = nil
    AE_feat:cuda()
    feat = AE_feat:forward(noise(inputs[i]):cuda()):float()
    Feat = feat:reshape(feat:size(1),feat:size(2),feat:size(3))
    AE_feat:float()
    collectgarbage()
    for t = 2,#Model do
        AE = Model[t]    
        AE_feat = stack_model(AE)
        AE = nil
        AE_feat:cuda()
        feat = AE_feat:forward(noise(inputs[i]):cuda()):float()
        feat = feat:reshape(feat:size(1),feat:size(2),feat:size(3))
        Feat = torch.cat(Feat,feat,1)
        AE_feat:float()
        collectgarbage()
   end
   table.insert(Features,Feat)
end

for i = 1,10 do
    Out = torch.zeros(#Model,inputs:size(3),inputs:size(4)):float()
    AE = Model[1]    
    AE:cuda()
    Out[{{1},{},{}}] = (AE:forward(noise(inputs[i]):cuda()):float()-targets[i]:float()):abs()
    AE:float()
    AE = nil
    collectgarbage()
    for t = 2,#Model do
        AE = Model[t]    
       AE:cuda()
       Out[{{t},{},{}}] = (AE:forward(noise(inputs[i]):cuda()):float()-targets[i]:float()):abs()
       AE:float()
        AE = nil
   end
   val, L = torch.min(Out:reshape(#Model,Out:size(2),Out:size(3)),1)
   for t = 1,#Model do
       Out[{{t},{},{}}] = torch.eq(L,t):float()
   end
   table.insert(Lable,Out)
end
Feat = Features[1]
L = Lable[1]
Feat = Feat:reshape(1,Feat:size(1),Feat:size(2),Feat:size(3))
L = L:reshape(1,L:size(1),L:size(2),L:size(3))
for temp = 2,#Features do
feat = Features[temp]
l = Lable[temp]
feat = feat:reshape(1,feat:size(1),feat:size(2),feat:size(3))
l = l:reshape(1,l:size(1),l:size(2),l:size(3))
Feat = torch.cat(Feat,feat,1)
L = torch.cat(L,l,1)
end
require 'hdf5'
myFile = hdf5.open('/home/mict/OCT_SR/Feat.h5', 'w')
myFile:write('/home/mict/OCT_SR', Feat)
myFile:close()
myFile = hdf5.open('/home/mict/OCT_SR/L.h5', 'w')
myFile:write('/home/mict/OCT_SR', L)
myFile:close()

