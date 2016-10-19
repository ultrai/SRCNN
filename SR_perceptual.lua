require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cudnn'
require 'cunn'
require 'loadcaffe'
require 'cutorch'
require 'nngraph'
dofile('f.lua')
cutorch.setDevice(1)      
torch.manualSeed(27)

torch.setdefaulttensortype('torch.FloatTensor')

require 'hdf5'
myFile = hdf5.open('Data.hdf5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.data
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.label
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),1,temp:size(2),temp:size(3)):copy(temp)

myFile = hdf5.open('Data_test.hdf5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.data
temp = temp:type('torch.FloatTensor')
inputs_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)
temp=Temp.label
temp = temp:type('torch.FloatTensor')
targets_test = torch.Tensor(temp:size(1),1,temp:size(2),temp:size(3)):copy(temp)

temp = nil

torch.manualSeed(27)


criterion = nn.MSECriterion()
criterion:cuda()
cnn = torch.load('VGG_19.t7')-- loadcaffe.load(params.proto_file, params.model_file, cudnn):float()
cnn:cuda()
  
kk=nn.DepthConcat(2)
kk:add(nn.Identity()):add(nn.Identity()):add(nn.Identity())
kk:cuda()
kk2=nn.DepthConcat(2)
kk2:add(nn.Identity()):add(nn.Identity()):add(nn.Identity())
kk2:cuda()
kk3=nn.DepthConcat(2)
kk3:add(nn.Identity()):add(nn.Identity()):add(nn.Identity())
kk3:cuda()
content_layers =  'conv1_1'
  
next_content_idx, next_style_idx = 1, 1
tv_mod = nn.TVLoss(1e-3):float()
tv_mod:cuda()
net = nn.Sequential() 
--net:add(tv_mod)
--[[for i = 1, #cnn.modules do
 -- print(i)
    if next_content_idx <= 1 then -- next_style_idx <= #style_layers then
      layer = cnn:get(i)
       name = layer.name
       layer_type = torch.type(layer)
       print(layer_type)
       if layer_type == 'cudnn.SpatialMaxPooling' then
        assert(layer.padW == 0 and layer.padH == 0)
         kW, kH = layer.kW, layer.kH
         dW, dH = layer.dW, layer.dH
         avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer:cuda()
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      if name == content_layers then
        print("Setting up content layer", i, ":", layer.name)
         next_content_idx = next_content_idx + 1
      end
    end
end
]]--
net:add(cnn:get(1))
cnn = nil
--[[for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'cudnn.SpatialConvolution' then
        print('removing acc for   ' .. torch.type(module))
        --net.modules[i].accGradParameters = function() end
    end
    if torch.type(module) == 'nn.Linear' then
        -- remove these, not used, but uses gpu memory
        print('cool')
        --net.modules[i].accGradParameters = function() end
    end
end]]--
collectgarbage()
net:cuda()
net2 = net:clone()
net2:cuda()
AE = nn.Sequential()
AE:add(cudnn.SpatialConvolution(1, 80, 7, 7 ,1, 1,3,3))
AE:add(cudnn.ReLU())
AE:add(cudnn.SpatialConvolution( 80, 60,1, 1))
AE:add(cudnn.ReLU())
AE:add(cudnn.SpatialConvolution(60, 1, 7, 7 ,1, 1,3,3))
AE:add(nn.SpatialFullConvolution(1, 1, 5, 5,2,1,2,2,1,0))-- in_filter,outfilters,kernel,kernel,scale,scale=1,kernel-1/2,kernel-1/2,scale-1,scale-1
AE:cuda()

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        f2=0
        f_test = 0
        neval = neval + 1
        for i = 1,10 do
            output = AE:forward(inputs[{{i},{},{},{}}]:cuda())
            err = criterion:forward(output,targets[{{i},{},{},{}}]:cuda())
            f = f + err
            
             expec = net:forward(kk:forward(output))
             actual = net2:forward(kk2:forward(targets[{{i},{},{},{}}]:cuda()))
            err = criterion:forward(expec,actual)
            f2=f2+err
            df_do = criterion:backward(expec,actual)
          df_dexpec = net:backward(kk3:forward(output),df_do)
          AE:backward(inputs[{{i},{},{},{}}]:cuda(), df_dexpec[{{},{1},{},{}}])
          collectgarbage()
      end
        table.insert(train,f/10)
        --AE:evaluate()
        for i = 1,17 do
            output = AE:forward(inputs_test[i]:cuda())
            err = criterion:forward(output,targets_test[i]:cuda())
            f_test = f_test + err
            collectgarbage()
        end
        --AE:training()
        table.insert(test,f_test/17)
        print(string.format('after %d evaluations J(x) = %f took %f %f', neval, f/10,  sys:toc(),gradParameters[1]))
      return f2/10,gradParameters/10
end

  
optimState = {maxIter = 100}
optimMethod = optim.lbfgs


AE:training()
parameters,gradParameters = AE:getParameters()
sys:tic()
train = {}
test = {}
--trainLogger = optim.Logger(('Results/train_normal.log'))
--testLogger = optim.Logger(('Results/test_normal.log'))
neval = 0
optimMethod(func, parameters, optimState)-- <------------------- optimization
--AE:evaluate()  
    