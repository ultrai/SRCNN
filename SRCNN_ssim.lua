require 'sys'
require 'torch'
require 'cunn'
require 'nn' 
require 'cudnn'
--matio = require 'matio'
require 'optim'
require 'cutorch'
require 'math'
im = require 'image'
require 'nngraph'
cutorch.setDevice(1)
torch.setdefaulttensortype('torch.FloatTensor')

require 'hdf5'
myFile = hdf5.open('Data.h5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.x
temp = temp:type('torch.FloatTensor')
inputs = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:div(255)
temp=Temp.y
temp = temp:type('torch.FloatTensor')
targets = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:div(255)

myFile = hdf5.open('Data_test.h5', 'r')
Temp = myFile:read(''):all()
myFile:close()
temp=Temp.x
temp = temp:type('torch.FloatTensor')
inputs_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:div(255)
temp=Temp.y
temp = temp:type('torch.FloatTensor')
targets_test = torch.Tensor(temp:size(1),temp:size(2),temp:size(3),temp:size(4)):copy(temp)--:div(255)

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
cmd:option('-nfeat2',80,'Number of filters to be considered')
cmd:option('-feat_sz',15,'Each filter size')
cmd:option('-feat_sz2',15,'Each filter size')
cmd:option('-iterations',300,'total no of iterations')
cmd:text()

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

Model = {}

--criterion = nn.MSECriterion()
--criterion:cuda()

n=11

mu_1 = grad.nn.SpatialAveragePooling(n,n,1,1)--,(n-1)/2,(n-1)/2)
mu_2 = grad.nn.SpatialAveragePooling(n,n,1,1)--,(n-1)/2,(n-1)/2)
mu_3 = grad.nn.SpatialAveragePooling(n,n,1,1)--,(n-1)/2,(n-1)/2)
mu_4 = grad.nn.SpatialAveragePooling(n,n,1,1)--,(n-1)/2,(n-1)/2)
mu_5 = grad.nn.SpatialAveragePooling(n,n,1,1)--,(n-1)/2,(n-1)/2)


cm_1 =  grad.nn.CMulTable()
cm_2 =  grad.nn.CMulTable()
cm_3 =  grad.nn.CMulTable()
cm_4 =  grad.nn.CMulTable()
cm_5 =  grad.nn.CMulTable()
cm_6 =  grad.nn.CMulTable()
cm_7 =  grad.nn.CMulTable()
cm_8 =  grad.nn.CMulTable()

cd_1 =  grad.nn.CDivTable()

ssim = function (y,x)
       mu_x = mu_1(x)
       mu_y = mu_2(y)
       mu_x_sq = cm_1({mu_x,mu_x})
       mu_y_sq = cm_2({mu_y,mu_y})
       mu_xy = cm_3({mu_x,mu_y})
       X_2 = cm_4({x,x})
       Y_2 = cm_5({y,y})
       XY = cm_6({x,y})
       sigma_x_sq = mu_3(X_2)-mu_x_sq
       sigma_y_sq = mu_4(Y_2)-mu_y_sq
       sigma_xy = mu_5(XY)-mu_xy
       A1 = mu_xy*2+5
       A2 = sigma_xy*2+45.5
       B1 = mu_x_sq+mu_y_sq+5
       B2 = sigma_x_sq+sigma_y_sq+45.5
       A = cm_7({A1,A2})
       B = cm_8({B1,B2})
       v = -torch.mean(cd_1({A,B}))
       return v
end
df = grad(ssim)





Cost = 999999

func = function(x)
        if x ~= parameters then
              parameters:copy(x)
        end
        gradParameters:zero()
        f = 0
        f_test = 0
        neval = neval + 1
batch=1
b =0
f2 = 0
        for i = 1,inputs:size(1) do
            output = AE:forward(inputs[i][{{1},{},{}}]:cuda())
            df_do, f = df(output:float(),inputs[i][{{1},{},{}}])
            AE:backward(inputs[i][{{1},{},{}}]:cuda(), df_do:cuda())
 	    collectgarbage()
        end
        table.insert(train,f/b)
--        AE:evaluate()
        for i = 1,inputs_test:size(1) do
            output = AE:forward(inputs_test[i][{{1},{},{}}]:cuda())
            err = criterion:forward(output,targets_test[i][{{1},{},{}}]:cuda())
            f_test = f_test + err
            collectgarbage()
        end
--        AE:training()
        table.insert(test,f_test/inputs_test:size(1))
        print(string.format('after %d evaluations J(x) = %f %f took %f %f', neval, f/b, f2/b, sys:toc(),f_test/inputs_test:size(1)))
      return f/b,gradParameters/b
end

optimState = {maxIter = opt.iterations}
optimMethod = optim.sgd--cg--adagrad--cg

input = nn.Identity()()
L1 = cudnn.ReLU(true)(cudnn.SpatialConvolution(80, 1, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(120, 80,1,1)(cudnn.ReLU(true)(cudnn.SpatialConvolution(1, 120, 15,15,1,1,7,7)(input))))))
--L1_out = nn.JoinTable(1)({L1,input})
--n=64
--L2 = cudnn.ReLU(true)(cudnn.SpatialConvolution(n, 1, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(1, n, 15,15,1,1,7,7)(L1))))))
--L2_1 = cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(L2))))))
--L2_2 = cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(L2_1))))))
--L2_3 = cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 15,15,1,1,7,7)(L2_2))))))

--L3 = cudnn.SpatialConvolution(n, 1, 3,3,1,1,1,1)(cudnn.ReLU(false)(cudnn.SpatialConvolution(n, n, 7,7,1,1,3,3)(cudnn.ReLU(true)(cudnn.SpatialConvolution(n, n, 9,9,1,1,4,4)(L2)))))

--L5 =  cudnn.ReLU(true)(nn.CAddTable()({L1, L2}))
AE = nn.gModule({input}, {L1})

AE:cuda()
AE:training()
parameters,gradParameters = AE:getParameters()
sys:tic()
train = {}
test = {}
--trainLogger = optim.Logger(('Results/train_normal.log'))
--testLogger = optim.Logger(('Results/test_normal.log'))
neval = 0
for temppp = 1,150 do
optimMethod(func, parameters, optimState)-- <------------------- optimization
end
AE:evaluate()
for i = 1,10 do
      output = AE:forward(inputs[i]:cuda())
      im.save('Results/Train_' .. i .. '_SRCNN.jpg', output:float():div(255))
      im.save('Results/Train_' .. i .. '_truth.jpg', targets[i]:div(255))
end
for i = 1,17 do
      output = AE:forward(inputs_test[i]:cuda())
      im.save('Results/Test_' .. i .. '_SRCNN.jpg', output:float():div(255))
      im.save('Results/Test_' .. i .. '_truth.jpg', targets_test[i]:div(255))
end
train = torch.Tensor(train)
test = torch.Tensor(test)
torch.save('Model_L5_L1.t7',AE)
torch.save('train_L5_L1.t7',train)--torch.save('train.txt',train,'ascii')
torch.save('test_L5_L1.t7',test)
--[[myFile = hdf5.open('test.h5', 'w')
myFile:write('', test)
myFile:close()]]--
AE = nil
collectgarbage()


