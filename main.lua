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


