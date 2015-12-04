# SRCNN
Embedding seperable filters into deeper architectures for sparsely sampled (super resolving) optical coherence tomography images
## Summary
As lower layer filters learns gabor like structures they can be learned using separable filters with parallel architecture. 
This achives similar performance to Fully connected with 20% less parameters.

<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/test.j1pg" alt = "Test image" width="100" >
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_SRCNN.j1pg" width="200">
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_Proposed.j1pg" width="200">
<img src="https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Test_1_truth.j1pg" width="200"a)Test image b)Fully connected CNN c)Modified CNN with 20% less parameters d)Anticipated super resolved image>



![image](https://raw.githubusercontent.com/ultrai/SRCNN/master/Results/Data_plot.p1ng PSNR profiles of CNN and modified CNN across training and testing datasets)



## computational Environment
8GB Ram + 2GB GPU

## Required installations
1) Anaconda python
2) cudnn
3) Torch
4) fblualib

## Data set
Prof. Sina Farsiu team has generously made the data available [here!](http://people.duke.edu/~sf59/Fang_TMI_2013.htm)

## Training models and reproduce results
$ th dofile('main.lua')






