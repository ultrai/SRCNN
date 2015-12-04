# SRCNN
Embedding seperable filters into deeper architectures for sparsely sampled (super resolving) optical coherence tomography images
## Summary
As lower layer filters learns gabor like structures they can be learned using separable filters with parallel architecture. 
This achives similar performance to Fully connected with 20% less parameters.




## computational Environment
8GB Ram + 2GB GPU

## Required installations
1) [Anaconda python](https://www.continuum.io/downloads)
2) [cudnn](https://developer.nvidia.com/cudnn)
3) [Torch](http://torch.ch/docs/getting-started.html#_)
4) [fblualib](https://github.com/facebook/fblualib)

## Data set
Prof. Sina Farsiu team has generously made the data available [here!](http://people.duke.edu/~sf59/Fang_TMI_2013.htm)

## Training models and reproduce results
$ th dofile('main.lua')






