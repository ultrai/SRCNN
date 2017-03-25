# SRCNN
Embedding seperable filters into deeper architectures for sparsely sampled (super resolving) optical coherence tomography images

*update: Incompatability between anaconda and fblualibn led to new Data.py and modified SRCNN.lua*
## Summary
As lower layer filters learns gabor like structures they can be learned using separable filters with parallel architecture. 
This achives similar performance to Fully connected with 46% less parameters.


## Computational Environment
8GB Ram + 2GB GPU Ubuntu 14.04

## Required installations
1. [Anaconda python](https://www.continuum.io/downloads)
2. [cudnn](https://developer.nvidia.com/cudnn)
3. [Torch](http://torch.ch/docs/getting-started.html#_)
~~4. [fblualib]~~

## Data set
Prof. Sina Farsiu team has generously made the data available [here!](http://people.duke.edu/~sf59/Fang_TMI_2013.htm)

## Training models and reproducing results
```bash
$ python Data.py (update)
$ th 
th> dofile('main.lua')

```
## Competing approaches
1. [Structured random forest code](http://lrs.icg.tugraz.at/downloads/srf_cvpr15_public_v1.00.zip)
2. [Structured random forest web adress](http://lrs.icg.tugraz.at/members/schulter)
