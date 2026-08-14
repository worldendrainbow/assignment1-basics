import torch
from torch import nn 
import math
def softmax(x:torch.Tensor,i:int)->torch.Tensor:
    # x:tensor(0,1,2,3,4, ... i ... n)
    _max=x.max(axis=i,keepdim=True).values
    x=x-_max
    s=torch.exp(x).sum(axis=i,keepdim=True)
    _softmax=torch.exp(x)/s
    return _softmax

if __name__=='__main__':
    x=torch.arange(12).view(3,4)
    print(x)
    print(softmax(x,1))