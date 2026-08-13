import math
import torch
import einops
from torch import nn
from .linear_and_embedding import Linear
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,
        device: torch.device | None = None ,
        dtype: torch.dtype | None = None ):
        super().__init__()
        self.W1=Linear(d_model,d_ff,device,dtype)
        self.W2=Linear(d_ff,d_model,device,dtype)
        self.W3=Linear(d_model,d_ff,device,dtype)

    def SiLU(x:torch.tensor):
        return x*torch.sigmoid(x)

    def FFN(self,x):
        w1x = self.W1.forward(x)
        w3x = self.W3.forward(x)
        t=SwiGLU.SiLU(w1x)*w3x
        return self.W2.forward(t)
    
if __name__=='__main__':
    pass