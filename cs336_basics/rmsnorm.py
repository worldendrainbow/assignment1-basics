import torch
import einops
from torch import nn 
import math
class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int ,
        eps: float = 1e-5 ,
        device: torch.device | None = None ,
        dtype: torch.dtype | None = None ):
        super().__init__()
        self.d_model=d_model
        self.G=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps=eps
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        #(batch_size, sequence_length, d_model)
        in_dtype=x.dtype
        x=x.to(torch.float32)
        RMS=torch.sqrt((x*x).mean(axis=-1)+self.eps)
        print(RMS.shape)
        RMS=einops.repeat(1./RMS,"b s->b s d",d=self.d_model)
        g=einops.repeat(self.G,"d->b s d",b=x.size(0),s=x.size(1))
        # ret=1./RMS*x*self.G
        ret=RMS*x*g
        x=x.to(in_dtype)
        return ret
    
if __name__=='__main__':
    model=RMSNorm(10)
    x=torch.arange(80).view(2,4,10)
    # print((x*x).size())
    # print(model.forward(x))