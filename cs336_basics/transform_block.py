import torch
from torch import nn
from cs336_basics.multihead_self_attention import Multihead_self_attention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.positionwise_feedforward import SwiGLU

class Transformer_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_ff=d_ff
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.norm1=RMSNorm(d_model)
        self.norm2=RMSNorm(d_model)
        self.swiglu=SwiGLU(d_model,d_ff)
        self.attn=Multihead_self_attention(d_model,num_heads,theta,max_seq_len)
        
    def forward(self,x:torch.Tensor):
        x1=x+self.attn.multihead_self_attention(self.norm1(x),torch.arange(x.shape[-2]))
        
        x2=x1+self.swiglu(self.norm2(x1))
        return x2