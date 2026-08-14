import torch
import einops
from torch import nn 
import math
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta , self.d_k , self.max_seq_len = theta , d_k , max_seq_len

        theta_=torch.tensor([[i*(self.theta**(-(2*k-2)/self.d_k)) 
                            for k in range(1,self.d_k//2+1)]
                            for i in range(self.max_seq_len)],
                            dtype=torch.float32,
                            device=device)
        self.register_buffer("cos",torch.cos(theta_),persistent=True)
        self.register_buffer("sin",torch.sin(theta_),persistent=True)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [... T d/2]
        # tp:[... T]
        cos=self.cos[token_positions]
        sin=self.sin[token_positions]
        x_even= x[...,0::2]
        x_odd = x[...,1::2]
        
        y_even= cos*x_even-sin*x_odd
        y_odd = sin*x_even+cos*x_odd
        # 0,1,0,1,0,1......
        y=torch.stack([y_even,y_odd],dim=-1).flatten(-2)
        return y

if __name__=='__main__':
    rope=RotaryPositionalEmbedding(0.1,4,8)
    # print(rope.cos)
    # print(rope.sin)
    token_positions=torch.arange(5*8).view(5,8)%8
    # print(token_positions)
    x=torch.randn((5,8,4))
    # print(x)
    # print(rope.forward(x,token_positions))

    # even=torch.tensor(range(10)[0:10:2])
    # odd=torch.tensor(range(10)[1:10:2])
    # o=torch.stack([even,odd],dim=-1).flatten()
    # print(o)