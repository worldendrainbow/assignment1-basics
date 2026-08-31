import torch
from torch import nn
from einops import einsum
from einops import rearrange
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.softmax import softmax
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear_and_embedding import Linear
import math
from itertools import product

class Multihead_self_attention_no_rope(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.qw=Linear(d_model,d_model)
        self.kw=Linear(d_model,d_model)
        self.vw=Linear(d_model,d_model)
        self.ow=Linear(d_model,d_model)
        self.num_heads=num_heads
        self.d_model=d_model
    def multihead_self_attention_no_rope(self,in_features):
        Q=rearrange(self.qw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)
        K=rearrange(self.kw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)
        V=rearrange(self.vw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)

        k_len=int(K.size(-2))
        q_len=int(Q.size(-2))
        mask=torch.tril(torch.ones((q_len,k_len),device=Q.device,dtype=torch.bool))
        mask=mask.unsqueeze(0).unsqueeze(0)

        attention=scaled_dot_product_attention(Q,K,V,mask=mask)
        return einsum(rearrange(self.ow.W,"d_model (h d_head)->d_model h d_head",h=self.num_heads),
                      attention,"d_model h d_head,... h seq_len d_head->... seq_len d_model")
    

class Multihead_self_attention(nn.Module):
    def __init__(self,d_model,num_heads,theta,max_seq_len):
        super().__init__()
        self.qw=Linear(d_model,d_model)
        self.kw=Linear(d_model,d_model)
        self.vw=Linear(d_model,d_model)
        self.ow=Linear(d_model,d_model)
        self.num_heads=num_heads
        self.d_model=d_model
        self.rp=RotaryPositionalEmbedding(theta,d_model//num_heads,max_seq_len)


    def multihead_self_attention(self,in_features,token_positions):
        Q_=rearrange(self.qw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)
        K_=rearrange(self.kw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)
        V=rearrange(self.vw.forward(in_features),"... seq_len (h d_head)->... h seq_len d_head",h=self.num_heads)

        
        Q=self.rp.forward(Q_,token_positions)
        K=self.rp.forward(K_,token_positions)
        
        k_len=int(K.size(-2))
        q_len=int(Q.size(-2))
        mask=torch.tril(torch.ones((q_len,k_len),device=Q.device,dtype=torch.bool))
        mask=mask.unsqueeze(0).unsqueeze(0)

        attention=scaled_dot_product_attention(Q,K,V,mask=mask)
        return einsum(rearrange(self.ow.W,"d_model (h d_head)->d_model h d_head",h=self.num_heads),
                      attention,"d_model h d_head,... h seq_len d_head->... seq_len d_model")


if __name__=='__main__':
    pass
    # a=torch.tensor([1,2,3])
    # print(torch.ones_like(a,dtype=bool))
    A=torch.randn(2,3,4)
    for idx in product(*(range(n) for n in A.shape[:-1])):
        print(idx+(1,))
    print(A[idx])