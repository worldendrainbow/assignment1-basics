import torch
from torch import nn
from einops import einsum
import einops
import math
class Linear(nn.Module):
    # y:... out_f
    # x:... in_f
    # y = W,x,"out in,... in->... out"
    def __init__(self,
        in_features: int ,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None):
        super().__init__()
        self.W=nn.Parameter(torch.empty(out_features,in_features,dtype=dtype,device=device))
        sigma=math.sqrt(2./(in_features+out_features))
        nn.init.trunc_normal_(self.W,0,sigma,-3*sigma,3*sigma)

    def forward(self,x:torch.Tensor):
        return einsum(self.W,x,"out in,... in->... out")



class Embedding(nn.Module):
    # x:... T
    # emb:T*d
    # emb=einsum(W,x," T->T d")

    def __init__(self, 
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None ,
            dtype: torch.dtype | None = None ):
        super().__init__()
        self.W=nn.Parameter(torch.empty(num_embeddings,embedding_dim,dtype=dtype,device=device))
        nn.init.trunc_normal_(self.W,mean=0,std=1,a=-3,b=3)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:#token_ids with shape ... 
        # tmp=[]
        # shape=token_ids.shape
        # for i in token_ids:
        #     tmp.append(self.W[i].tolist())
        # return torch.tensor(torch.tensor(tmp).view(shape+(-1,)))
        return self.W[token_ids]
if __name__=='__main__':
    model = Embedding(10,10)
    token_ids=torch.arange(5)+3
    print(model.forward(token_ids))