import torch
from einops import einsum
from cs336_basics.softmax import softmax
import math

def scaled_dot_product_attention(Q,K,V,mask=None):
    d_k=Q.shape[-1]
    presoft=einsum(Q,K,"... q_len dk,... k_len dk->... q_len k_len")/math.sqrt(d_k)
    t_inf=torch.where(mask,0.,-torch.inf)
    presoft=presoft+t_inf
    softed=softmax(presoft,-1)
    return einsum(softed,V,"... q_len k_len,... k_len d_v->... q_len d_v")
if __name__=='__main__':
    # print(torch.exp(torch.tensor([-torch.inf])))
    # print(-0.1 *torch.inf)
    mask = torch.tensor([[True,False],[False,False]])
    
    # t=torch.tensor([0 if i else -torch.inf for i in mask])

    t_inf=torch.tensor([[0 if m_ else -torch.inf for m_ in line]for line in mask]).view_as(mask)
    print(t_inf)