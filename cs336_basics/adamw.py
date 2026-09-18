from collections.abc import Callable,Iterable
import torch
import math
class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),weight_decay=0.01,eps=1e-8):
        if lr<0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults={"lr":lr,"betas":betas,"weight_decay":weight_decay,"eps":eps}
        super().__init__(params,defaults)
    
    def step(self,closure:Callable|None=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            betas=group["betas"]
            beta1=betas[0]
            beta2=betas[1]
            weight_decay=group["weight_decay"]
            eps=group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state=self.state[p]
                t=state.get("t",1)
                m=state.get("m",0)
                v=state.get("v",0)
                grad=p.grad.data
                lr_t=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data*=(1-lr*weight_decay)
                m=beta1*m+(1-beta1)*grad
                v=beta2*v+(1-beta2)*(grad**2)
                p.data-=lr_t*m/(torch.sqrt(v)+eps)
                state["m"]=m;state["v"]=v;state["t"]=t+1
        return loss
