import torch
from torch import nn
from cs336_basics.transform_block import Transformer_block
from cs336_basics.linear_and_embedding import Embedding,Linear
from cs336_basics.rmsnorm import RMSNorm
class Transformer_lm(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta):
        super().__init__()
        self.vocab_size,self.context,self.d_model,self.num_layers,self.num_heads,self.d_ff,self.rope=vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta

        self.layers=nn.ModuleList([Transformer_block(d_model,num_heads,d_ff,rope_theta,context_length) for _ in range(num_layers)])
        self.embedding=Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.ln_final=RMSNorm(d_model)
        self.lm_head=Linear(d_model,vocab_size)

    def forward(self,input):#in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            #`sequence_length` is at most `context_length`.
        y=self.embedding.forward(input)
        for l in self.layers:
            y=l(y)#?
        y=self.ln_final(y)
        y=self.lm_head(y)#?
        return y