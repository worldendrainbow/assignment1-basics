import importlib.metadata
from . import rope,tokenizer,rmsnorm,linear_and_embedding,bpe_accelerate,positionwise_feedforward,softmax,scaled_dot_product_attention
from . import multihead_self_attention,transform_block,transform_lm
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
