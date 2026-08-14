import importlib.metadata
from . import rope,tokenizer,rmsnorm,linear_and_embedding,bpe_accelerate,positionwise_feedforward,softmax
try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
