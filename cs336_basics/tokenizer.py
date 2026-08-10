from collections.abc import Iterable,Iterator
import regex as re
import pickle

class Tokenizer:
    def __init__(
            self,
            vocab:dict[int,bytes],
            merges:list[tuple[bytes,bytes]],
            special_tokens:list[str] | None=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        self.vocab_inv={}
        for key,v in vocab:
            self.vocab_inv[key]=v
    
    @classmethod
    def from_files(cls,
                   vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens:list[str] | None=None):

        with open(vocab_filepath,'rb') as v:
            vocab=pickle.load(v)
        with open(merges_filepath,'rb') as m:
            merge_pair=pickle.load(m)
        return Tokenizer(vocab,merge_pair,special_tokens)
    
    def encode(self,text:str)->list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        special_pattern = '|'.join(re.escape(token) for token in self.special_tokens)

        tokens=[]
        for segment in re.split(special_pattern, text):
            for match in re.finditer(PAT, segment):             
                tokens.append(tuple(bytes([b]) for b in match.group(0).encode("utf-8")))

        for merge in self.merges:
            for now in range(len(tokens)):
                new_token=tuple()
                i=0
                while i < len(tokens[now])-1:
                    if tokens[now][i]==merge[0] and tokens[now][i+1]==merge[1]:
                        new_token=new_token+tuple(merge[0]+merge[1])
                        i+=2
                    else:
                        i+=1
                tokens[now]=new_token
                
        return [self.vocab[t] for token in tokens for t in token]

    
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:

        raise NotImplementedError

    def decode(self,ids:list[int])->str:

        raise NotImplementedError