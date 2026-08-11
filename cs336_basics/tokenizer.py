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
        for key,v in vocab.items():
            self.vocab_inv[v]=key
    
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
        tokens=[]

        if self.special_tokens is not None:
            sorted_spec=sorted(self.special_tokens,key=len,reverse=True)
            special_pattern = '('+'|'.join(re.escape(token) for token in sorted_spec)+')'
            for segment in re.split(special_pattern, text):
                if segment in sorted_spec:
                    tokens.append((bytes(segment.encode('utf-8')),))
                    continue
                for match in re.finditer(PAT, segment):
                    tokens.append(tuple(bytes([b]) for b in match.group(0).encode("utf-8")))
        else:
            segment=text
            for match in re.finditer(PAT, segment):
                tokens.append(tuple(bytes([b]) for b in match.group(0).encode("utf-8")))

        for merge in self.merges:
            for now in range(len(tokens)):
                new_token=tuple()
                i=0
                while i < len(tokens[now])-1:
                    if tokens[now][i]==merge[0] and tokens[now][i+1]==merge[1]:
                        new_token=new_token+(merge[0]+merge[1],)
                        i+=2
                    else:
                        new_token=new_token +(tokens[now][i],)
                        i+=1
                if i==len(tokens[now])-1:
                    new_token=new_token+(tokens[now][-1],)
                tokens[now]=new_token
        # for token in tokens:
        #     for t in token:
        #         print(t)
        # print((tokens))
        return [self.vocab_inv[t] for token in tokens for t in token]

    
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self,ids:list[int])->str:
        ret=bytes([])
        for id in ids:
            ret=ret+self.vocab[id]
        return ret.decode("utf-8",errors='replace')
        # return [self.vocab[id].decode("utf-8") for id in ids]
    

if __name__=='__main__':
    vocab_filepath='data/processed/output/vocab.pkl'
    merges_filepath='data/processed/output/merge_pair.pkl'
    special_tokens=['<|endoftext|>']
    T=Tokenizer.from_files(vocab_filepath,merges_filepath,special_tokens)
    text="""<|endoftext|>"""
    # text = """I love kitty!"""
    print(T.encode(text))