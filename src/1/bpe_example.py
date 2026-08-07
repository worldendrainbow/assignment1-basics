#An example of bpe


text="""low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

from collections import Counter

Merge_round=6
Special_tokens=["<|endoftext|>"]
pre_tokenized_text=text.split()

counter=Counter()
vacabulary=[bytes([byte]) for byte in range(256)] + [bytes(Special_token.encode("utf-8")) for Special_token in Special_tokens]
merge_rank=[]
# print("vacabulary:", vacabulary)

for pretoken in pre_tokenized_text:
    key=tuple(bytes([c]) for c in pretoken.encode("utf-8"))
    counter[key]+=1

print("counter:", counter)

#Training
def merge(pair):
    new_counter=Counter()
    for old_token,cnt in counter.items:       #token shape:(bytes[],bytes[],byetes[],...)
        new_token=old_token
        for i in range(len(old_token)-1):
            if new_token[i]==pair[0] and new_token[i+1]==pair[1]:
                new_token= new_token[:i]+(pair[0]+pair[1],)+new_token[i+2:]
        new_counter[new_token]+=cnt
    return new_counter


def find_best_pair():
    pass


def train():
    pass


