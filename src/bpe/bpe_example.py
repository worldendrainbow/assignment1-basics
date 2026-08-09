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
merge_pair=[]

for pretoken in pre_tokenized_text:
    key=tuple(bytes([c]) for c in pretoken.encode("utf-8"))
    counter[key]+=1


#Training
def merge(pair):
    new_counter=Counter()
    for old_token,cnt in counter.items():       #token shape:(bytes[],bytes[],byetes[],...)
        new_token=old_token
        i=0
        while i < len(new_token)-1:
            if new_token[i]==pair[0] and new_token[i+1]==pair[1]:
                new_token= new_token[:i]+(pair[0]+pair[1],)+new_token[i+2:]
            i+=1
        new_counter[new_token]+=cnt
    return new_counter

def find_best_pair():
    cnt=Counter()
    for token in counter:
        for i in range(len(token)-1):
            cnt[(token[i],token[i+1])]+=counter[token]
    max_c=0;arg_pair=()
    for pair,c in cnt.items():
        if c>max_c:
            max_c,arg_pair = c,pair
            continue
        if c==max_c and arg_pair<pair:
            arg_pair=pair
    return arg_pair

def train():
    for m in range(Merge_round):
        global counter
        merge_pair.append(find_best_pair())
        counter=merge(merge_pair[-1])
        vacabulary.append(merge_pair[-1][0]+merge_pair[-1][1])
        # print(counter)

def main():
    train()
    print(merge_pair)
    print(vacabulary)

if __name__=="__main__":
    main()