#An example of bpe


# text="""low low low low low
# lower lower widest widest widest
# newest newest newest newest newest newest"""
# text="""abab"""
from collections import Counter
import regex as re

Merge_round=10000-256-1
Special_tokens=["<|endoftext|>"]

vacabulary=[bytes([byte]) for byte in range(256)] + [bytes(Special_token.encode("utf-8")) for Special_token in Special_tokens]
merge_pair=[]



#Training
def merge(counter,pair,pair_cnt):
    new_counter=Counter()

    for old_token,cnt in counter.items():
        new_token=()
        i=0
        while i<len(old_token)-1:
            if old_token[i]==pair[0] and old_token[i+1]==pair[1]:
                new_token=new_token+(pair[0]+pair[1],)
                pair_cnt[(pair[0],pair[1])]-=cnt
                if i>0:
                    pair_cnt[(new_token[-2],old_token[i])]-=cnt
                    pair_cnt[(new_token[-2],pair[0]+pair[1])]+=cnt
                if i<len(old_token)-2:
                    pair_cnt[(old_token[i+1],old_token[i+2])]-=cnt
                    pair_cnt[(pair[0]+pair[1],old_token[i+2])]+=cnt
                i+=2
            else:
                new_token=new_token+(old_token[i],)
                i+=1
        if i==len(old_token)-1:
            new_token=new_token+(old_token[i],)
        new_counter[new_token]+=cnt
    pair_cnt+=Counter()
    return new_counter,pair_cnt

def find_best_pair(counter,pair_cnt): #counter used only once.
    if pair_cnt is None:
        pair_cnt=Counter()
        for token in counter:
            for i in range(len(token)-1):
                pair_cnt[(token[i],token[i+1])]+=counter[token]
    best_pair,max_c=(),0
    for pair,c in pair_cnt.items():
        if c>max_c:
            max_c,best_pair = c,pair
            continue
        if c==max_c and best_pair<pair:
            best_pair=pair
    
    return best_pair,pair_cnt

def train(counter):
    pair_cnt=Counter()
    for m in range(Merge_round):
        best_pair,pair_cnt=find_best_pair(counter,None if m==0 else pair_cnt)
        if best_pair==():
            break
        merge_pair.append(best_pair)
        counter,pair_cnt=merge(counter,merge_pair[-1],pair_cnt)
        vacabulary.append(merge_pair[-1][0]+merge_pair[-1][1])
        # print(counter)

def main():
    counter=Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open('C:\\Users\\Kai_Admin\\Desktop\\assignment1-basics\\data\\debug\\chunk0_outof_100.txt','rb') as text:
        content=text.read().decode("utf-8")
    special_pattern = '|'.join(re.escape(token) for token in Special_tokens)
    for segment in re.split(special_pattern, content):
        for match in re.finditer(PAT, segment):
            pretoken=tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            # print(pretoken)
            counter[pretoken]+=1
    train(counter)
    # print(merge_pair)
    with open('C:\\Users\\Kai_Admin\\Desktop\\assignment1-basics\\data\\debug\\chunk0_output.txt','w') as output:
        output.write(str(vacabulary))

if __name__=="__main__":
    main()