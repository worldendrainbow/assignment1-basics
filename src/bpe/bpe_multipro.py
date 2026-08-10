
from collections import Counter
import regex as re
from multiprocessing import Pool
import os
Log = True
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

def train(counter,Merge_round,vocabulary):
    pair_cnt=Counter()
    merge_pair=[]
    for m in range(Merge_round):
        best_pair,pair_cnt=find_best_pair(counter,None if m==0 else pair_cnt)
        if best_pair==():
            break
        merge_pair.append(best_pair)
        counter,pair_cnt=merge(counter,merge_pair[-1],pair_cnt)
        vocabulary.append(merge_pair[-1][0]+merge_pair[-1][1])
        if Log:
            print(f'Merge round {m+1}/{Merge_round}: Merged pair {best_pair} with count {pair_cnt[best_pair]}')

    return merge_pair

def worker(input_path,Special_tokens):
    with open(input_path,'rb') as text:
        content=text.read().decode("utf-8")
    counter=Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    special_pattern = '|'.join(re.escape(token) for token in Special_tokens)
    for segment in re.split(special_pattern, content):
        for match in re.finditer(PAT, segment):
            pretoken=tuple(bytes([b]) for b in match.group(0).encode("utf-8"))
            # print(pretoken)
            counter[pretoken]+=1

    if Log:
        print('Finished processing:', os.path.basename(input_path))
    return counter

def main(input_path,chunk_size,num_workers,vocab_size,Special_tokens):
    vocabulary=[bytes([byte]) for byte in range(256)] + [bytes(Special_token.encode("utf-8")) for Special_token in Special_tokens]
    Merge_round=vocab_size-256-len(Special_tokens)
    
    counter=Counter()

    with Pool(processes=num_workers) as pool:
        counters = pool.starmap(worker, [(f'{input_path}/chunk{chunk_id}_outof_128.txt', Special_tokens) for chunk_id in range(chunk_size)])
        for cnt in counters:
            counter.update(cnt)
    if Log:
        print('Finished processing all chunks.')

    merge_pair=train(counter,Merge_round,vocabulary)

    vocab_ret={}
    for i in range(len(vocabulary)):
        vocab_ret[i]=vocabulary[i]
    return (vocab_ret,merge_pair)


if __name__=="__main__":
    main()