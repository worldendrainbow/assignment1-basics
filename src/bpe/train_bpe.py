import regex as re
import bpe_accelerate
import os
from multiprocessing import Pool,Process
import pickle

"""configs"""

input_path='data/processed'
output_dir='data/processed/output'
vocab_size=300
special_tokens=["<|endoftext|>"]
num_workers=4
chunk_size=128

def worker(input_path,output_dir,vocab_size,special_tokens,chunk_id):
    with open(f'{output_dir}/chunk{chunk_id}.pkl','wb') as f:
        vocab,merge_pair=bpe_accelerate.main(f'{input_path}/chunk{chunk_id}_outof_128.txt',vocab_size,special_tokens)
        pickle.dump((vocab,merge_pair),f)

if __name__ == '__main__':
    os.makedirs(output_dir,exist_ok=True)
    assert chunk_size>0 and num_workers>0
    with Pool(processes=num_workers) as pool:
        pool.starmap(worker, [(input_path, output_dir, vocab_size, special_tokens, chunk_id) for chunk_id in range(chunk_size)])