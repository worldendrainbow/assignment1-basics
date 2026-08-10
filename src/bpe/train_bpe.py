import regex as re
import bpe_multipro
import os
from multiprocessing import Pool,Process
import pickle

"""configs"""

input_path='data/processed'
output_dir='data/processed/output'
vocab_size=10000
special_tokens=["<|endoftext|>"]
num_workers=8
chunk_size=128

if __name__ == '__main__':
    os.makedirs(output_dir,exist_ok=True)
    assert chunk_size>0 and num_workers>0
    vocab, merge_pair = bpe_multipro.main(input_path, chunk_size, num_workers, vocab_size, special_tokens)
    pickle.dump(vocab, open(os.path.join(output_dir, 'vocab.pkl'), 'wb'))
    pickle.dump(merge_pair, open(os.path.join(output_dir, 'merge_pair.pkl'), 'wb'))