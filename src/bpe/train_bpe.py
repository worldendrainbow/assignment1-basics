import regex as re
import bpe_accelerate
import os

"""configs"""

input_path='data/debug/chunk0_outof_100.txt'
output_dir='data/debug/output'
vocab_size=1000
special_tokens=["<|endoftext|>"]

with open(f'{output_dir}/chunk0.txt','w') as f:
    vocab,merge_pair=bpe_accelerate.main(input_path,vocab_size,special_tokens)
    f.write(str(vocab))
    f.write(str(merge_pair))
    # f.write(bpe_accelerate.main(input_path,vocab_size,special_tokens))