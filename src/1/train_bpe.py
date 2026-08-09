import regex as re
import bpe_accelerate

"""configs"""

input_path='/Users/kaiadmin/Desktop/assignment1-basics/data/debug/chunk0_outof_100.txt'
output_dir='/Users/kaiadmin/Desktop/assignment1-basics/data/debug/output'
vocab_size=10000
special_tokens=["<|endoftext|>"]

bpe_accelerate.main(input_path,vocab_size,special_tokens)