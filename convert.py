from transformers import GPT2Tokenizer
import torchaudio
from tqdm import tqdm
import os
import argparse

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

data_dir = './data'
train_set = 'train-clean-100'
test_set = 'test-clean'
parser = argparse.ArgumentParser("conformer")
parser.add_argument("--convert_set", type=str, default="test-clean")
args = parser.parse_args()

# load dataset
librispeech = torchaudio.datasets.LIBRISPEECH(root=data_dir, url=args.convert_set, download=False)
text = librispeech[0][2]
print(f"Original Text: {text}")

# use GPT-2 tokenizer
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
tokenizer.pad_token = tokenizer.eos_token

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
print(tokenizer.vocab_size)
print(tokenizer.unk_token)
print(tokenizer.bos_token)
print(tokenizer.eos_token)
print(tokenizer.pad_token)
print(tokenizer._convert_token_to_id(tokenizer.pad_token))

def tokenize_batch(batch):
    return tokenizer(batch, truncation=True, padding="longest")

file = open(os.path.join(data_dir, f"{args.convert_set}-ids.txt"), "w")
for i in tqdm(librispeech):
    text = i[2]
    mark = f"{i[3]}-{i[4]}-{i[5]}"
    token_ids = tokenize_batch(text)['input_ids']
    file.write(f"{mark}: {token_ids}\n")

