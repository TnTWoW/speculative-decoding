import gzip
import random
import tqdm
import numpy as np
import time
from functools import wraps, partial
import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.cuda import synchronize, Event
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

timer = partial(Event, enable_timing = True)

from speculative_decoding import (
    Decoder,
    base_decoding,
    speculative_decoding
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512
GAMMA = 5

DEVICE_STR = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama2-13b" : "/data1/share/Llama-2-13b-chat-hf",
    "llama2-70b" : "/data1/share/Llama-2-70b-chat-hf",
}
# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def benchmark(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        start_event = timer()
        end_event = timer()
        start_event.record()

        out = fn(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        return out, elapsed_time_ms
    return inner

# instantiate transformer

device = torch.device(DEVICE_STR)
approx_model_name = MODELZOO["llama2-13b"]
target_model_name = MODELZOO["llama2-70b"]
tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
approx_model = AutoModelForCausalLM.from_pretrained(approx_model_name, trust_remote_code=True).to(device)
target_model = AutoModelForCausalLM.from_pretrained(target_model_name, trust_remote_code=True).to(device)

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

approx_model.eval()
target_model.eval()

inp = random.choice(val_dataset)[:PRIME_LENGTH]
prime = decode_tokens(inp)
print(f"%s \n\n %s", (prime, "*" * 100))

prompt = inp[None, ...]

sampled, base_decode_elapsed = benchmark(base_decoding)(target_model, prompt, GENERATE_LENGTH)

(spec_decode_sampled, num_accepted), spec_decode_elapsed = benchmark(speculative_decoding)(approx_model, target_model, prompt,
                                                                                           GENERATE_LENGTH, GAMMA)

base_decode_output = decode_tokens(sampled[0])
spec_decode_output = decode_tokens(spec_decode_sampled[0])

print("\nbase decoding:\n\n", base_decode_output, "\n")
print("\nspec decoding:\n\n", spec_decode_output, "\n")

print(f'base decoding in: {base_decode_elapsed:.3f}ms\n')
print(f'spec decoding in: {spec_decode_elapsed:.3f}ms\n')
print(f'average num accepted: {num_accepted:.1f} / {GAMMA}\n')