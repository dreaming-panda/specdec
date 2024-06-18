import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
from model import Transformer
import argparse

from backend import LMBackend


parser = argparse.ArgumentParser(description='Your CLI description.')

parser.add_argument('--checkpoint_path', type=Path, default=Path("/home/zhuominc/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--batch', type=int, help='batch size')
parser.add_argument('--maxlen', type=int, help='max len')
parser.add_argument('--declen', type=int, help='decode len')
parser.add_argument('--prefixlen', type=int, help='prefix len')
parser.add_argument('--device', type=str, help='device')
args = parser.parse_args()

checkpoint_path = args.checkpoint_path
device = args.device
precision = torch.bfloat16
use_tp = False
max_seq_length = args.maxlen
max_batch_size = args.batch
prefix_len = args.prefixlen
declen = args.declen
declen_mix = args.declen
warm_up = 10
T = 1000
torch.cuda.set_device(device=device)
causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device))

llm = LMBackend(dtype=precision, device=device)
llm.load_model(checkpoint_path)
if args.compile:
    llm.compile()


llm.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)


prompt = torch.randint(low=3, high=30000, size=(1, prefix_len), device=device)
input_pos = torch.arange(0, prefix_len, device=device)



dec = torch.randint(low=3, high=30000, size=(1, declen), device=device)
dec_pos = torch.arange(prefix_len, prefix_len + declen, device=device)
cache_pos = torch.arange(prefix_len, prefix_len + declen, device=device)
dec_mask = causal_mask[prefix_len:prefix_len + declen][None, None, :, :]


llm.encode(input_ids=prompt, position_ids=input_pos, storage_ids=None, attention_mask=None)
with torch.inference_mode():
        t1 = time.perf_counter()
        for _ in range(T):
            logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
        t2 = time.perf_counter()
        #torch.cuda.synchronize()
        # t1 = time.perf_counter()
        # for _ in range(T):
        #     logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
        # torch.cuda.synchronize()
        # t2 = time.perf_counter()

print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(max_seq_length, declen, prefix_len, (t2 - t1)/ T))

prefix_len_mix = prefix_len + 2
prompt = torch.randint(low=3, high=30000, size=(1, prefix_len_mix), device=device)
input_pos = torch.arange(0, prefix_len_mix, device=device)
dec = torch.randint(low=3, high=30000, size=(1, declen_mix), device=device)
dec_pos = torch.arange(prefix_len_mix, prefix_len_mix + declen_mix, device=device)
cache_pos = torch.arange(prefix_len_mix, prefix_len_mix + declen_mix, device=device)
dec_mask = causal_mask[prefix_len_mix:prefix_len_mix + declen_mix][None, None, :, :]


llm.encode(input_ids=prompt, position_ids=input_pos, storage_ids=None, attention_mask=None)
with torch.inference_mode():
        t1 = time.perf_counter()
        for _ in range(T):
            logits = llm.inference(input_ids=dec, position_ids=dec_pos, storage_ids=cache_pos, attention_mask=dec_mask)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

print("Max Length :{}, Decode Length :{}, Prefix Length :{}, inference time:{}s".format(max_seq_length, declen, prefix_len, (t2 - t1)/ T))