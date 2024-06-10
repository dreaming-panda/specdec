from llama import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer
import torch.nn.functional as F
import torch.distributed as dist
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask

dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

DEVICE = torch.device("cuda", local_rank)
DTYPE = torch.float16
STEP = 32
GAMMA = 3
TEMP = 0.6
VOCAB = 32000


data = get_dataset("gsm8k", 100)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
BOS = tokenizer.bos_token_id
if PAD == None:
    PAD = EOS
draft = LLMEngine(model_name="meta-llama/Llama-2-7b-chat-hf", batch_size=1, max_length=4096, device=DEVICE, dtype=DTYPE)
#target = LLMEngine(model_name="meta-llama/Llama-2-13b-chat-hf", batch_size=1, max_length=4096, device=DEVICE, dtype=DTYPE)
causal_mask = _make_causal_mask((1, 4096), DTYPE, DEVICE)
storage_ids = torch.arange(start=0, end=4096, device=DEVICE).long()
position_ids = torch.arange(start=0, end=4096, device=DEVICE).long().unsqueeze(0)

draft_proba_buffer = torch.zeros((GAMMA, VOCAB)).to(DEVICE)

NUM_STEPS = 0
ACCEPT_TOKENS = 0

    
tokens = torch.tensor([[    1,  2627,   300, 30010, 29879,   868,  4684,  6568, 29871, 29896,
         29953, 29808,   639,  2462, 29889,  2296,   321,  1446,  2211,   363,
         26044,  1432,  7250,   322,   289,  6926,   286,  3096,  1144,   363,
           902,  7875,  1432,  2462,   411,  3023, 29889,  2296,   269, 10071,
           278, 21162,   472,   278,  2215, 13269, 29915,  9999, 14218,   363,
           395, 29906,   639, 10849,   868,   384, 19710, 29889,  1128,  1568,
           297, 17208,   947,  1183,  1207,  1432,  2462,   472,   278,  2215,
         13269, 29915,  9999, 29973,    13, 26626,   300]], device=DEVICE)

prompt_len = tokens.shape[1]
dlogits = draft.inference(input_ids=tokens, storage_ids=storage_ids[:prompt_len], 
            position_ids=position_ids[:,:prompt_len], attention_mask=causal_mask[None, None, :, :][...,:prompt_len,:])

if local_rank == 0:
    print(dlogits)

draft.llm.kv_cache.gather_kv_incremental(indices=[74, 75, 76], offset=74)


tokens = torch.tensor([[29915]], device=DEVICE)


dlogits = draft.inference(input_ids=tokens, storage_ids=storage_ids[77:78], 
            position_ids=position_ids[:,77:78], attention_mask=causal_mask[None, None, :, :][...,77:78,:])
if local_rank == 0:
    print(dlogits)
