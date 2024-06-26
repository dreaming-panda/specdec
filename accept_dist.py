from llama import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='random_seed')
args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(int(args.seed))
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

data = get_dataset("gsm8k", 10)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
BOS = tokenizer.bos_token_id
if PAD == None:
    PAD = EOS
draft = LLMEngine(model_name="meta-llama/Llama-2-7b-chat-hf", batch_size=1, max_length=4096, device=DEVICE, dtype=DTYPE)
target = LLMEngine(model_name="meta-llama/Llama-2-13b-chat-hf", batch_size=1, max_length=4096, device=DEVICE, dtype=DTYPE)
causal_mask = _make_causal_mask((1, 4096), DTYPE, DEVICE)
storage_ids = torch.arange(start=0, end=4096, device=DEVICE).long()
position_ids = torch.arange(start=0, end=4096, device=DEVICE).long().unsqueeze(0)

draft_proba_buffer = torch.zeros((GAMMA, VOCAB)).to(DEVICE)

NUM_STEPS = 0
ACCEPT_TOKENS = 0
data_id = 0
for patch in data:
    
    
    tokens = tokenizer.encode(patch["question"])
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    input_text = (
                    tokenizer.decode(
                    tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
    if local_rank == 0:
        print(" ".join(input_text), end=" ", flush=True)
    pos = 0
    generated_ids = []
    prompt_len = tokens.shape[1]
    dlogits = draft.inference(input_ids=tokens, storage_ids=storage_ids[:prompt_len], 
            position_ids=position_ids[:,:prompt_len], attention_mask=causal_mask[None, None, :, :][...,:prompt_len,:])

    
    tlogits = target.inference(input_ids=tokens, storage_ids=storage_ids[:prompt_len], 
            position_ids=position_ids[:,:prompt_len], attention_mask=causal_mask[None, None, :, :][...,:prompt_len,:])
    tlogits = tlogits[:,-1,:]
    
    tproba = F.softmax(tlogits/TEMP, dim=-1)
    sampled_tokens = tproba.multinomial(num_samples=1)
    num_generated_tokens = 0
    last_verified_position = prompt_len
    
    for s in range(STEP):
        
        
        for i in range(GAMMA):
            
            tokens = torch.cat([tokens, sampled_tokens], dim = -1)
            dlogits = draft.inference(
                input_ids=tokens[:, last_verified_position + i: last_verified_position + i + 1], 
                storage_ids=storage_ids[last_verified_position + i: last_verified_position + i + 1], 
                position_ids=position_ids[:,last_verified_position + i: last_verified_position + i + 1], 
                attention_mask=causal_mask[None, None, :, :][...,last_verified_position + i: last_verified_position + i + 1,:])
            
            
            dlogits = dlogits[:,-1,:]
            
            dproba = F.softmax(dlogits/TEMP, dim=-1)
            draft_proba_buffer[i] = dproba[0]
            sampled_tokens = torch.multinomial(dproba, num_samples=1)
            dist.broadcast(tensor=sampled_tokens, src=0)
        
        tlogits = target.inference(
                input_ids=tokens[:, last_verified_position: last_verified_position + GAMMA], 
                storage_ids=storage_ids[last_verified_position: last_verified_position + GAMMA], 
                position_ids=position_ids[:,last_verified_position: last_verified_position + GAMMA], 
                attention_mask=causal_mask[None, None, :, :][...,last_verified_position: last_verified_position + GAMMA,:])
        
        
        tlogits = tlogits[:,-GAMMA:,:].squeeze(0)
        tproba = F.softmax(tlogits/TEMP, dim=-1)
        tokens_for_verify = tokens[:, last_verified_position + 1: last_verified_position + GAMMA].T
        
        target_token_proba = tproba[:-1].gather(dim=-1, index=tokens_for_verify)
        draft_token_proba = draft_proba_buffer[:-1].gather(dim=-1, index=tokens_for_verify)

        
        accept_proba = target_token_proba / (draft_token_proba + 1e-4)
        r = torch.rand_like(accept_proba)
        accept = (r < accept_proba).T.squeeze()
        num_accept_tokens = torch.zeros(1).to(DEVICE).long()
        for idx, ac in enumerate(accept):
            if ac: num_accept_tokens += 1
            else: break
        
        dist.broadcast(tensor=num_accept_tokens, src=0)
        num_accept_tokens = num_accept_tokens.item()
        
        tokens = tokens[:,:last_verified_position + num_accept_tokens + 1]
        draft.llm.kv_cache.gather_kv_incremental(indices=list(range(last_verified_position, last_verified_position + num_accept_tokens + 1)), offset=last_verified_position)
        target.llm.kv_cache.gather_kv_incremental(indices=list(range(last_verified_position, last_verified_position + num_accept_tokens + 1)), offset=last_verified_position)
        last_verified_position = last_verified_position + num_accept_tokens + 1
        extra_proba = tproba[num_accept_tokens].unsqueeze(0)
        sampled_tokens = torch.multinomial(extra_proba, num_samples=1)
        dist.broadcast(tensor=sampled_tokens, src=0)
        if (torch._is_any_true(tokens[:,prompt_len:] == BOS) or torch._is_any_true(tokens[:,prompt_len:] == EOS)  or torch._is_any_true(tokens[:,prompt_len:] == PAD)):
            break
        
        generated_ids.extend(tokens[0][last_verified_position - num_accept_tokens - 1:].tolist())

        generated_text = (
                    tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False,
                )
                .strip()
                .split(" ")
                )
        now = len(generated_text) - 1
        if now > pos:
            if local_rank == 0:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now
        dist.barrier()
        NUM_STEPS = NUM_STEPS + 1
        ACCEPT_TOKENS = ACCEPT_TOKENS + num_accept_tokens + 1
    if local_rank == 0:
        print("\nData ID {}: decoding step: {}, large model step: {}, {}".format(data_id, ACCEPT_TOKENS, NUM_STEPS, ACCEPT_TOKENS / NUM_STEPS), flush=True)
    data_id += 1
    dist.barrier()
