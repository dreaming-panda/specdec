from llama_gpu import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer
import torch.nn.functional as F
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

DEVICE = "cuda:0"
DTYPE = torch.float16
STEP = 32
GAMMA = 3
TEMP = 0.3
VOCAB = 32000
MAX_LEN = 1024
VERBOSE = False
data = get_dataset("mtbench", 100, num_fewshots=2)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
BOS = tokenizer.bos_token_id
if PAD == None:
    PAD = EOS
draft = LLMEngine(model_name="JackFram/llama-68m", batch_size=1, max_length=1536, device=DEVICE, dtype=DTYPE)
target = LLMEngine(model_name="meta-llama/Llama-2-7b-chat-hf", batch_size=1, max_length=1536, device=DEVICE, dtype=DTYPE)
draft.initialize_cuda_graph([1,2])
target.initialize_cuda_graph([GAMMA, GAMMA + 1])
causal_mask = _make_causal_mask((1, 1536), DTYPE, DEVICE)
storage_ids = torch.arange(start=0, end=1536, device=DEVICE).long()
position_ids = torch.arange(start=0, end=1536, device=DEVICE).long().unsqueeze(0)

draft_proba_buffer = torch.zeros((GAMMA, VOCAB)).to(DEVICE)

NUM_STEPS = 0
ACCEPT_TOKENS = 0
for data_id, patch in enumerate(data):
    
    
    tokens = tokenizer.encode(patch)
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
    
    print(" ".join(input_text), end=" ", flush=True)
    pos = 0
    generated_ids = []
    prompt_len = tokens.shape[1]
    if prompt_len > MAX_LEN:
        continue
    dlogits = draft.inference(input_ids=tokens, storage_ids=storage_ids[:prompt_len], 
            position_ids=position_ids[:,:prompt_len], attention_mask=causal_mask[None, None, :, :][...,:prompt_len,:])
    
    tlogits = target.inference(input_ids=tokens, storage_ids=storage_ids[:prompt_len], 
            position_ids=position_ids[:,:prompt_len], attention_mask=causal_mask[None, None, :, :][...,:prompt_len,:])
    tlogits = tlogits[:,-1,:]
    
    tproba = F.softmax(tlogits/TEMP, dim=-1)
    sampled_tokens = tproba.multinomial(num_samples=1)
    
    tokens = torch.cat([tokens, sampled_tokens], dim = -1)
    num_generated_tokens = 0
    last_verified_position = prompt_len
    draft_kv_len = prompt_len
    LOCAL_NUM_STEPS = 0
    LOCAL_ACCEPT_TOKENS = 0
    num_total_tokens = tokens.shape[1]
    r = torch.rand((GAMMA, 1), dtype=DTYPE).to(DEVICE)
    t1 = time.perf_counter()
    while num_total_tokens < MAX_LEN:
        
        for i in range(GAMMA):
            
            dlogits = draft.inference(
                input_ids=tokens[:, draft_kv_len: num_total_tokens], 
                storage_ids=storage_ids[draft_kv_len: num_total_tokens], 
                position_ids=position_ids[:,draft_kv_len: num_total_tokens], 
                attention_mask=causal_mask[None, None, :, :][...,draft_kv_len: num_total_tokens,:])
            
            
            draft_kv_len = num_total_tokens    
            dlogits = dlogits[:,-1,:]
            
            dproba = F.softmax(dlogits/TEMP, dim=-1)
            draft_proba_buffer[i] = dproba[0]
            sampled_tokens = torch.multinomial(dproba, num_samples=1)
            tokens = torch.cat([tokens, sampled_tokens], dim = -1)
            num_total_tokens = tokens.shape[1]

        
        tlogits = target.inference(
                input_ids=tokens[:, last_verified_position: last_verified_position + GAMMA + 1], 
                storage_ids=storage_ids[last_verified_position: last_verified_position + GAMMA + 1], 
                position_ids=position_ids[:,last_verified_position: last_verified_position + GAMMA + 1], 
                attention_mask=causal_mask[None, None, :, :][...,last_verified_position: last_verified_position + GAMMA + 1,:])
        
        
        tlogits = tlogits[:,-GAMMA - 1:,:].squeeze(0)
        tproba = F.softmax(tlogits/TEMP, dim=-1)
        
        tokens_for_verify = tokens[:, last_verified_position + 1: last_verified_position + GAMMA + 1].T
        
        target_token_proba = tproba[:-1].gather(dim=-1, index=tokens_for_verify)

        
        draft_token_proba = draft_proba_buffer.gather(dim=-1, index=tokens_for_verify)

        
        accept_proba = target_token_proba / (draft_token_proba + 1e-4)
        
        
        accept = (r < accept_proba).T.squeeze()
        num_accept_tokens = 0
        for idx, ac in enumerate(accept):
            if ac: num_accept_tokens += 1
            else: break
        

        tokens = tokens[:,:last_verified_position + num_accept_tokens + 1]
        if num_accept_tokens < GAMMA:
            #draft.llm.kv_cache.gather_kv_incremental(indices=list(range(last_verified_position, last_verified_position + num_accept_tokens + 1)), offset=last_verified_position)
            #target.llm.kv_cache.gather_kv_incremental(indices=list(range(last_verified_position, last_verified_position + num_accept_tokens + 1)), offset=last_verified_position)
            last_verified_position = last_verified_position + num_accept_tokens + 1
            draft_kv_len = last_verified_position
            extra_proba = tproba[num_accept_tokens].unsqueeze(0)
            sampled_tokens = torch.multinomial(extra_proba, num_samples=1)
            tokens = torch.cat([tokens, sampled_tokens], dim = -1)
            num_total_tokens = tokens.shape[1]
        else:
            # draft.inference(
            #     input_ids=tokens[:, last_verified_position + GAMMA: last_verified_position + GAMMA + 1], 
            #     storage_ids=storage_ids[last_verified_position + GAMMA: last_verified_position + GAMMA + 1], 
            #     position_ids=position_ids[:,last_verified_position + GAMMA: last_verified_position + GAMMA + 1], 
            #     attention_mask=causal_mask[None, None, :, :][...,last_verified_position + GAMMA: last_verified_position + GAMMA + 1,:])

            last_verified_position = last_verified_position + num_accept_tokens + 1
            extra_proba = tproba[num_accept_tokens].unsqueeze(0)
            sampled_tokens = torch.multinomial(extra_proba, num_samples=1)
            tokens = torch.cat([tokens, sampled_tokens], dim = -1)
            num_total_tokens = tokens.shape[1]
        
        if torch._is_any_true(tokens[:,prompt_len:] == BOS) or torch._is_any_true(tokens[:,prompt_len:] == EOS)  or torch._is_any_true(tokens[:,prompt_len:] == PAD):
            break
        if VERBOSE:
            generated_ids.extend(tokens[0][last_verified_position - num_accept_tokens - 1: -1].tolist())

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
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now
            if "[/INST]" in generated_text or "[INST]" in generated_text:
                break
        #dist.barrier()
        NUM_STEPS = NUM_STEPS + 1
        ACCEPT_TOKENS = ACCEPT_TOKENS + num_accept_tokens + 1
        LOCAL_NUM_STEPS = LOCAL_NUM_STEPS + 1
        LOCAL_ACCEPT_TOKENS = LOCAL_ACCEPT_TOKENS + num_accept_tokens + 1

    t2 = time.perf_counter()
    print("\nData ID {}: decoding step: {}, large model step: {}, {}, latency: {} ms".format(data_id, LOCAL_ACCEPT_TOKENS, LOCAL_NUM_STEPS, LOCAL_ACCEPT_TOKENS / LOCAL_NUM_STEPS, 1000 * (t2 - t1) / (num_total_tokens - prompt_len)), flush=True)







    



        
        

        
    





