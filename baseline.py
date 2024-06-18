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
MAX_LEN = 256

data = get_dataset("mtbench", 100, num_fewshots=2)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
BOS = tokenizer.bos_token_id
if PAD == None:
    PAD = EOS
draft = LLMEngine(model_name="JackFram/llama-68m", batch_size=1, max_length=512, device=DEVICE, dtype=DTYPE)
target = LLMEngine(model_name="meta-llama/Llama-2-7b-chat-hf", batch_size=1, max_length=512, device=DEVICE, dtype=DTYPE)
draft.initialize_cuda_graph([1,2])
target.initialize_cuda_graph([1, GAMMA, GAMMA + 1])
causal_mask = _make_causal_mask((1, 512), DTYPE, DEVICE)
storage_ids = torch.arange(start=0, end=512, device=DEVICE).long()
position_ids = torch.arange(start=0, end=512, device=DEVICE).long().unsqueeze(0)

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
    generated_ids.extend(tokens[0][- 1: ].tolist())

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
    t1 = time.perf_counter()
    while num_total_tokens < MAX_LEN:
        
       
        
        dlogits = target.inference(
                input_ids=tokens[:, draft_kv_len: num_total_tokens], 
                storage_ids=storage_ids[draft_kv_len: num_total_tokens], 
                position_ids=position_ids[:,draft_kv_len: num_total_tokens], 
                attention_mask=causal_mask[None, None, :, :][...,draft_kv_len: num_total_tokens,:])
            
            
        draft_kv_len = num_total_tokens    
        dlogits = dlogits[:,-1,:]
            
        dproba = F.softmax(dlogits/TEMP, dim=-1)
        
        sampled_tokens = torch.multinomial(dproba, num_samples=1)
        tokens = torch.cat([tokens, sampled_tokens], dim = -1)
        num_total_tokens = tokens.shape[1]

        
        
        if torch._is_any_true(tokens[:,prompt_len:] == BOS) or torch._is_any_true(tokens[:,prompt_len:] == EOS)  or torch._is_any_true(tokens[:,prompt_len:] == PAD):
            break
        
        generated_ids.extend(tokens[0][- 1: ].tolist())

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
        
        LOCAL_NUM_STEPS = LOCAL_NUM_STEPS + 1

    t2 = time.perf_counter()
    print("\nData ID {}: decoding step: {}, latency: {} ms".format(data_id, LOCAL_NUM_STEPS, 1000 * (t2 - t1) / (num_total_tokens - prompt_len)), flush=True)
    







    



        
        

        
    





