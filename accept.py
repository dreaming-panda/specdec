from llama_gpu import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
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
STEP = 128
GAMMA = 5
TEMP = 0.6
VOCAB = 32000
data = get_dataset("gsm8k")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
draft = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", torch_dtype=DTYPE).to(DEVICE)
target = LlamaForCausalLM.from_pretrained("JackFram/llama-160m", torch_dtype=DTYPE).to(DEVICE)
draft_proba_buffer = torch.zeros((GAMMA, VOCAB)).to(DEVICE)
draft_tokens_buffer = torch.zeros((1, GAMMA + 1)).to(DEVICE).long()
with torch.inference_mode():
    for patch in data:
        tokens = tokenizer.encode(patch["question"])
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
        
        draft_output = draft(input_ids=tokens, use_cache=True, past_key_values=None)
        target_output = target(input_ids=tokens, use_cache=True, past_key_values=None)
        tlogits = target_output.logits
        tlogits = tlogits[:,-1,:]
        tproba = F.softmax(tlogits/TEMP, dim=-1)
        new_input = tproba.multinomial(num_samples=1)
        num_generated_tokens = 1
        draft_past_key_values = draft_output.past_key_values
        target_past_key_values = target_output.past_key_values
        text_tokens = tokens.tolist()
        for _ in range(STEP):
            draft_tokens_buffer[:,0] = new_input
            for i in range(GAMMA):
                draft_output = draft(input_ids=new_input, use_cache=True, past_key_values=draft_past_key_values)
                dlogits = draft_output.logits
                dlogits = dlogits[:,-1,:]
                dproba = F.softmax(dlogits/TEMP, dim=-1)
                new_input = torch.multinomial(dproba, num_samples=1)
                draft_tokens_buffer[:,i + 1] = new_input
                draft_proba_buffer[i] = dproba.squeeze()
                
                #print(dproba[0][new_input[0][0]])
                
                draft_past_key_values = draft_output.past_key_values

            draft_output = draft(input_ids=new_input, use_cache=True, past_key_values=draft_past_key_values)
            draft_past_key_values = draft_output.past_key_values
            
            target_output = target(input_ids=draft_tokens_buffer, use_cache=True, past_key_values=target_past_key_values)
            tlogits = target_output.logits

            tproba = F.softmax(tlogits/TEMP, dim=-1).squeeze()[:GAMMA]
            candidates = draft_tokens_buffer[:,-GAMMA:].T
            
            target_token_proba = tproba.gather(dim=-1, index=candidates).T
            

            draft_token_proba = draft_proba_buffer.gather(dim=-1, index=candidates).T
            
            ratio = target_token_proba / (draft_token_proba + 1e-4)
            r = torch.rand_like(ratio)
            accept = (ratio > r).squeeze().tolist()
            last_accept_position = -1
            
            for idx, ac in enumerate(accept):
                if not ac: break
                last_accept_position = idx
            
            

            
            
            

            
           
        
        


                
            


           
            


        
    





