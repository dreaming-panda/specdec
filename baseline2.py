from llama_gpu import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer
import torch.nn.functional as F
from pathlib import Path
from backend import LMBackend

DEVICE = "cuda:0"
DTYPE = torch.float16
STEP = 32
GAMMA = 3
TEMP = 0.3
VOCAB = 32000
MAX_LEN = 1024
VERBOSE = True
torch.cuda.set_device(DEVICE)
data = get_dataset("mtbench", 100, num_fewshots=2)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
EOS = tokenizer.eos_token_id
PAD = tokenizer.pad_token_id
BOS = tokenizer.bos_token_id
if PAD == None:
    PAD = EOS



target = LMBackend(dtype=DTYPE, device=DEVICE)

target.load_model(Path("/home/zhuominc/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"))

target.compile()
#draft.initialize_cuda_graph([1,2])

target.setup_caches(max_batch_size=1, max_seq_length=MAX_LEN)
causal_mask = torch.tril(torch.ones(MAX_LEN, MAX_LEN, dtype=torch.bool, device=DEVICE))
storage_ids = torch.arange(start=0, end=MAX_LEN, device=DEVICE).long()
position_ids = torch.arange(start=0, end=MAX_LEN, device=DEVICE).long()
tokens = torch.zeros((1, MAX_LEN), device=DEVICE, dtype=torch.long)


for data_id, patch in enumerate(data):
    
    tokens.zero_()
    patch_tokens = tokenizer.encode(patch)
    patch_tokens = torch.LongTensor(patch_tokens).unsqueeze(0).to(DEVICE)
    prompt_len = patch_tokens.shape[1]

    tokens[:,:prompt_len] = patch_tokens
    input_text = (
                    tokenizer.decode(
                    tokens[0][:prompt_len],
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
    tlogits = target.encode(input_ids=tokens[:,:prompt_len], storage_ids=None, 
            position_ids=position_ids[:prompt_len], attention_mask=None)
    tlogits = tlogits[:,-1,:]
    
    tproba = F.softmax(tlogits/TEMP, dim=-1)
    sampled_tokens = tproba.multinomial(num_samples=1)
    
    tokens[:,prompt_len] = sampled_tokens.squeeze(0)
    
    draft_kv_len = prompt_len
    
    num_total_tokens = prompt_len + 1
    # cache_id = storage_ids[draft_kv_len: num_total_tokens]
    # pos_id = position_ids[draft_kv_len: num_total_tokens]
    # dec_mask = causal_mask[draft_kv_len: num_total_tokens,:][None, None, :, :]
    #input_ids = sampled_tokens

    # print(input_ids.shape)
    
    t1 = time.perf_counter()
    while num_total_tokens < MAX_LEN - 32:
        
        tlogits = target.inference(
                input_ids=tokens[:,draft_kv_len: num_total_tokens], 
                storage_ids=storage_ids[draft_kv_len: num_total_tokens], 
                position_ids=position_ids[draft_kv_len: num_total_tokens], 
                attention_mask=causal_mask[draft_kv_len: num_total_tokens,:][None, None, :, :])
        
        tlogits = tlogits[:,-1,:]
    
        tproba = F.softmax(tlogits/TEMP, dim=-1)
        sampled_tokens = tproba.multinomial(num_samples=1)
    
        tokens[:,num_total_tokens] = sampled_tokens.squeeze(0)
        draft_kv_len = num_total_tokens
        num_total_tokens = num_total_tokens + 1
        if torch._is_any_true(tokens[:,prompt_len:num_total_tokens] == BOS) or torch._is_any_true(tokens[:,prompt_len:num_total_tokens] == EOS)  or torch._is_any_true(tokens[:,prompt_len:num_total_tokens] == PAD):
                break
            
            

    t2 = time.perf_counter()
    if VERBOSE:
        generated_ids.extend(tokens[0][prompt_len: num_total_tokens].tolist())

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
    print("\nData ID {} : latency: {} ms".format(data_id, 1000 * (t2 - t1) / (num_total_tokens - prompt_len)), flush=True)
    
