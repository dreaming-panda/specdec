from llama_gpu import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
DEVICE = "cuda:6"
DTYPE = torch.float16
MAX_GEN_TOKENS = 64
TEMP = 0.6
TOPP = 1.0
TOPK = 32000
MAX_SAMPLE = 100
data = get_dataset("gsm8k", 10)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
draft = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=DTYPE, device_map="auto")
target = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=DTYPE, device_map="auto")
total_acceptance_rate = 0.0
num_samples = 0
with torch.inference_mode():
    for patch in tqdm(data,total=len(data)):
        
        input = patch["question"]
        tokens = tokenizer.encode(input)
        tokens = torch.LongTensor(tokens).unsqueeze(0).to('cuda')
        
        input_sentence = tokenizer.decode(tokens[0])
        
        initial_len = tokens.shape[1]
        outputs = target.generate(tokens, do_sample=True, temperature=TEMP, top_p=1.0, top_k=TOPK, max_new_tokens=MAX_GEN_TOKENS)

        output_sentence = tokenizer.decode(outputs[0])
        
        
        output_len = outputs.shape[1]

        target_logits = target(outputs).logits[:,initial_len:,:]
        draft_logits = draft(outputs).logits[:,initial_len:,:]

        target_proba = F.softmax(target_logits/TEMP, dim=-1).unsqueeze(-1)
        draft_proba = F.softmax(draft_logits/TEMP, dim=-1).unsqueeze(-1)

        probas = torch.cat([target_proba, draft_proba], dim=-1)
        probas = torch.min(probas, dim=-1).values
        acceptance_rate = probas.sum(dim=-1)
        num_samples += (output_len - initial_len)
        total_acceptance_rate += acceptance_rate.sum()
        print(acceptance_rate.sum() / (output_len - initial_len), flush=True)


print(total_acceptance_rate / num_samples)


        


        
        
            

            
            
            

            
           
        
        


                
            


