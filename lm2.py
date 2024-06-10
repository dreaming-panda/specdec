from llama_gpu import LLMEngine
import argparse
import time
import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
DEVICE = "cuda:0"
DTYPE = torch.float16
MAX_GEN_TOKENS = 128
TEMP = 0.6
TOPP = 1.0
TOPK = 32000
MAX_SAMPLE = 100
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
args = parser.parse_args()
print(args)
data = get_dataset("cnn", MAX_SAMPLE)
tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
draft = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=DTYPE, device_map="auto")
target = LlamaForCausalLM.from_pretrained(args.target, torch_dtype=DTYPE, device_map="auto")
total_acceptance_rate = torch.zeros(MAX_GEN_TOKENS).to(DEVICE)
num_samples = torch.zeros(MAX_GEN_TOKENS).to(DEVICE)
with torch.inference_mode():
    for patch in tqdm(data,total=len(data)):
        
        input = "You are a writter of cnn news. You will be shown a news and asked to write a summarization. Here is the news " + patch["article"] + " This is the end of the news. [INST] Please summarize the news in 200 words, Your Summarization is:"
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
        num_samples[:acceptance_rate.shape[1]] += 1
        total_acceptance_rate[:acceptance_rate.shape[1]] += acceptance_rate[0]
        


import matplotlib.pyplot as plt

avg_acceptance_rate = (total_acceptance_rate / num_samples).cpu().numpy()
plt.plot(list(range(MAX_GEN_TOKENS)), avg_acceptance_rate)
plt.savefig(args.model.split("/")[1] + "-" + args.target.split("/")[1] + ".pdf")


        


        
        
            

            
            
            

            
           
        
        


                
            


