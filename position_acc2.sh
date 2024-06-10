CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model meta-llama/Llama-2-7b-chat-hf --target meta-llama/Llama-2-70b-chat-hf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-70b-hf

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT --target meta-llama/Llama-2-70b-chat-hf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B --target meta-llama/Llama-2-70b-hf

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-70b-chat-hf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-70b-hf

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model neuralmagic/Llama-2-7b-pruned70-retrained --target meta-llama/Llama-2-70b-hf
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python lm2.py --model neuralmagic/Llama-2-7b-ultrachat200k-pruned_70 --target meta-llama/Llama-2-70b-chat-hf