CUDA_VISIBLE_DEVICES=8 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-7b-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B --target meta-llama/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=8 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-13b-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-1.3B --target meta-llama/Llama-2-13b-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B --target meta-llama/Llama-2-13b-hf

CUDA_VISIBLE_DEVICES=8 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT --target meta-llama/Llama-2-7b-chat-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT --target meta-llama/Llama-2-7b-chat-hf


CUDA_VISIBLE_DEVICES=8 python lm2.py --model JackFram/llama-68m --target meta-llama/Llama-2-13b-chat-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT --target meta-llama/Llama-2-13b-chat-hf
CUDA_VISIBLE_DEVICES=8 python lm2.py --model princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT --target meta-llama/Llama-2-13b-chat-hf