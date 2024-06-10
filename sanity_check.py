import torch
from convert_dataset import get_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F
DEVICE = "cuda:1"
DTYPE = torch.float16
STEP = 128
GAMMA = 5
TEMP = 0.6
VOCAB = 32000

draft = LlamaForCausalLM.from_pretrained("JackFram/llama-68m", torch_dtype=DTYPE).to(DEVICE)
target = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=DTYPE).to(DEVICE)

input_ids = torch.tensor([[    1,  2627,   300, 30010, 29879,   868,  4684,  6568, 29871, 29896,
         29953, 29808,   639,  2462, 29889,  2296,   321,  1446,  2211,   363,
         26044,  1432,  7250,   322,   289,  6926,   286,  3096,  1144,   363,
           902,  7875,  1432,  2462,   411,  3023, 29889,  2296,   269, 10071,
           278, 21162,   472,   278,  2215, 13269, 29915,  9999, 14218,   363,
           395, 29906,   639, 10849,   868,   384, 19710, 29889,  1128,  1568,
           297, 17208,   947,  1183,  1207,  1432,  2462,   472,   278,  2215,
         13269, 29915,  9999, 29973,    13, 12024, 29915, 29879]],
       device='cuda:1').long()

draft = draft.eval()
target = target.eval()
with torch.inference_mode():
    logits = target(input_ids = input_ids).logits
    print(logits)
