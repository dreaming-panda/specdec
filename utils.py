import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from model import Transformer

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

def load_model(checkpoint_path, device, precision):

    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def model_forward(model :Transformer, x :torch.Tensor, input_pos :torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    return model(x, input_pos, cache_pos, attention_mask)

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, cache_pos :torch.Tensor = None, attention_mask: torch.Tensor = None) -> torch.Tensor:
    # input_pos: [B, S]
    return model(x, input_pos, cache_pos, attention_mask)
    