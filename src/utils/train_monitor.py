# Monitor training progress
import torch
from tqdm import tqdm

def format_mem():
    if not torch.cuda.is_available():
        return ""
    m = torch.cuda.max_memory_allocated() / (1024**2)
    return f" | max_mem {m:.0f}MB"

def step_bar(iterable, desc="train"):
    return tqdm(iterable, desc=desc, leave=False, smoothing=0.1, dynamic_ncols=True)

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().data.norm(2).item() ** 2
    return total ** 0.5
