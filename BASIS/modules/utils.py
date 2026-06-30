import torch

def _any_tensor(*args):
    return any(torch.is_tensor(arg) for arg in args)

def _as_tensors(*args, dtype=None, device=None):
    tensors = []
    for arg in args:
        if torch.is_tensor(arg):
            tensors.append(arg.to(dtype=dtype, device=device))
        else:
            tensors.append(torch.tensor(arg, dtype=dtype, device=device))
    return tensors