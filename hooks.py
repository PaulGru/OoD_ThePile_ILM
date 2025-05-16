# hooks.py
import torch

def nan_forward_hook(module, input, output):
    def check_tensor(t):
        return torch.is_tensor(t) and (torch.isnan(t).any() or torch.isinf(t).any())
    if isinstance(output, (list, tuple)):
        for idx, out in enumerate(output):
            if check_tensor(out):
                print(f"[Forward] NaN/Inf détecté dans {module} à l'index {idx}")
    else:
        if check_tensor(output):
            print(f"[Forward] NaN/Inf détecté dans {module}")

def nan_backward_hook(module, grad_input, grad_output):
    for idx, grad in enumerate(grad_output):
        if grad is not None and (torch.isnan(grad).any() or torch.isinf(grad).any()):
            print(f"[Backward] NaN/Inf détecté dans {module} à l'index {idx}")

def register_nan_forward_hooks(model):
    for module in model.modules():
        module.register_forward_hook(nan_forward_hook)

def register_nan_backward_hooks(model):
    for module in model.modules():
        module.register_full_backward_hook(nan_backward_hook)
