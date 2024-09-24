import torch

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print('grad return nan')
                return False
            if torch.any(torch.isinf(param.grad)):
                print('grad return inf')
                return False
    return True