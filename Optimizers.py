import torch
from transformers import AdamW
def build_Adam(params,lr,weight_decay):
    return torch.optim.Adam(params=params,lr=lr,weight_decay=weight_decay)

def build_AdamW(params, lr, weight_decay=0):
    return AdamW(params=params, lr=lr, weight_decay=weight_decay)

optimizers = {
    "Adam":build_Adam,
    "AdamW":build_AdamW,
    
}