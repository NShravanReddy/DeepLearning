import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from dataclasses import dataclass
@dataclass

class ModelArgs:
    block_size = 256
    embeddings_dim = 512
    device='cuda:0'

class Swish(nn.Module):
    def __init__(self,
        block_size:int=ModelArgs.block_size,
        embeddings_dims:int=ModelArgs.embeddings_dim,
        device=ModelArgs.device
        ):
        super().__init__()
        self.sig=torch.nn.Sigmoid()
    def forward(self,x):
        swish= x* self.sig(x)
        return swish
    
if __name__ == "__main__":
    dummy_input = torch.randn( ModelArgs.embeddings_dim)  # batch_size=2, seq_len=10
    norm = Swish()
    output = norm(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Input:",dummy_input[1])
    print("output:",output[1])