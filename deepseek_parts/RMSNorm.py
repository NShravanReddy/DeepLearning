import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import math
import tqdm

from dataclasses import dataclass
from torch.nn  import RMSNorm

@dataclass
class ModelArgs:
    embeddings_dim=512

class Normalization(nn.Module):
    def __init__(self,embeddings_dim:int= ModelArgs.embeddings_dim):
        super().__init__()
        self.rmsnorm_layer=RMSNorm(embeddings_dim)

    def forward(self,x):
        x=self.rmsnorm_layer(x)
        return x

if __name__ == "__main__":
    dummy_input = torch.randn( ModelArgs.embeddings_dim)  # batch_size=2, seq_len=10
    norm = Normalization()
    output = norm(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("output:",output[1])
    print("Input:",dummy_input[1])