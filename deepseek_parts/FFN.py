import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from dataclasses import dataclass

@dataclass

class ModelArgs:
    block_size=256
    vocab_size=10
    embeddings_dims=512
    dropout=0.1
    device='mps'

class FFN(nn.Module):

    def __init__(self,
                 embeddings_dims=ModelArgs.embeddings_dims,
                 block_size:int = ModelArgs.block_size,
                 vocab_size=ModelArgs.vocab_size,
                 dropout=ModelArgs.dropout,
                 device=ModelArgs.device):
        super().__init__()

        self.linear_layer=nn.Linear(in_features=embeddings_dims,out_features=embeddings_dims,dtype=torch.float32,device=device)
        self.linear_layer2=nn.Linear(in_features=embeddings_dims,out_features=embeddings_dims,dtype=torch.float32,device=device)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        x=self.linear_layer(x)
        x=F.gelu(x)
        x=self.linear_layer2(x)
        x=F.gelu(x)
        return x         
    

if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    device = torch.device(ModelArgs.device)

    model =FFN().to(device)
    dummy_input = torch.randn(2, 10, ModelArgs.embeddings_dims).to(device)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Input[0][0][:5]:", dummy_input[0, 0, :5])
    print("Output[0][0][:5]:", output[0, 0, :5])