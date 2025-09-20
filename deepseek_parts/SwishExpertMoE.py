import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import Swish
from Swish import Swish

from dataclasses import dataclass
@dataclass

class ModelArgs:
    block_size=256
    embeddings_dims=512
    device='mps' #'cuda:0'

class SwishExpertMoE(nn.Module):
    def __init__(self,
                 block_size:int=ModelArgs.block_size,
                 embeddings_dims:int=ModelArgs.embeddings_dims,
                 device=ModelArgs.device
                 ):
        super().__init__()
        self.hidden_dims=((embeddings_dims*2) * 4) //3
        self.swish= Swish(block_size=block_size,embeddings_dims=embeddings_dims)
        self.linear_layer1=nn.Linear(in_features=embeddings_dims,out_features=self.hidden_dims,bias=False,device=device)
        self.linear_layer2=nn.Linear(in_features=embeddings_dims,out_features=self.hidden_dims,bias=False,device=device)
        self.linear_layer3=nn.Linear(in_features=self.hidden_dims,out_features=embeddings_dims,bias=False,device=device)


    def forward(self,x):
        key=self.swish(self.linear_layer1(x))
        value=self.linear_layer2(x)
        res=torch.mul(key,value)
        out=self.linear_layer3(res)
        print("Output device:", out.device)
        return out
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    device = torch.device(ModelArgs.device)

    model = SwishExpertMoE().to(device)
    dummy_input = torch.randn(2, 10, ModelArgs.embeddings_dims).to(device)

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Input[0][0][:5]:", dummy_input[0, 0, :5])
    print("Output[0][0][:5]:", output[0, 0, :5])