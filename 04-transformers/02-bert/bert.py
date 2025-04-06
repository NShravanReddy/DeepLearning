import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):

    def __init__(self,features:int,eps:float=1e-6)->None:
        super().__init__()
        self.eps=eps
        self.alpha= nn.Parameter(torch.ones(features))
        self.beta= nn.Parameter(torch.zeros(features))
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim= True)
        std  = x.std(dim=-1, keepdim=True)
        return self.alpha *(x-mean)/(std+self.eps)+self.bias
    
class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int, vocab_size:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__():
        super().__init__()

    def forward(self,x):
        

##unable to understand the postional encoding

