import torch
import torch.nn as nn
import math


class FeedForward(nn.Module):

    def __init__(self,k , d_model:int,d_ff:int)->None:
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)

    def forward(self):
        return self.linear2(torch.relu(self.linear1))

class InputEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocab_size:int) -> None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedded=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h :int, dropout:nn.Dropout )->None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.d_k=d_model//h

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.w_o=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    @staticmethod

    def attention(query, key, value, mask,dropout:nn.Dropout):
        d_k = query.shape[-1]
        attention_scores= (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        if mask is not None:
            attention_scores.maskeed_fill_(mask ==0, -1e9 )
        if dropout is not None:
            attention_scores=dropout(attention_scores)
    
        return (attention_scores @ value)
    
    def forward(self,query, key, value, mask):
        query= self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.view(query.shape[0])
        key = query.view(key.shape[0])
        value = value.view(value.shape[0])
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.droput)

        return self.w_o(x)

class LayerNormalization(nn.Module):
    
    def __init__(self) ->None:
        super().__init__()

    def forward(self,x):
        mean=x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return 
    
class PostionalEncoding(nn.Module):

    def __init__(self, d_model:int) -> None:
        super().__init__()
        self.d_model=d_model
        pe=torch.sin()
        pe=torch.cos()

        

    def forward(self):
        return 
    

class ResidualConnection(nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout=nn.Dropout()
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):

    def __init__(self,self_attention_block : MultiHeadAttentionBlock, feed_forward_block:FeedForward) ->None:
        super().__init__()
        self.self_attenton_block= self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections= nn.ModuleList([ResidualConnection() for _ in range(2)])
    def forward(self,x):
        x=self.residual_connections[0](x, lambda x:self.self_attenton_block(x,x,x))
        x=self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList) ->None:
        super().__init__()
        self.layers=layers
        self.norm= LayerNormalization()

    def forward(self,x, mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
