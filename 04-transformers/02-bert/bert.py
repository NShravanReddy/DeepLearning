import torch
import torch.nn as nn
import math

# === Embedding Layer ===
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# === Positional Encoding ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# === Multi-Head Attention ===
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, L, D = q.shape
        H = self.h

        def transform(x, linear):
            x = linear(x)  # [B, L, D]
            x = x.view(B, L, H, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
            return x

        q = transform(q, self.q_linear)
        k = transform(k, self.k_linear)
        v = transform(v, self.v_linear)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, L]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = attn @ v  # [B, H, L, d_k]
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        return self.out_proj(output)

# === FeedForward Block ===
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# === Encoder Block ===
class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn = self.self_attention_block(x, x, x, src_mask)
        x = self.ln1(x + self.dropout(attn))
        ff = self.feed_forward_block(x)
        x = self.ln2(x + self.dropout(ff))
        return x

# === Encoder ===
class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

# === Projection Layer (e.g., for MLM or classification) ===
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)

# === Final Transformer Class (Encoder-only) ===
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: InputEmbeddings, src_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def forward(self, src, src_mask):
        encoded = self.encode(src, src_mask)
        return self.project(encoded)

    def project(self, x):
        return self.projection_layer(x)

# === Build Transformer (no decoder) ===
def build_transformer(src_vocab_size: int, src_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    projection_layer = ProjectionLayer(d_model, src_vocab_size)

    transformer = Transformer(encoder, src_embed, src_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
