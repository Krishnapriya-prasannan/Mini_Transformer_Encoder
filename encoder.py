import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==== Step 1: Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ==== Step 2: Multi-Head Self-Attention ====
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = attn_weights @ v
        context = context.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(context)

# ==== Step 3: Transformer Encoder Block ====
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

# ==== Step 4: Mini BERT Model ====
class MiniBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x

# ==== Step 5: Test with Synthetic Data ====
if __name__ == "__main__":
    vocab_size = 100
    seq_len = 10
    batch_size = 32
    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    num_layers = 2
    max_len = 20

    model = MiniBERT(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len)
    input_data = torch.randint(0, vocab_size, (batch_size, seq_len))  # Fake token IDs
    output = model(input_data)

    print("Input shape: ", input_data.shape)
    print("Output shape:", output.shape)  # [batch_size, seq_len, embed_dim]
