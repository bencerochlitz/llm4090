
import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, C, num_heads, hidden_dim):
        super().__init__()

        self.ln_1 = nn.LayerNorm(C)
        self.dense_1 = nn.Linear(C, C * 3)
        self.att = nn.MultiheadAttention(C, num_heads)
        self.dense_2 = nn.Linear(C, C)
        self.ln_2 = nn.LayerNorm(C)
        # feedforward layer
        self.ff = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.GELU(),
            nn.Linear(C * 4, C)
        )

    def forward(self, x):
        y = self.ln_1(x)
        y = self.dense_1(y)
        y = self.att(y)
        x = self.dense_2(y) + x
        y = self.ln_2(x)
        y = self.ff(y) + x
        return y


class LLM(nn.Module):

    def __init__(self, V, T, C, num_heads, num_layers, hidden_dim):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(V, C)
        self.pos_embedding = nn.Embedding(T, C)

        self.transformer = nn.Sequential()
        for _ in range(num_layers):
            self.transformer.append(TransformerBlock(C, num_heads, hidden_dim))

        self.ln = nn.LayerNorm(C)
        self.dense = nn.Linear(C, V)

    def forward(self, x, ids):
        x = self.tok_embedding(x) + self.pos_embedding(ids)
        x = self.transformer(x)
        x = self.ln(x)
        x = self.dense(x)
        return x
