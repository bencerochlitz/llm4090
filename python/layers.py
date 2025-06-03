
import torch
from torch import nn


class GPT2Layer(nn.Module):
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

    def __init__(self, V, C, num_heads, num_layers, hidden_dim):
        super().__init__()

        self.transformer = nn.Sequential()
        for _ in range(num_layers):
            self.transformer.append(GPT2Layer(C, num_heads, hidden_dim))

        self.ln = nn.LayerNorm(C)
        self.dense = nn.Linear(C, V)

    def forward(self, x):
        x = self.transformer(x)
        x = self.ln(x)
        x = self.dense(x)
