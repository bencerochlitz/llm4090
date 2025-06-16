
import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import math

class TransformerBlock(nn.Module):
    def __init__(self, C, T, num_heads):
        super().__init__()
        
        self.C = C

        self.ln_1 = nn.LayerNorm(C)
        self.dense_1 = nn.Linear(C, C * 3)
        self.att = nn.MultiheadAttention(C, num_heads, dropout=0.1, batch_first=True)
        self.dense_2 = nn.Linear(C, C)
        self.ln_2 = nn.LayerNorm(C)
        # feedforward layer
        self.ff_dense_1 = nn.Linear(C, C * 4)
        self.gelu = nn.GELU()
        self.ff_dense_2 = nn.Linear(C * 4, C)
        
        self.att_mask = torch.tril(torch.ones((T, T)), diagonal=0)
        self.att_mask.requires_grad = False

    def forward(self, x):
        y = self.ln_1(x)
        y = self.dense_1(y)
        
        # split
        q, k, v = y[..., :self.C], y[..., self.C:self.C*2], y[..., self.C*2:]
        
        # attention mask for packed sequence
        y, _ = self.att(q, k, v, attn_mask=self.att_mask, need_weights=False, is_causal=True)
        
        # projection
        x = self.dense_2(y) + x
        y = self.ln_2(x)
        
        # feedforward
        y = self.ff_dense_1(y)
        y = self.gelu(y)
        y = self.ff_dense_2(y) + x
        
        return y


class LLM(nn.Module):
    def __init__(self, V, T, C, num_heads, num_layers):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(V, C)
        self.pos_embedding = nn.Embedding(T, C)

        self.transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(TransformerBlock(C, T, num_heads))

        # # adding dense_1 down projection and // 12 gives a 33% perf boost
        # self.dense_1 = nn.Linear(C, C // 12)
        # self.ln = nn.LayerNorm(C // 12)
        # self.dense_2 = nn.Linear(C // 12, V)
        
        self.ln = nn.LayerNorm(C)
        self.dense = nn.Linear(C, V)
        
        # for inference
        self.sm = nn.Softmax(dim=-1)
        
        with torch.no_grad():
            # weight init suggestions are from Claude
            for layer in self.children():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(layer.bias)
                    
                # these are default anyway
                if isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
            # dense projection + feedforward projection layer scaling
            scaling = 1.0 / math.sqrt(num_layers)
            for tr_block in self.transformers:
                tr_block.dense_2.weight[:] *= scaling
                tr_block.ff_dense_2.weight[:] *= scaling

            # token embedding
            nn.init.normal_(self.tok_embedding.weight, mean=0.0, std=0.02)
            
            # position embedding
            nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.01)

    def forward(self, x, ids):
        x = self.tok_embedding(x) + self.pos_embedding(ids)
        
        for transformer in self.transformers:
            # print("transformer in: ", x.shape)
            x = transformer(x)
            
        # x = self.dense_1(x)
        # x = self.ln(x)
        # x = self.dense_2(x)
        
        x = self.ln(x)
        x = self.dense(x)
        
        return x
    
    def infer(self, x, ids):
        x = self(x, ids)
        x = self.sm(x)
        # print(x.shape)
        # remove batch dim
        x = x.squeeze(0)
        x = torch.multinomial(x, 1).squeeze(-1)
        return x
