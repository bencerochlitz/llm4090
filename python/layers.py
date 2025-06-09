
import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TransformerBlock(nn.Module):
    def __init__(self, C, num_heads, hidden_dim):
        super().__init__()
        
        self.C = C

        self.ln_1 = nn.LayerNorm(C)
        self.dense_1 = nn.Linear(C, C * 3)
        self.att = nn.MultiheadAttention(C, num_heads, batch_first=True)
        self.dense_2 = nn.Linear(C, C)
        self.ln_2 = nn.LayerNorm(C)
        # feedforward layer
        self.ff = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.GELU(),
            nn.Linear(C * 4, C)
        )

    def forward(self, x, att_mask):
        y = self.ln_1(x)
        y = self.dense_1(y)
        
        # split
        q, k, v = y[..., :self.C], y[..., self.C:self.C*2], y[..., self.C*2:]
        
        # attention mask for packed sequence
        y, _ = self.att(q, k, v, attn_mask=att_mask, need_weights=False, is_causal=False)
        
        x = self.dense_2(y) + x
        y = self.ln_2(x)
        y = self.ff(y) + x
        return y


class LLM(nn.Module):
    def __init__(self, V, T, C, num_heads, num_layers, hidden_dim):
        super().__init__()
        
        self.tok_embedding = nn.Embedding(V, C)
        self.pos_embedding = nn.Embedding(T, C)

        self.transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(TransformerBlock(C, num_heads, hidden_dim))

        self.ln = nn.LayerNorm(C)
        self.dense = nn.Linear(C, V)
        
        # for inference
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x, ids, att_mask):
        x = self.tok_embedding(x) + self.pos_embedding(ids)
        
        for transformer in self.transformers:
            # print("transformer in: ", x.shape)
            x = transformer(x, att_mask)
            
        x = self.ln(x)
        x = self.dense(x)
        return x
    
    @torch.jit.export
    def infer(self, x, ids, att_mask):
        x = self(x, ids, att_mask)
        x = self.sm(x)
        # print(x.shape)
        # remove batch dim
        x = x.squeeze(0)
        x = torch.multinomial(x, 1).squeeze(-1)
        return x
