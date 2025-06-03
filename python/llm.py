import torch
from torch import nn

from layers import *

from datasets import load_dataset

assert torch.cuda.is_available()
device = torch.device('cuda:0')


if __name__ == "__main__":

    # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    V = 50257
    B = 512  # seq_length 1024
    C = 384  # embed_dim 768
    num_heads = 6  # 12
    hidden_dim = 256
    num_layers = 6  # 12

    print("building llm...")
    net = LLM(V, C, num_heads, num_layers, hidden_dim).to(device)
    num_p = sum(p.numel() for p in net.parameters())
    print("building llm done, total params: {}M".format(num_p / 1.e6))
