import torch
from torch import nn
import time

from dataset_tokenizer import load_packed_padded_data
from layers import *

assert torch.cuda.is_available()
device = torch.device('cuda:0')


if __name__ == "__main__":

    # load data
    print("loading dataset...")
    t_s = time.perf_counter()

    path_tokens_packed = './data/wikitext_tok_packed.h5'
    d_train, d_val, d_test = load_packed_padded_data(path_tokens_packed)

    print("d_train.shape: ", d_train.shape)
    print("d_val.shape: ", d_val.shape)
    print("d_test.shape: ", d_test.shape)

    d_train = d_train.to(device)
    d_val = d_val.to(device)
    d_test = d_test.to(device)

    print("done loading dataset, took ", time.perf_counter() - t_s)

    # llm params hardcoded for now
    V = 50257
    T = 512  # seq_length 1024
    C = 384  # embed_dim 768
    num_heads = 6  # 12
    hidden_dim = 256
    num_layers = 6  # 12

    # build llm
    print("building llm...")
    t_s = time.perf_counter()

    net = LLM(V, T, C, num_heads, num_layers, hidden_dim).to(device)
    num_p = sum(p.numel() for p in net.parameters())

    print("building llm done, total params: {}M, took {}s".format(
        num_p / 1.e6, time.perf_counter() - t_s))
