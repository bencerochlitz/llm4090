import torch
import os
import shutil

from torch.utils.tensorboard import SummaryWriter

import time

from llm import LLM_training

assert torch.cuda.is_available()
device = torch.device('cuda:0')


if __name__ == "__main__":
    # load data
    path_tokens_packed = './data/wikitext_tok_packed.h5'

    # llm params hardcoded for now
    V = 50257
    T = 512 //2  # seq_length 1024
    C = 768 //2  # embed_dim 768
    num_heads = 6  # 12
    hidden_dim = 256
    num_layers = 6  # 12
    
    eot_token = V - 1
    B = 8
    
    # boilerplate
    mode = 'train'
    dir = "runs/LLM_{}".format(mode)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    writer = SummaryWriter(dir)

    # llm trainer
    trainer = LLM_training(V, T, C, num_heads, num_layers, hidden_dim,
                           path_tokens_packed, B, eot_token,
                           writer)
    
    # run
    num_epochs = 100
    num_grad_steps = 100
    trainer.train(num_epochs, num_grad_steps)
    

