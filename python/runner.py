import torch
import os
import shutil
import tiktoken

from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser

from llm import LLM_training

assert torch.cuda.is_available()
device = torch.device('cuda:0')


parser = ArgumentParser()
parser.add_argument("--mode", choices=['train', 'infer'], default='train', help="train or infer")
parser.add_argument("--profile", type=bool, default=False, help="profile")


if __name__ == "__main__":
    args = parser.parse_args()
    mode = args.mode
    profile = args.profile
    
    # load data
    path_tokens_packed = './data/wikitext_tok_packed.h5'

    # llm params
    V = 50257
    T = 512 //4  # seq_length 1024
    C = 768 //2  # embed_dim 768
    num_heads = 6  # 12
    hidden_dim = 256
    num_layers = 6  # 12
    
    eot_token = V - 1
    B = 8
    
    # boilerplate
    dir = "runs/LLM_{}".format(mode)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    writer = SummaryWriter(dir)

    # llm trainer
    model = LLM_training(V, T, C, num_heads, num_layers, hidden_dim,
                           path_tokens_packed, B, eot_token,
                           writer, dir)
    
    # train
    if mode == 'train':
        num_epochs = 100
        num_grad_steps = 100
        model.train(num_epochs, num_grad_steps, profile)
    
    if mode == 'infer':
        ckpt = "runs/LLM_train/best.ckpt"
        
        # test seq
        test_seq = "I like dogs because "
        # test_seq = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series"
        
        # num token to predict
        num_tokens = 20
        
        # infer
        model.infer(test_seq, num_tokens, ckpt=ckpt)

