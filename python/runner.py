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
    path_tokens_packed = './data/dataset_tok_packed.h5'

    # llm params
    # the wikitext sequnces are pretty short anyway, so saving compute here
    T = 512  # seq_length 1024
    C = 768 //2  # embed_dim 768
    num_heads = 6  # 12
    num_layers = 6  # 12
    
    B = 8
    num_batches = 16
    
    # boilerplate
    dir = "runs/LLM_{}".format(mode)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    writer = SummaryWriter(dir)

    # llm trainer
    model = LLM_training(T, C, num_heads, num_layers,
                         B, num_batches, writer, dir)
    
    # train
    if mode == 'train':
        # NOTE: learning rate scheduler is hardcoded for now
        num_epochs = 500
        num_grad_steps = 100
        model.set_lr_scheduler(num_epochs * num_grad_steps)
        model.train(num_epochs, num_grad_steps, profile)
    
    if mode == 'infer':
        ckpt = "runs/LLM_train/best.ckpt"
        
        # test seq - not in training or val set
        test = model.loader.load_ready_data(0.1, begin=False)
        test0 = test[-1][0:128]
        
        # TEXT:
        #  Western ways to help LGBT people in the Middle East establish and
        #  promote their rights. Firas is able to live more freely here in
        #  New York City and was excited to tell me about an upcoming birthday
        #  weekend he had planned for his boyfriend. However, he struggles
        #  with having to conceal his identity among colleagues, given his position.
        #  Firas explains,”The reason for secrecy is because in the Arabian peninsula
        #  and specially in the Gulf, tribalism plays a huge part in diplomatic relations;
        #  my security can be at risk when I go back home. Coming out to my immediate
        #  family and close friends was easy, but taking my journey to the next level could not
        
        print(model.enc.decode(test0.tolist()))
        
        # num token to predict
        num_tokens = 50
        
        # infer
        model.infer(test0, num_tokens, ckpt=ckpt)

