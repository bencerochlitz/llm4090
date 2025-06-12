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
    # the wikitext sequnces are pretty short anyway, so saving compute here
    T = 512 //4  # seq_length 1024
    C = 768 //2  # embed_dim 768
    num_heads = 6  # 12
    num_layers = 6  # 12
    
    eot_token = V - 1
    # perf scales linearly until 64, for 128 VRAM is full
    B = 8
    num_batches = 1
    
    # boilerplate
    dir = "runs/LLM_{}".format(mode)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    writer = SummaryWriter(dir)

    # llm trainer
    model = LLM_training(V, T, C, num_heads, num_layers,
                           path_tokens_packed, B, num_batches,
                           eot_token, writer, dir)
    
    # train
    if mode == 'train':
        # NOTE: learning rate scheduler is hardcoded for now
        num_epochs = 500
        num_grad_steps = 100
        model.set_lr_scheduler(num_epochs * num_grad_steps)
        model.train(num_epochs, num_grad_steps, profile)
    
    if mode == 'infer':
        ckpt = "runs/LLM_train_best/best.ckpt"
        
        # test seq
        # this was in training
        # test_seq = "The game began development in 2010 , carrying over a large portion of the work done"
        # test_seq = "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . "\
            # "While it retained the standard features of the series"
        
        # from the test set
        test_seq = "In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . "\
            "He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . "\
            "How to Curse was performed at Bush Theatre in the"
        # what follows:
        # "London Borough of Hammersmith and Fulham . "
        # "Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . "\
        # "In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , "\
        # "followed by an appearance on the television series Survivors in November 2008 . "\
        # "He had a recurring role in ten episodes of the television series Casualty in 2010 , as " Kieron Fletcher " . "\
        # "Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."\
        
        # num token to predict
        num_tokens = 20
        
        # infer
        model.infer(test_seq, num_tokens, ckpt=ckpt)

