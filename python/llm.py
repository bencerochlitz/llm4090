import torch
from torch import nn
import time

from layers import *
from llm_utils import *
from dataset_tokenizer import load_packed_padded_data

assert torch.cuda.is_available()
device = torch.device('cuda:0')


class LLM_training():
    
    def __init__(self, V, T, C, num_heads, num_layers, hidden_dim,
                 data_path, B, eot_token,
                 writer):
        
        # LLM
        print("building llm...")
        t_s = time.perf_counter()
        
        #torch.jit.script
        self.llm = (LLM(V, T, C, num_heads, num_layers, hidden_dim)).to(device)

        num_p = sum(p.numel() for p in self.llm.parameters())
        print("building llm done, total params: {}M, took {}s".format(
            num_p / 1.e6, time.perf_counter() - t_s))
        
        # loss
        #torch.jit.script
        self.loss_fn = (nn.CrossEntropyLoss(reduction='mean'))
        
        # optimizer
        # params according to Claude
        self.optimizer = torch.optim.AdamW(self.llm.parameters(),
                                          lr = 3e-4,
                                          betas=(0.9, 0.99),
                                        #   weight_decay=0.1,
                                          capturable=True)
        
        # graph capture
        self.g = torch.cuda.CUDAGraph()
        
        # counters
        self.step = 0
        self.epoch = 0
        
        # stats
        self.writer = writer
        self.loss_sum = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)
        self.loss_curr = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)
        
        # data
        self.tr, self.va, self.te, \
        self.tr_ids, self.va_ids, self.te_ids = load_packed_padded_data(data_path, T, device=device)
        
        self.tr_size = len(self.tr)
        self.va_size = len(self.va)
        self.te_size = len(self.te)
        
        # I wanted to use uint16 to store data but nothing is implemented for uint16 in pytorch
        dt = torch.int
        
        self.tr, self.va, self.te = \
            self.tr.to(dt), self.va.to(dt), self.te.to(dt)

        self.tr_ids, self.va_ids, self.te_ids = \
            self.tr_ids.to(dt), self.va_ids.to(dt), self.te_ids.to(dt)
        
        # token targets
        self.tr_targets = torch.zeros_like(self.tr)
        self.va_targets = torch.zeros_like(self.va)
        self.te_targets = torch.zeros_like(self.te)
        
        # compute targets to avoid computing on the fly
        print("compute_targets...")
        t_s = time.perf_counter()
        compute_targets(self.tr, self.tr_targets, eot_token)
        compute_targets(self.va, self.va_targets, eot_token)
        compute_targets(self.te, self.te_targets, eot_token)
        print("compute_targets done, took ", time.perf_counter() - t_s)
        
        # batch samples
        self.batch = torch.zeros((B, T), dtype=torch.int, device=device)
        self.batch_ids = torch.zeros((B, T), dtype=torch.int, device=device)
        self.batch_targets = torch.zeros((B, T), dtype=torch.int, device=device)
        
    def train_step(self):
        # graph safe method
        
        # sample
        sample_batch(self.batch, self.batch_ids,
                     self.tr, self.tr_ids,
                     self.tr_targets, self.batch_targets)
        
        # forward
        out_logits = self.llm(self.batch, self.batch_ids)
        
        # cross entropy loss only supports N, C input and C target shapes
        out_logits = torch.flatten(out_logits, start_dim=0, end_dim=-2)
        targets = torch.flatten(self.batch_targets).long()
        
        # loss
        loss = self.loss_fn(out_logits, targets)
        
        # optimizer
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        # add loss
        self.loss_sum[:] += loss
        self.loss_curr[:] = loss
    
    @torch.no_grad()
    def eval(self):
        # sample
        sample_batch(self.batch, self.batch_ids,
                     self.va, self.va_ids,
                     self.va_targets, self.batch_targets)
        
        # infer
        out_logits = self.llm(self.batch, self.batch_ids)
        
        # loss
        loss = self.loss_fn(out_logits, self.batch_targets)
        
        # record
        self.loss_curr[:] = loss
    
    def train_epoch(self, num_grad_steps):
        # train
        self.llm.train(True)
        
        t_s = time.perf_counter()
        
        for n in range(num_grad_steps):
            
            if n % 10 == 0:
                t_s = time.perf_counter()
            
            self.train_step()
            self.step += 1
            
            if (n + 1) % 10 == 0:
                print("step: {}, avg step time: {}s".format((n+1), time.perf_counter() - t_s))
            
            # stats
            self.writer.add_scalar('loss/avg', self.loss_sum.item() / (n+1), self.step)
            self.writer.add_scalar('loss/curr', self.loss_curr.item(), self.step)
        
        # stats
        self.writer.add_scalar('loss/train', self.loss_sum.item() / num_grad_steps, self.epoch)
        
        # reset loss
        self.loss_sum[:] = 0.0
        self.loss_curr[:] = 0.0
        
        # eval validation batch
        self.llm.eval()
        
        self.eval()
        
        # stats
        self.writer.add_scalar('loss/validation', self.loss_curr.item(), self.epoch)
    
    def train(self, num_epochs, num_grad_steps):
        for _ in range(num_epochs):
            
            print("starting epoch {}...".format(self.epoch))
            t_s = time.perf_counter()
            
            self.train_epoch(num_grad_steps)
            
            print("epoch {} done, took {}s".format(self.epoch, time.perf_counter() - t_s))
            
            self.epoch += 1
    
    def infer(self, encoder):
        pass
        
            
        
        