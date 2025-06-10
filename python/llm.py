import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import tiktoken

import time
import os
from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from layers import *
from llm_utils import *
from dataset_tokenizer import load_packed_padded_data

assert torch.cuda.is_available()
device = torch.device('cuda:0')


class LLM_training():
    
    def __init__(self, V, T, C, num_heads, num_layers,
                 data_path, B, eot_token,
                 writer, writer_dir):
        
        self.V = V
        self.T = T
        self.num_heads = num_heads
        self.eot_token = eot_token
        
        # encoder
        self.enc = tiktoken.get_encoding("gpt2")
        
        # LLM
        print("building llm...")
        t_s = time.perf_counter()
        
        self.llm = torch.jit.script(LLM(V, T, C, num_heads, num_layers)).to(device)

        num_p = sum(p.numel() for p in self.llm.parameters())
        print("building llm done, total params: {}M, took {}s".format(
            num_p / 1.e6, time.perf_counter() - t_s))
        
        # loss
        self.loss_fn = torch.jit.script(nn.CrossEntropyLoss(reduction='mean'))
        
        # optimizer
        # params are according to Claude
        self.optimizer = torch.optim.AdamW(self.llm.parameters(),
                                          lr = 3e-4,
                                          betas=(0.9, 0.99),
                                          weight_decay=0.1,
                                          capturable=True)
        
        # graph capture
        self.g = torch.cuda.CUDAGraph()
        
        # counters
        self.step = 0
        self.epoch = 0
        
        # stats
        self.writer = writer
        self.writer_dir = writer_dir
        self.loss_sum = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)
        self.loss_curr = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)
        self.loss_val_best = 1.0e6
        
        # data
        self.tr, self.va, self.te, \
        self.tr_ids, self.va_ids, self.te_ids = load_packed_padded_data(data_path, T, device=device)
        
        self.tr_size = len(self.tr)
        self.va_size = len(self.va)
        self.te_size = len(self.te)
    
        print("num training tokens: {}M".format(self.tr.numel() / 1e6))
        
        # I wanted to use uint16 to store data but nothing is implemented for uint16 in pytorch
        dt = torch.int
        
        self.tr, self.va, self.te = \
            self.tr.to(dt), self.va.to(dt), self.te.to(dt)

        self.tr_ids, self.va_ids, self.te_ids = \
            self.tr_ids.to(dt), self.va_ids.to(dt), self.te_ids.to(dt)
            
        self.tr.requires_grad = False
        self.va.requires_grad = False
        self.te.requires_grad = False
        self.tr_ids.requires_grad = False
        self.va_ids.requires_grad = False
        self.te_ids.requires_grad = False
        
        # token targets
        self.tr_targets = torch.zeros_like(self.tr, requires_grad=False)
        self.va_targets = torch.zeros_like(self.va, requires_grad=False)
        self.te_targets = torch.zeros_like(self.te, requires_grad=False)
        
        # compute targets to avoid computing on the fly
        print("compute_targets...")
        t_s = time.perf_counter()
        compute_targets(self.tr, self.tr_targets, eot_token)
        compute_targets(self.va, self.va_targets, eot_token)
        compute_targets(self.te, self.te_targets, eot_token)
        print("compute_targets done, took ", time.perf_counter() - t_s)
        
        # batch samples
        self.batch = torch.zeros((B, T), dtype=torch.int, device=device, requires_grad=False)
        self.batch_ids = torch.zeros((B, T), dtype=torch.int, device=device, requires_grad=False)
        self.batch_targets = torch.zeros((B, T), dtype=torch.int, device=device, requires_grad=False)
        
        # self.eot_tens = torch.full((B, T), eot_token, dtype=torch.int, device=device, requires_grad=False)
        self.mask = torch.zeros((B, T), dtype=torch.bool, device=device, requires_grad=False)
    
    def set_lr_scheduler(self, total_steps):
        # Cosine annealing scheduler
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=3e-5  # Minimum LR (10% of your base 3e-4)
        )
        
    def train_step(self):
        # graph safe method
        
        # sample
        with record_function("sample_batch"):
            sample_batch(self.batch, self.batch_ids,
                        self.tr, self.tr_ids,
                        self.tr_targets, self.batch_targets)
        
        # compute attention mask for packed sequence
        with record_function("att_mask_packed_seq"):
            att_mask = att_mask_packed_seq(self.batch, self.eot_token)
            att_mask = torch.repeat_interleave(att_mask, self.num_heads, dim=0)  # [N * H, T, S]
        
        # forward
        with record_function("forward"):
            out_logits = self.llm(self.batch, self.batch_ids, att_mask)
            
            # cross entropy loss only supports N, C input and C target shapes
            out_logits = torch.flatten(out_logits, start_dim=0, end_dim=-2)
            targets = torch.flatten(self.batch_targets).long()
            
            # mask the loss for input eot tokens
            self.mask[:] = self.batch == self.eot_token
            mask = torch.flatten(self.mask).unsqueeze(-1)
            # set the logits vector to a uniform distribution
            out_logits = torch.where(mask, 1, out_logits)
        
            # loss
            loss = self.loss_fn(out_logits, targets)
        
        # optimizer
        with record_function("optimizer"):
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        
        # add loss
        self.loss_sum[:] += loss.detach()
        self.loss_curr[:] = loss.detach()
    
    def eval(self):
        # sample
        sample_batch(self.batch, self.batch_ids,
                     self.va, self.va_ids,
                     self.va_targets, self.batch_targets)
        
        # compute attention mask for packed sequence
        att_mask = att_mask_packed_seq(self.batch, self.eot_token)
        att_mask = torch.repeat_interleave(att_mask, self.num_heads, dim=0)  # [N * H, T, S]
        
        # forward
        out_logits = self.llm(self.batch, self.batch_ids, att_mask)
        
        # cross entropy loss only supports N, C input and C target shapes
        out_logits = torch.flatten(out_logits, start_dim=0, end_dim=-2)
        targets = torch.flatten(self.batch_targets).long()
        
        # mask the loss for input eot tokens
        self.mask[:] = self.batch == self.eot_token
        mask = torch.flatten(self.mask).unsqueeze(-1)
        # set the logits vector to a uniform distribution
        out_logits = torch.where(mask, 1, out_logits)
    
        # loss
        loss = self.loss_fn(out_logits, targets)
        
        # record
        self.loss_curr[:] = loss.detach()
        
    def trace_handler(self, p):
        sort_by_keyword = "self_" + "device" + "_time_total"
        output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
    
    def train_epoch(self, num_grad_steps, should_profile=False):
        # train
        self.llm.train(True)
        
        t_s = time.perf_counter()
        
        if should_profile and self.epoch == 0:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=2,
                    warmup=3,
                    active=10),
                on_trace_ready=self.trace_handler,
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as p:
                for n in range(num_grad_steps):
                    self.g.replay()
                    p.step()  
        elif should_profile:
            return
        
        for n in range(num_grad_steps):
            if n % 10 == 0:
                t_s = time.perf_counter()
                
            # graph capture and replay
            if self.epoch == 0 and self.step < 3:
                self.train_step()
            elif self.epoch == 0 and self.step == 3:
                with torch.cuda.graph(self.g):
                    self.train_step()
            else:
                self.g.replay()
                
            # step
            self.step += 1
                
            # learning rate scheduler
            self.cosine_scheduler.step()
            lr = self.cosine_scheduler.get_last_lr()[0]
            
            # stats
            loss_avg = self.loss_sum.item() / (n+1)
            self.writer.add_scalar('loss/avg', self.loss_sum.item() / (n+1), self.step)
            self.writer.add_scalar('loss/curr', self.loss_curr.item(), self.step)
            self.writer.add_scalar('lr', lr, self.step)
            
            if (n + 1) % 10 == 0:
                print("step: {}, loss: {}, avg step time: {}s".format(
                    (n+1), loss_avg, time.perf_counter() - t_s))
        
        # stats
        self.writer.add_scalar('loss/train', self.loss_sum.item() / num_grad_steps, self.epoch)
        
        # reset loss
        self.loss_sum[:] = 0.0
        self.loss_curr[:] = 0.0
        
        # eval validation batch
        self.llm.eval()
        with torch.no_grad():
            self.eval()
        
        # print
        loss_val = self.loss_curr.item()
        print("epoch: {} validation loss: {}".format(self.epoch, loss_val))
        
        # save best
        if loss_val < self.loss_val_best:
            self.loss_val_best = loss_val
            torch.save(self.llm.state_dict(), os.path.join(self.writer_dir, f'best.ckpt'))
            print("saved ckpt with validation loss: ", loss_val)
        
        # stats
        self.writer.add_scalar('loss/validation', loss_val, self.epoch)
    
    def train(self, num_epochs, num_grad_steps, should_profile):
        if should_profile:
            self.train_epoch(num_grad_steps, should_profile=True)
            return
        
        t_total = 0.0
        for _ in range(num_epochs):
            
            print("starting epoch {}...".format(self.epoch))
            t_s = time.perf_counter()
            
            self.train_epoch(num_grad_steps, should_profile=False)
            
            t_i = time.perf_counter() - t_s
            t_total += t_i
            print("epoch {} done, took {}s, total time: {}".format(self.epoch, t_i, t_total))
            
            self.epoch += 1
            
    def process_test_seq(self, seq_in):
        l = len(seq_in)
        
        # prepare tensor inputs
        x = torch.full((self.T, ), self.eot_token, device=device)
        x[:l] = torch.as_tensor(seq_in)
        
        ids = torch.arange(0, self.T, device=device)
        ids[l:] = 0
        
        x = x.unsqueeze(0)
        ids = ids.unsqueeze(0)
        
        # compute attention mask for packed sequence
        att_mask = att_mask_packed_seq(x, self.eot_token)
        att_mask = torch.repeat_interleave(att_mask, self.num_heads, dim=0)
        
        # the attention mask should ignore the eot padding for inference
        att_mask[:, l, :] = True
        att_mask[:, :, l] = True
        att_mask[:, l, l] = False
        
        # print(att_mask.shape)
        # print(att_mask[0, :l*2, :l*2])
        
        return x, ids, att_mask
    
    def infer(self, test_seq, num_tokens, ckpt=str):
        with torch.no_grad():
            # load ckpt
            self.llm.load_state_dict(torch.load(ckpt, weights_only=True))
            print("loaded best checkpoint")
            
            
            # process the raw string test sequence
            test_seq = self.enc.encode(test_seq)
            print("input test_seq: {}".format(test_seq))
            
            # store next tokens
            out = []
            
            for _ in range(num_tokens):
                # copy the original seq and append the already predicted tokens
                seq_in = test_seq.copy()
                seq_in.extend(out)
                l = len(seq_in)
                
                # process
                x, ids, att_mask = self.process_test_seq(seq_in)
                
                # forward pass
                y = self.llm.infer(x, ids, att_mask)
                
                # print the predections
                y_lst = y[:l].tolist()
                y_str = self.enc.decode(y_lst)
                # print("all pred tokens: ", y_lst)
                print("all pred decoded: ", y_str)
                
                # take the next predicted token
                out.append(y_lst[-1])

            # print the final predections
            out_str = self.enc.decode(out)
            print("\n########## NEXT TOKEN PREDECTIONS ##########")
            print("next tokens: ", out)
            print("next decoded: ", out_str)
            
        
        
        
            
        
        