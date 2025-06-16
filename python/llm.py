import torch
from torch import nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import tiktoken

import time
import os
from torch.profiler import profile, record_function, ProfilerActivity

from torch.optim.lr_scheduler import CosineAnnealingLR

from layers import *
from llm_utils import *
from dataset_tokenizer import load_packed_padded_data
from dataset_tokenizer_v2 import DataLoader

assert torch.cuda.is_available()
device = torch.device('cuda:0')


class LLM_training():
    
    def __init__(self, T, C, num_heads, num_layers,
                 B, num_batches, writer, writer_dir):
        
        self.loader = DataLoader()
        
        self.V = self.loader.V
        self.eot_token = self.loader.eot_token
        
        self.T = T
        self.num_heads = num_heads
        self.num_batches = num_batches
        
        # LLM
        print("building llm...")
        t_s = time.perf_counter()
        
        self.llm = LLM(self.V, T, C, num_heads, num_layers).to(torch.bfloat16).to(device)
        self.llm.compile()

        num_p = sum(p.numel() for p in self.llm.parameters())
        print("building llm done, total params: {}M, took {}s".format(
            num_p / 1.e6, time.perf_counter() - t_s))
        
        # loss
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        
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
        self.loader = DataLoader()
        n_GBs = 8
        data = self.loader.load_ready_data(n_GBs)
        
        n_train = int(len(data) * 0.8)
        n_val = len(data) - n_train
        
        self.tr = data[: n_train].to(device)
        self.val = data[n_train: ].to(device)
        self.tr.requires_grad = False
        self.val.requires_grad = False
        print("num training tokens: {}M".format(self.tr.numel() / 1e6))
        print("num validation tokens: {}M".format(self.val.numel() / 1e6))
        
        # batch samples
        self.batch = torch.zeros((B, T), dtype=torch.int, device=device, requires_grad=False)
        self.batch_ids = torch.arange(0, T, 1, device=device).unsqueeze(0)
        self.batch_ids = self.batch_ids.repeat(B, 1)
        assert(self.batch.shape == self.batch_ids.shape)
        
    @property
    def enc(self):
        # encoder
        return self.loader.enc
        
    def set_lr_scheduler(self, total_steps):
        # Cosine annealing scheduler
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=3e-5  # Minimum LR (10% of your base 3e-4)
        )
        
    def train_step(self):
        # graph safe method
        self.optimizer.zero_grad(set_to_none=True)
        self.loss_curr[:] = 0.0
        
        # run N batches sequentially
        for _ in range(self.num_batches):
            # sample
            with torch.no_grad():
                with record_function("sample_batch"):
                    sample_batch(self.batch, self.tr)
                
            # forward pass
            with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
                with record_function("forward"):
                
                    logits = self.llm(self.batch, self.batch_ids)
                
                with record_function("compute_loss"):
                    
                    # scales the loss by num_batches
                    loss = compute_loss(logits, self.batch,
                                        self.eot_token, self.loss_fn,
                                        self.num_batches)
                    
            # accumulate gradients
            with record_function("backward"):
                loss.backward()
            
            # stats
            self.loss_curr[:] += loss.detach()
        
        # optimizer
        with record_function("optimizer"):
            self.optimizer.step()
        
        # stats
        self.loss_sum[:] += self.loss_curr
    
    def eval(self):
        # sample
        sample_batch(self.batch, self.val)
        
        # forward
        logits = self.llm(self.batch, self.batch_ids)
        
        # scales the loss by num_batches
        loss = compute_loss(logits, self.batch,
                            self.eot_token, self.loss_fn, 1)
        
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
                    self.train_step()
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
            torch.save(self.llm.state_dict(), os.path.join(self.writer_dir, 'best.ckpt'))
            print("saved ckpt with validation loss: ", loss_val)
            
        # save after every 10 epochs to allow reload
        if (self.epoch + 1) % 10 == 0:
            torch.save(self.llm.state_dict(), os.path.join(self.writer_dir, f'epoch_{self.epoch + 1}.ckpt'))
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
    
    def infer(self, test_enc, num_tokens, ckpt=str):
        with torch.no_grad():
            # load ckpt
            self.llm.load_state_dict(torch.load(ckpt, weights_only=True))
            print("loaded best checkpoint")
            
            # input buffer
            x_in = torch.full((self.T, ), self.eot_token, device=device)
            l = len(test_enc)
            x_in[:l] = test_enc
            
            # ids
            ids = torch.arange(0, self.T, 1, device=device).unsqueeze(0)
            
            # TODO: implement sequence truncation
            assert(l + num_tokens < self.T)
            
            # store next tokens
            out = []
            
            for _ in range(num_tokens):
                # forward pass
                y = self.llm.infer(x_in.unsqueeze(0), ids)
                
                # next predected token
                new_prediction = y[l - 1]
                
                # append to existing sequence and increment
                x_in[l] = new_prediction
                l += 1
                
                # store
                out.append(new_prediction.item())

            # print the final predections
            out_str = self.enc.decode(out)
            print("\n########## NEXT TOKEN PREDECTIONS ##########")
            print("next tokens: ", out)
            print("next decoded: ", out_str)
            
        
        
        
            
        
        