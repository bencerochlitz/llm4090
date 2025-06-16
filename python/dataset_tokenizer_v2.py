
import tiktoken
# print(tiktoken.list_encoding_names())
from datasets import load_dataset
import numpy as np
import sys
import os
import h5py
import time
import torch
from argparse import ArgumentParser

import glob


assert torch.cuda.is_available()
device = torch.device('cuda:0')


class DataLoader():
    def __init__(self):
        self.path_download = './data'
        self.path_str = './data/dataset_str.h5'
        self.path_tok = './data/dataset_tok.h5'
        self.path_tok_sliced = './data/dataset_tok_sliced.h5'
        
        # encoder
        self.enc = tiktoken.get_encoding("cl100k_base")
        print("n vocab: ", self.enc.n_vocab)
        print("eot token: ", self.enc.eot_token)
        
    @property
    def V(self):
        return self.enc.n_vocab
        
    @property
    def eot_token(self):
        return self.enc.eot_token
    
    def get_arrow_files(self):
        arrow_files = glob.glob(self.path_download + "/train/*.arrow")
        print("found {} arrow files".format(len(arrow_files)))
        return arrow_files
        
    def download(self, dataset_name):
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        dataset.save_to_disk(self.path_download)
        
    def load_raw_and_save_as_str(self, data_file):
        t_s = time.perf_counter()
        
        dataset = load_dataset('arrow', data_files=data_file)
        
        data_train = dataset['train']['text']
        
        # sum = 0
        # for s in data_train:
        #     sum += len(s)
        # sum /= len(data_train)

        # print("train: ", len(data_train))
        # print("avg seq length: ", sum)
        # print("Example data:\n", data_train[0][0:128])
        
        # create the dataset if it doesn't exist yet
        if not os.path.exists(self.path_str):
            with h5py.File(self.path_str, 'a') as h5:
                ds = h5.create_dataset('train', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        
        with h5py.File(self.path_str, 'a') as h5:
            ds = h5['train']
            
            ds_idx = len(ds)
            
            ds_end = ds_idx + len(data_train)
            
            ds.resize(ds_end, axis=0)

            ds[ds_idx : ds_end] = data_train
            
        print("done load_raw_and_save_as_str, took ", time.perf_counter() - t_s)
            
            
    def load_str_and_tokenize(self, idx_start: int, idx_end: int):
        # load
        with h5py.File(self.path_str, 'r') as h5:
            ds = h5['train']
            
            # return if we are done
            if idx_start >= len(ds):
                return False
            
            data = ds[idx_start : idx_end]

        data = [s.decode('utf-8') for s in data]

        # tokenize
        num_threads = 4  # seems to be the sweet spot
        print("len(data): ", len(data))
        print("0.5 GB of data takes ~7s")

        t_s = time.perf_counter()
        tokens = self.enc.encode_batch(data, num_threads=num_threads)
        
        for t in tokens:
            t.append(self.enc.eot_token) # Ａdd the end of text token
            
        print("done encoding, took ", time.perf_counter() - t_s)
        
        # sanity check for the first sequence
        t0 = tokens[0]
        str0 = self.enc.decode(t0)
        print("Example data:\n", str0[0:128])

        # save tokenized data
        t_s = time.perf_counter()

        dt = h5py.vlen_dtype(np.dtype('int32'))
        
        # create the dataset if it doesn't exist yet
        if not os.path.exists(self.path_tok):
            with h5py.File(self.path_tok, 'a') as h5:
                ds = h5.create_dataset('train', shape=(0,), maxshape=(None,), dtype=dt)
        
        with h5py.File(self.path_tok, 'a') as h5:
            ds = h5['train']
            
            ds_idx = len(ds)
            
            ds_end = ds_idx + len(tokens)
            
            ds.resize(ds_end, axis=0)

            ds[ds_idx : ds_end] = tokens
            
        print("done saving tokens, took ", time.perf_counter() - t_s)
        
        return True
            
            
    def slice_to_uniform_sequences(self, idx_start: int, idx_end: int, T: int=512):
        print("slice_to_uniform_sequences...")
        t_s = time.perf_counter()
        
        # load
        with h5py.File(self.path_tok, 'r') as h5:
            ds = h5['train']
            
            # return if we are done
            if idx_start >= len(ds):
                return False
            
            tokens = ds[idx_start : idx_end]
            
        # # count the sequences shorter than T
        # count = 0
        # for t in tokens:
        #     if len(t) < T:
        #         count += 1
                
        # print("sequences shorter than T: ", count)
        
        # count the number of useful sequences we can extract
        count = 0
        for t in tokens:
            count += len(t) // T
            
        print("number of useful sequences: ", count)
        
        # buffer to collect sequences
        buf = torch.empty((count, T), dtype=torch.int32)
        
        idx = 0
        for t in tokens:
            n = len(t) // T

            buf[idx : idx + n, :] = torch.as_tensor(t[: n * T]).view(n, T)
            
            idx += n
            
        print("slice_to_uniform_sequences done, took ", time.perf_counter() - t_s)
        t_s = time.perf_counter()
            
        # create the dataset if it doesn't exist yet
        if not os.path.exists(self.path_tok_sliced):
            with h5py.File(self.path_tok_sliced, 'a') as h5:
                ds = h5.create_dataset('train', shape=(0, T), maxshape=(None, T), dtype=np.int32)
        
        with h5py.File(self.path_tok_sliced, 'a') as h5:
            ds = h5['train']
            
            ds_idx = len(ds)
            
            ds_end = ds_idx + len(buf)
            
            ds.resize(ds_end, axis=0)

            ds[ds_idx : ds_end] = buf
            
        print("saving sliced dataset, took ", time.perf_counter() - t_s)
        
        return True
    
    
    def load_ready_data(self, n_GBs, T=512, begin=True):
        print("load_ready_data...")
        t_s = time.perf_counter()
        
        seq_size = 4 * T
        requested_size = 1024 * 1024 * 1024 * n_GBs
        
        num_rows = int(requested_size / seq_size)
        print("num_rows: ", num_rows)
        
        # load
        with h5py.File(self.path_tok_sliced, 'r') as h5:
            ds = h5['train']
            
            if begin:
                tokens = ds[:num_rows]
            else:
                L = len(ds)
                tokens = ds[L-num_rows :]
            
        print("load_ready_data done, took ", time.perf_counter() - t_s)
        
        return torch.as_tensor(tokens)
    
    

parser = ArgumentParser()
parser.add_argument("--save_str", type=bool, default=False, help="save the dataset as encoded string data")
parser.add_argument("--tokenize", type=bool, default=False, help="perform tokenization and save raw variable-length tokens")
parser.add_argument("--pack", type=bool, default=False, help="perform sequence packing, padding, positional embedding + save data")

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    dataset_name = "Skylion007/openwebtext"
    
    loader = DataLoader()
    
    # loader.download(dataset_name)
    
    
    if not os.path.exists(loader.path_str):
        
        arrow_files = loader.get_arrow_files()

        for f in arrow_files:
            loader.load_raw_and_save_as_str(f)
    
            
    if not os.path.exists(loader.path_tok):
        N = int(1e5)
        idx_start = 0
        idx_end = N
        
        not_done = True
        while not_done:
            
            not_done = loader.load_str_and_tokenize(idx_start, idx_end)
            idx_start = idx_end
            idx_end += N
        
        
    if not os.path.exists(loader.path_tok_sliced):
        
        N = int(1e5)
        idx_start = 0
        idx_end = N
        
        not_done = True
        while not_done:
            
            not_done = loader.slice_to_uniform_sequences(idx_start, idx_end)
            idx_start = idx_end
            idx_end += N
        
    
