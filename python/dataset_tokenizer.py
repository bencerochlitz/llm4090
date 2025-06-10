
import tiktoken
from datasets import load_dataset
import numpy as np
import sys
import os
import h5py
import time
import torch
from argparse import ArgumentParser


assert torch.cuda.is_available()
device = torch.device('cuda:0')


MAX_LENGTH = 512
EOT_TOKEN = 50256


def save_string_data(path, dataset):
    data_train = dataset['train']['text']
    data_val = dataset['validation']['text']
    data_test = dataset['test']['text']

    print("train: ", np.shape(data_train))
    print("validation: ", np.shape(data_val))
    print("test: ", np.shape(data_test))

    print("Example data:\n", data_train[0:5])

    if os.path.exists(path):
        os.remove(path)

    with h5py.File(path, 'a') as h5:
        ds = h5.create_dataset('train', shape=len(data_train),
                               dtype=h5py.string_dtype())
        ds[:] = data_train

        ds = h5.create_dataset('val', shape=len(data_val),
                               dtype=h5py.string_dtype())
        ds[:] = data_val

        ds = h5.create_dataset('test', shape=len(data_test),
                               dtype=h5py.string_dtype())
        ds[:] = data_test


def load_string_data(path):
    with h5py.File(path, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]

    d_train = [s.decode('utf-8') for s in d_train]
    d_val = [s.decode('utf-8') for s in d_val]
    d_test = [s.decode('utf-8') for s in d_test]

    return d_train, d_val, d_test


def encode_dataset(data):
    enc = tiktoken.get_encoding("gpt2")  # Load the GPT-2 tokenizer

    num_threads = 4  # seems to be the sweet spot
    print("len(data): ", len(data))
    print("0.5 GB of data takes ~60s")

    tokens = enc.encode_batch(data, num_threads=num_threads)

    for t in tokens:
        t.append(enc.EOT_TOKEN)  # Ａdd the end of text token

    return tokens


def save_tokenized_data(path, data_train, data_val, data_test):
    if os.path.exists(path):
        os.remove(path)

    dt = h5py.vlen_dtype(np.dtype('uint16'))

    with h5py.File(path, 'a') as h5:
        ds = h5.create_dataset('train', shape=len(data_train), dtype=dt)
        ds[:] = data_train

        ds = h5.create_dataset('val', shape=len(data_val), dtype=dt)
        ds[:] = data_val

        ds = h5.create_dataset('test', shape=len(data_test), dtype=dt)
        ds[:] = data_test


def sequence_packing(tensor: torch.Tensor, np_arr: np.ndarray):
    # naive sequence packing
    # the useful data is only 15% of the total after padding without packing...
    
    curr_idx = 0
    i = 0
    for tokens in np_arr:
        # cut longer sequences
        n = min(len(tokens), MAX_LENGTH)
        
        # if it fits, pack, otherwise move on to the next entry
        end_idx = curr_idx + n
        
        if end_idx >= MAX_LENGTH:
            # next tensor entry
            i += 1
            curr_idx = 0
            end_idx = curr_idx + n
        
        # pack
        tensor[i][curr_idx : end_idx] = torch.from_numpy(tokens[: n])
        
        # increment curr_idx
        curr_idx = end_idx
    
    print("packing compressed sequences from {} to {}".format(len(tensor), i + 1))
    
    # return the reduced tensor
    return tensor[: i + 1]


def sequence_ids_gpu(tokens: torch.Tensor):
    B, T = tokens.shape
    
    ids = torch.tensor(torch.arange(0, T), device=device).unsqueeze(0)
    ids = ids.repeat(B, 1)
    print("sequence_ids ids.shape: ", ids.shape)
    
    tokens = tokens.to(device)
    
    mask = tokens == EOT_TOKEN
    
    for i in range(T-1):
        mask_i = mask[:, i]
        reduce_i = ids[:, i+1]
        
        ids[:, i+1:] -= (reduce_i * mask_i).unsqueeze(-1)
        
    return ids.cpu()
        

def save_packed_padded_data(path_tokens, path_tokens_packed):
    with h5py.File(path_tokens, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]

    # this still takes 9s...
    n_tr = len(d_train)
    n_va = len(d_val)
    n_te = len(d_test)
    tr = torch.full((n_tr, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16)
    va = torch.full((n_va, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16)
    te = torch.full((n_te, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16)

    # sequence packed tensors, reduced to the useful size
    tr = sequence_packing(tr, d_train)
    va = sequence_packing(va, d_val)
    te = sequence_packing(te, d_test)
    
    # make sure to overwrite the last token if the sequence was cut off
    tr[:, -1] = EOT_TOKEN
    va[:, -1] = EOT_TOKEN
    te[:, -1] = EOT_TOKEN
    
    print("sequence_ids...")
    t_s = time.perf_counter()
    
    tr_ids = sequence_ids_gpu(tr)
    va_ids = sequence_ids_gpu(va)
    te_ids = sequence_ids_gpu(te)
    
    print("sequence_ids took ", time.perf_counter() - t_s)

    # save the now padded data so we can load that for training
    if os.path.exists(path_tokens_packed):
        os.remove(path_tokens_packed)
        
    dt = np.uint16
    with h5py.File(path_tokens_packed, 'a') as h5:
        ds = h5.create_dataset('train', shape=(len(tr), MAX_LENGTH), dtype=dt)
        ds[:] = tr

        ds = h5.create_dataset('val', shape=(len(va), MAX_LENGTH), dtype=dt)
        ds[:] = va

        ds = h5.create_dataset('test', shape=(len(te), MAX_LENGTH), dtype=dt)
        ds[:] = te
        
        ds = h5.create_dataset('train_ids', shape=(len(tr), MAX_LENGTH), dtype=dt)
        ds[:] = tr_ids

        ds = h5.create_dataset('val_ids', shape=(len(va), MAX_LENGTH), dtype=dt)
        ds[:] = va_ids

        ds = h5.create_dataset('test_ids', shape=(len(te), MAX_LENGTH), dtype=dt)
        ds[:] = te_ids


def load_packed_padded_data(path, T=512, device='cpu'):
    print("loading dataset in uint16...")
    t_s = time.perf_counter()
    
    with h5py.File(path, 'r') as h5:
        tr = h5['train'][:, :T]
        va = h5['val'][:, :T]
        te = h5['test'][:, :T]
        tr_ids = h5['train_ids'][:, :T]
        va_ids = h5['val_ids'][:, :T]
        te_ids = h5['test_ids'][:, :T]
        
    tr = torch.from_numpy(tr).to(device)
    va = torch.from_numpy(va).to(device)
    te = torch.from_numpy(te).to(device)
    tr_ids = torch.from_numpy(tr_ids).to(device)
    va_ids = torch.from_numpy(va_ids).to(device)
    te_ids = torch.from_numpy(te_ids).to(device)
    
    print("tr.shape: ", tr.shape)
    print("va.shape: ", va.shape)
    print("te.shape: ", te.shape)
    
    assert(tr.shape == tr_ids.shape)
    assert(va.shape == va_ids.shape)
    assert(te.shape == te_ids.shape)
    
    dt = torch.uint16
    assert(tr.dtype == dt and va.dtype == dt and va.dtype == dt)
    assert(tr_ids.dtype == dt and va_ids.dtype == dt and te_ids.dtype == dt)

    print("done loading dataset, took ", time.perf_counter() - t_s)
    
    return tr, va, te, tr_ids, va_ids, te_ids


parser = ArgumentParser()
parser.add_argument("--save_str", type=bool, default=False, help="save the dataset as encoded string data")
parser.add_argument("--tokenize", type=bool, default=False, help="perform tokenization and save raw variable-length tokens")
parser.add_argument("--pack", type=bool, default=False, help="perform sequence packing, padding, positional embedding + save data")

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    path = './data/wikitext.h5'
    path_tokens = './data/wikitext_tok.h5'
    path_tokens_packed = './data/wikitext_tok_packed.h5'
    
    # path = './openwebtext.h5'
    # path_tokens = './data/openwebtext_tok.h5'
    # path_tokens_packed = './data/openwebtext_tok_packed.h5'
    
    if args.save_str:
        # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        # dataset = load_dataset("Skylion007/openwebtext")
        save_string_data(path, dataset)
    
    if args.tokenize:
        d_train, d_val, d_test = load_string_data(path)

        t_s = time.perf_counter()

        print("starting encoding d_train...")
        d_train = encode_dataset(d_train)

        print("starting encoding d_val...")
        d_val = encode_dataset(d_val)

        print("starting encoding d_test...")
        d_test = encode_dataset(d_test)

        print("done encoding, took ", time.perf_counter() - t_s)

        save_tokenized_data(path_tokens, d_train, d_val, d_test)
    
    if args.pack:
        print("packing sequences and padding with eot...")
        t_s = time.perf_counter()
        save_packed_padded_data(path_tokens, path_tokens_packed)
        print("done packing and padding data, took ", time.perf_counter() - t_s)
    
    
    tr, va, te, tr_ids, va_ids, te_ids = load_packed_padded_data(path_tokens_packed)

    # test a token after packing
    enc = tiktoken.get_encoding("gpt2")
    
    tr_np = tr[0].tolist()
    print("training token ids: __{}__".format(tr_ids[0]))
    print("training token seq: __{}__".format(tr_np))
    print("training token seq: __{}__".format(enc.decode(tr_np)))
    
    print("enc.decode([0]): __{}__".format(enc.decode([0])))
    print("enc.decode([50256]): __{}__".format(enc.decode([50256])))
