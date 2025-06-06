
import tiktoken
from datasets import load_dataset
import numpy as np
import sys
import os
import h5py
import time
import torch


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

    # Test encoding
    enc = tiktoken.get_encoding("gpt2")  # Load the GPT-2 tokenizer

    arr = []
    num_threads = 4  # seems to be the sweet spot
    num_s = len(data)
    # num_batches = num_s // num_threads
    print("num_s: ", num_s)
    print("1e6 takes ~30s")
    # print("num_batches: ", num_batches)

    # for i in range(num_batches):
    # print("encoding batch: ", i)
    # idx_b = i * num_threads
    # idx_e = (i + 1) * num_threads

    tokens = enc.encode_batch(data, num_threads=num_threads)

    for t in tokens:
        t.append(enc.EOT_TOKEN)  # Ａdd the end of text token

    # tokens = np.asarray(tokens, dtype=np.uint16)
    # arr.extend(tokens)

    # if i < 3:
    #     # print("tokens[0] type: ", type(tokens[0]))
    #     # print("tokens[0] size: ", sys.getsizeof(tokens[0]))
    #     print(s)
    #     print(tokens)
    #     print(enc.decode(tokens))

    # print("tokenized data: ", arr)

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


def pad_eot(tokens):
    pad = MAX_LENGTH - len(tokens)

    if pad <= -1:
        return

    if pad == 0:
        tokens[-1] = EOT_TOKEN
        return

    return np.append(tokens, ([EOT_TOKEN] * pad))


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
        

def save_packed_padded_data(path_tokens, path_tokens_packed):
    with h5py.File(path_tokens, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]

    # # this is much slower than overriding torch tensors
    # d_train = list(map(pad_eot, d_train))
    # d_val = list(map(pad_eot, d_val))
    # d_test = list(map(pad_eot, d_test))
    # print("d_train ", d_train[10])

    # this still takes 9s...
    n_tr = len(d_train)
    n_va = len(d_val)
    n_te = len(d_test)
    tr = torch.full((n_tr, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16, pin_memory=True)
    va = torch.full((n_va, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16, pin_memory=True)
    te = torch.full((n_te, MAX_LENGTH), EOT_TOKEN, dtype=torch.uint16, pin_memory=True)

    # sequence packed tensors, reduced to the useful size
    tr = sequence_packing(tr, d_train)
    va = sequence_packing(va, d_val)
    te = sequence_packing(te, d_test)
    
    # make sure to overwrite the last token if the sequence was cut off
    tr[:, -1] = EOT_TOKEN
    va[:, -1] = EOT_TOKEN
    te[:, -1] = EOT_TOKEN

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
        
    # return just for testing
    return tr, va, te


def load_tokenized_data(path):
    with h5py.File(path, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]


if __name__ == "__main__":

    path = './data/wikitext.h5'
    path_tokens = './data/wikitext_tok.h5'
    path_tokens_packed = './data/wikitext_tok_packed.h5'
    
    if False:
        # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        save_string_data(path, dataset)
    
    if False:
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
    
    print("packing sequences and padding with eot...")
    t_s = time.perf_counter()
    tr, va, te = save_packed_padded_data(path_tokens, path_tokens_packed)
    print("done packing and padding data, took ", time.perf_counter() - t_s)

    # test a token after packing
    enc = tiktoken.get_encoding("gpt2")
    
    np_arr = tr[0].tolist()
    print("training token seq: __{}__".format(np_arr))
    print("training token seq: __{}__".format(enc.decode(np_arr)))
    
    print("enc.decode([0]): __{}__".format(enc.decode([0])))
    print("enc.decode([50256]): __{}__".format(enc.decode([50256])))
