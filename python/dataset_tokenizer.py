
import tiktoken
from datasets import load_dataset
import numpy as np
import sys
import os
import h5py
import time
import torch


def load_and_convert_to_h5(path, dataset):
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
        t.append(enc.eot_token)  # Ａdd the end of text token

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
        ds = h5.create_dataset('train', shape=len(data_train),
                               dtype=dt)
        ds[:] = data_train

        ds = h5.create_dataset('val', shape=len(data_val),
                               dtype=dt)
        ds[:] = data_val

        ds = h5.create_dataset('test', shape=len(data_test),
                               dtype=dt)
        ds[:] = data_test


def append_to_tensor(t: torch.Tensor, np_data: np.ndarray, max_length):
    for i, d in enumerate(np_data):
        n = min(len(d), max_length)
        t[i][: n] = torch.from_numpy(d[: n])

        # if i == 1e6:
        #     break


def load_tokenized_data(path, max_length=512):
    with h5py.File(path, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]

    n_tr = len(d_train)
    n_va = len(d_val)
    n_te = len(d_test)
    tr = torch.zeros((n_tr, max_length), dtype=torch.uint16, pin_memory=True)
    va = torch.zeros((n_va, max_length), dtype=torch.uint16, pin_memory=True)
    te = torch.zeros((n_te, max_length), dtype=torch.uint16, pin_memory=True)

    append_to_tensor(tr, d_train, max_length)
    append_to_tensor(va, d_val, max_length)
    append_to_tensor(te, d_test, max_length)

    return tr, va, te


if __name__ == "__main__":

    path = './data/wikitext.h5'
    # # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    # load_and_convert_to_h5(path, dataset)

    d_train, d_val, d_test = load_string_data(path)

    t_s = time.perf_counter()

    print("starting encoding d_train...")
    d_train = encode_dataset(d_train)

    print("starting encoding d_val...")
    d_val = encode_dataset(d_val)

    print("starting encoding d_test...")
    d_test = encode_dataset(d_test)

    print("done encoding, took ", time.perf_counter() - t_s)

    path = './data/wikitext_tok.h5'
    save_tokenized_data(path, d_train, d_val, d_test)
