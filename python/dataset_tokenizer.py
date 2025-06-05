
import tiktoken
from datasets import load_dataset
import numpy as np
import sys
import os
import h5py


def load_and_convert_to_h5(path):

    # dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    data_train = dataset['train']['text']
    data_val = dataset['validation']['text']
    data_test = dataset['test']['text']

    print("train: ", np.shape(data_train))
    print("validation: ", np.shape(data_val))
    print("test: ", np.shape(data_test))

    print("Example data:\n", data_train[0:5])

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


def load_h5(path):
    with h5py.File(path, 'r') as h5:
        d_train = h5['train'][:]
        d_val = h5['val'][:]
        d_test = h5['test'][:]

    return d_train, d_val, d_test


def encode_dataset(data):

    # Test encoding
    enc = tiktoken.get_encoding("gpt2")  # Load the GPT-2 tokenizer

    # arr = np.array([])

    for s in data[0:5]:

        tokens = enc.encode(s.decode('utf-8'))  # Tokenize the string
        tokens.append(enc.eot_token)  # Ａdd the end of text token

        print("tokens[0] type: ", type(tokens[0]))
        print("tokens[0] size: ", sys.getsizeof(tokens[0]))

        print(tokens)

        np.append(arr, tokens)

    print("tokenized data: ", arr)


if __name__ == "__main__":

    path = './data/wikitext.h5'
    # load_and_convert_to_h5(path)

    d_train, d_val, d_test = load_h5(path)

    encode_dataset(d_train)
