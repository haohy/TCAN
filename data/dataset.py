import os
import torch
import pickle
import unidecode
import observations
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from torch.utils import data
from torch.autograd import Variable

import logging
from IPython import embed


class RawDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus=True):
        super(RawDataset, self).__init__()  
        # self.dir_data = os.path.join(dir_data_root, dataset_name)
        self.seq_len = seq_len
        self.data_all, self.label_all, self.n_dict = self._get_data(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus)
       
    def __getitem__(self, index):
        data = Variable(self.data_all[index])
        label = Variable(self.label_all[index])
        return data, label

    def __len__(self):
        return len(self.data_all)
    
    def _get_data(self, dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus):
        dir_data = os.path.join(dir_data_root, dataset_name)
        if dataset_name == 'penn':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))
                
        elif dataset_name == 'char_penn':
            dir_data = os.path.join(dir_data_root, dataset_name)
            if os.path.exists(dir_data + "/corpus") and is_corpus:

                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                file, testfile, valfile = getattr(observations, 'ptb')(dir_data)
                corpus = Corpus_char(file + " " + valfile + " " + testfile)
                corpus.train = char_tensor(corpus, file)
                corpus.valid = char_tensor(corpus, valfile)
                corpus.test = char_tensor(corpus, testfile)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))

        elif dataset_name == 'penn_attn':
                if os.path.exists(dir_data + "/corpus") and is_corpus:
                    corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
                else:
                    corpus = Corpus_word(dir_data)
                    pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))

            # if task == 'train':
            #     data_task = char_tensor(corpus, file)
            # elif task == 'valid':
            #     data_task = char_tensor(corpus, valfile)
            # elif task == 'test':
            #     data_task = char_tensor(corpus, testfile)
            
            # n_characters = len(corpus.dict)
            # n_dict = len(corpus.dict)

        n_dict = len(corpus.dictionary)
        if task == 'train':
            data_task = corpus.train
        elif task == 'valid':
            data_task = corpus.valid
        elif task == 'test':
            data_task = corpus.test

        num_data = data_task.size(0) // valid_len
        data_all, label_all = [], []
        for i in range(num_data):
            if i*valid_len+seq_len+1 > data_task.size(0):
                break
            data_all.append(data_task[i*valid_len:i*valid_len+seq_len])
            label_all.append(data_task[i*valid_len+1:i*valid_len+seq_len+1])

        # embed(header="dataset")

        return data_all, label_all, n_dict


def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in tqdm(range(len(string)), ncols=80):
        tensor[i] = corpus.dictionary.char2idx[string[i]]
    return Variable(tensor)


class Dictionary_char(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus_char(object):
    def __init__(self, string):
        self.dictionary = Dictionary_char()
        for c in string:
            self.dictionary.add_word(c)
        self.dictionary.prep_dict()
        self.train = None
        self.valid = None
        self.test = None


class Corpus_word(object):
    def __init__(self, path):
        self.dictionary = Dictionary_word()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                # words = line.split() 
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                # words = line.split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

class Dictionary_word(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


if __name__ == '__main__':
    dir_data_root = './'
    dataset_name = 'penn_attn'
    task = 'train'
    batch_size = 1
    seq_len = 20
    valid_len = 10
    rawdataset = RawDataset(dir_data_root, dataset_name, task, seq_len, valid_len)
    embed()
    total = 0
    for _ in rawdataset:
        total += 1
    print(total)

