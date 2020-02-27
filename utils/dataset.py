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
from torchvision import datasets, transforms

import logging
from IPython import embed


class RawDataset(data.Dataset):
    def __init__(self, dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus=True, is_permute=False, seed=1111):
        super(RawDataset, self).__init__()  
        self.is_permute = is_permute
        if is_permute:
            torch.manual_seed(seed)
            self.permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.data_all, self.label_all, self.n_dict, self.dictionary = self._get_data(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus)
       
    def __getitem__(self, index):
        data = Variable(self.data_all[index])
        if self.dataset_name == 'mnist':
            data = data.view(1, 784).float()
            # data = data.view(784).long()
            if self.is_permute:
                data = data[:, self.permute]
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

        elif dataset_name == 'mnist':
            corpus = Corpus_mnist(dir_data)

        elif dataset_name == 'wikitext-2':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))
        
        elif dataset_name == 'wikitext-103':
            if os.path.exists(dir_data + "/corpus") and is_corpus:
                corpus = pickle.load(open(dir_data + '/corpus', 'rb'))
            else:
                corpus = Corpus_word(dir_data)
                pickle.dump(corpus, open(dir_data + '/corpus', 'wb'))

        n_dict = len(corpus.dictionary)
        dictionary = corpus.dictionary
        if task == 'train':
            data_task = corpus.train
        elif task == 'valid':
            data_task = corpus.valid
        elif task == 'test':
            data_task = corpus.test

        if self.dataset_name == 'mnist':
            if task == 'valid':
                task = 'test'
            # return getattr(data_task, '{}_data'.format(task))[:640], getattr(data_task, '{}_labels'.format(task))[:640], n_dict
            return getattr(data_task, '{}_data'.format(task)), getattr(data_task, '{}_labels'.format(task)), n_dict

        num_data = data_task.size(0) // valid_len
        data_all, label_all = [], []
        for i in range(num_data):
            if i*valid_len+seq_len+1 > data_task.size(0):
                break
            data_all.append(data_task[i*valid_len:i*valid_len+seq_len])
            label_all.append(data_task[i*valid_len+1:i*valid_len+seq_len+1])

        return data_all, label_all, n_dict, dictionary


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
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
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

class Corpus_mnist(object):
    def __init__(self, path):
        self.dictionary = list(range(10))
        self.train = datasets.MNIST(root=path, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        self.valid = datasets.MNIST(root=path, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
        self.test = datasets.MNIST(root=path, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))


if __name__ == '__main__':
    dir_data_root = '../data'
    dataset_name = 'char_penn'
    task = 'train'
    batch_size = 16
    seq_len = 80
    valid_len = 40
    rawdataset = RawDataset(dir_data_root, dataset_name, task, seq_len, valid_len)
    embed()
    total = 0
    for _ in rawdataset:
        total += 1
    print(total)

