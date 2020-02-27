import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
from utils.dataset import RawDataset 
from utils.utils import save_model, load_model, save_visual_info, record, draw_attn

def load_data(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus):
    dataset = RawDataset(dir_data_root, dataset_name, task, seq_len, valid_len, is_corpus)
    return dataset

def load_dataloader(dataset, batch_size, num_workers=4, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
