import os
import torch
from torch.autograd import Variable
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import pickle
import pandas as pd
import numpy as np
import logging

"""
Note: The meaning of batch_size in PTB is different from that in MNIST example. In MNIST, 
batch_size is the # of sample data that is considered in each iteration; in PTB, however,
it is the number of segments to speed up computation. 

The goal of PTB is to train a language model to predict the next word.
"""
logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

def save_model(model, args):
    # with open(args.dir_model+'/'+args.log+"_model.pt", 'wb') as f:
    #     torch.save(model, f)
    model_name = args.dir_model+'/'+args.log+"_model.pt"
    torch.save({'state_dict': model.state_dict()}, model_name)
    logging.info('Save model!')

def load_model(model, args):
    # with open(args.dir_model+'/'+args.log+"_model.pt", 'rb') as f:
    #     model = torch.load(f)
    model_name = args.dir_model+'/'+args.log+"_model.pt"
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def save_visual_info(visual_info_all, args):
    if args.continue_train:
        with open(args.dir_model+'/'+args.log+"_visual_info.pt", 'rb') as f:
            visual_info_old = torch.load(f)
            logging.info("load old visual info!")
        visual_info_all = visual_info_old + visual_info_all

    with open(args.dir_model+'/'+args.log+"_visual_info.pt", 'wb') as f:
        torch.save(visual_info_all, f)
    logging.info('Save visual info!')

def record(end_time, path_results, dataset_name, optimizer, key_size_dim, VHDropout, levels, batch_size, \
        num_epochs, lr, num_subblocks, en_res, temp_attn, loss_test, ppl_test, num_parameters_train, log_info):
    if os.path.isfile(path_results+'/results.csv'):
        df = pd.read_csv(path_results+'/results.csv', header=0)
    else:
        df = pd.DataFrame(columns=['end_time', 'dataset_name', 'optimizer', 'key_size_dim', 'VHDropout', 'levels', \
                'batch_size', 'num_epochs', 'lr', 'num_subblocks', 'en_res', 'temp_attn', 'loss_test',
                'ppl_test', 'num_parameters_train', 'log_info'])

    df.loc[df.shape[0]+1] = [end_time, dataset_name, optimizer, key_size_dim, VHDropout, levels, batch_size, \
            num_epochs, lr, num_subblocks, en_res, temp_attn, loss_test, ppl_test, num_parameters_train, log_info]
    df.reset_index(drop=True, inplace=True)

    df.to_csv(path_results+'/results.csv', index=False)
    logging.info("Saved the results")

def draw_attn(visual_info, epoch, args, *dic):
    dir_root = "{}/pngs/{}".format(args.dir_model, args.log)
    if not os.path.isdir(dir_root):
        os.system('mkdir {}'.format(dir_root))
        for sentence_idx in range(len(visual_info[2][0])):
            # if not os.path.isdir(dir_root+'/'+str(sentence_idx)):
            os.system('mkdir {}'.format(dir_root+'/'+str(sentence_idx)))

    if len(dic) != 0:
        filename_sentences = dir_root + '/' + 'sentences.txt'
        if os.path.exists(filename_sentences):
            os.remove(filename_sentences)
        dictionary = dic[0]
        for i in range(2):
            sample_sent_word_list, target_sent_word_list = [], []
            sample_sent_idx_list = visual_info[0].detach().cpu().numpy()[i]
            target_sent_idx_list = visual_info[1].detach().cpu().numpy()[i]
            for idx in sample_sent_idx_list:
                sample_sent_word_list.append(dictionary.idx2word[idx])
            for idx in target_sent_idx_list:
                target_sent_word_list.append(dictionary.idx2word[idx])
            with open(filename_sentences, 'a+') as f:
                f.write(str(i)+'\n')
                f.write(' '.join(sample_sent_word_list)+'\n')
                f.write(' '.join(target_sent_word_list)+'\n')

    for block_idx in range(len(visual_info[2])):
        for sentence_idx in range(len(visual_info[2][0])):
            plt.figure()
            plt.imshow(visual_info[2][block_idx][sentence_idx])
            new_ticks = np.append([0],np.arange(4, args.seq_len, 5))
            plt.xticks(new_ticks, new_ticks+1)
            plt.yticks(new_ticks, new_ticks+1)
            plt.colorbar()
            plt.savefig(dir_root+'/'+str(sentence_idx)+'/'+str(block_idx)+'_'+str(epoch)+'.png', format='png')
            plt.close()
    logging.info("Draw attention picture.")


# if __name__ == '__main__':
#     record(0, "/home/haohy/yanyan/TCN_based/TCANet/results", 'Adam', 10, 0, 1, 1, 1, 1, 1,True, True, 1, 1, 1, 'hello')
