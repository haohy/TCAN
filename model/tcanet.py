import torch
import logging
from torch import nn
import torch.nn.functional as F
from model.tcn_block import TemporalConvNet
# from model.pe import PositionEmbedding
# from model.optimizations import VariationalDropout, WeightDropout

from IPython import embed

logging.basicConfig( \
    level = logging.INFO, \
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

class TCANet(nn.Module):

    def __init__(self, emb_size, input_output_size, num_channels, seq_len, num_sub_blocks, temp_attn, nheads, en_res,
                 conv, key_size, kernel_size=2, dropout=0.3, wdrop=0.0, emb_dropout=0.1, tied_weights=False, 
                 dataset_name=None, visual=True):
        super(TCANet, self).__init__()
        self.temp_attn = temp_attn
        self.dataset_name = dataset_name
        self.num_levels = len(num_channels)
        self.word_encoder = nn.Embedding(input_output_size, emb_size)
        if dataset_name == 'mnist':
            self.word_encoder = nn.Embedding(256, emb_size)
        # self.position_encoder = PositionEmbedding(emb_size, seq_len)
        self.tcanet = TemporalConvNet(input_output_size, emb_size, num_channels, \
            num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, kernel_size, visual=visual, dropout=dropout)
        # self.tcanet = WeightDropout(self.tcanet, self.get_conv_names(num_channels), wdrop)
        # self.drop = VariationalDropout(emb_dropout) # drop some embeded features, e.g. [16,80,600]->[16,80,421]
        self.drop = nn.Dropout(emb_dropout)
        self.decoder = nn.Linear(num_channels[-1], input_output_size)
        if tied_weights:
            if self.dataset_name != 'mnist':
                self.decoder.weight = self.word_encoder.weight
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.word_encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def get_conv_names(self, num_channels):
        conv_names_list = []
        for level_i in range(len(num_channels)):
            conv_names_list.append(['network', level_i, 'net', 0, 'weight_v'])
            conv_names_list.append(['network', level_i, 'net', 4, 'weight_v'])
        return conv_names_list

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # input: [batchsize, seq_len]
        # emb = self.drop(torch.cat([self.word_encoder(input), self.position_encoder(input)], dim=2))
        if self.dataset_name == 'mnist':
            # emb = self.drop(self.word_encoder(input))
            if self.temp_attn:
                y, attn_weight_list = self.tcanet(input) # input should have dimension (N, C, L)
                # y, attn_weight_list = self.tcanet(emb.transpose(1, 2)) # input should have dimension (N, C, L)
                o = self.decoder(y[:, :, -1])
                # return F.log_softmax(o, dim=1).contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
                return F.log_softmax(o, dim=1).contiguous()
            else:
                y = self.tcanet(input) # input should have dimension (N, C, L)
                # y = self.tcanet(emb.transpose(1, 2)) # input should have dimension (N, C, L)
                o = self.decoder(y[:, :, -1])
                return F.log_softmax(o, dim=1).contiguous()

        emb = self.drop(self.word_encoder(input))
        if self.temp_attn:
            y, attn_weight_list = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous(), [attn_weight_list[0], attn_weight_list[self.num_levels//2], attn_weight_list[-1]]
        else:
            y = self.tcanet(emb.transpose(1, 2))
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous()

