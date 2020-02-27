import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from IPython import embed

class PositionEmbedding(nn.Module):
    """Grpah position embedding."""
    def __init__(self, emb_size, key_size, value_size, seq_len):
        super(PositionEmbedding, self).__init__()
        self.linear_query = nn.Linear(emb_size, key_size)
        self.linear_keys = nn.Linear(emb_size, key_size)
        self.linear_values = nn.Linear(emb_size, value_size)
        self.flatten_position_linear = nn.Linear(seq_len*seq_len, emb_size)
        # self.
        self.sqrt_key_size = math.sqrt(key_size)

    def direction_param(self, k, q):
        pass

    def weight_pos_graph(self, query, keys):
        """Weighted position graph for sentence order.
        Args:
            query: (N, T, emb_size)
            keys: (N, T, emb_size)
        """
        self.n_graph=query.size(1)

    def forward(self, input):
        # input is dim (N, T, emb_size) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.size(1))] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask).cuda()
        
        # date : 2019/11/4
        # auther : Yan Wang
        # mask = mask.bool() 
        # end

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)


        # temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp = weight_pos_graph(query, keys) # shape: (N, T, T)


        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        # date : 2019/11/4
        # auther : Yan Wang
        # reture without "input"
        # return torch.cat((input, temp), dim=2) # shape: (N, T, emb_size + value_size)
        return temp # shape: (N, T, emb_size)
        # end


# class PositionEmbedding(nn.Module):
#     """Position embedding for word's raw order number."""
#     # def __init__(self, input_output_size, emb_size, seq_len):
#     def __init__(self, emb_size, seq_len):
#         super(PositionEmbedding, self).__init__()
#         self.seq_len = seq_len
#         self.position_layer = nn.Linear(seq_len*2, emb_size)
#         self.init_weights()

#     def init_weights(self):
#         self.position_layer.weight.data.normal_(0, 0.01)
#         self.position_layer.bias.data.fill_(0)

#     def forward(self, x):
#         # x: [batchsize, seq_len]
#         seq_len_now = x.size(1)
#         if x.size(1) < self.seq_len*2:
#             # embed(header="PositionEmbedding")
#             x = F.pad(x, (0,self.seq_len*2-x.size(1)), "constant", 0)
#         position_info = self.position_layer(x.float()) # position_info: [batchsize, emb_size]
#         position_embeded = position_info.unsqueeze(1).repeat(1, seq_len_now, 1) # position_embeded: [batchsize, seq_len, emb_size]
#         return position_embeded


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_seq_len):
#         """初始化。
        
#         Args:
#             d_model: 一个标量。模型的维度，论文默认是512
#             max_seq_len: 一个标量。文本序列的最大长度
#         """
#         super(PositionalEncoding, self).__init__()
        
#         # 根据论文给的公式，构造出PE矩阵
#         position_encoding = np.array([
#           [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
#           for pos in range(max_seq_len)])
#         # 偶数列使用sin，奇数列使用cos
#         position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
#         position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

#         # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
#         # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
#         # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
#         # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
#         pad_row = torch.zeros([1, d_model])
#         position_encoding = torch.cat((pad_row, torch.tensor(position_encoding).float()))
#         # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
#         # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
#         # self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
#         # self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)
#         self.position_encoder = nn.Embedding.from_pretrained(position_encoding, freeze=True)

#     def forward(self, input_len_list):
#         """神经网络的前向传播。

#         Args:
#           input_len_list: torch.Tensor,  一个张量，为输入句子的长度的序列。

#         Returns:
#           返回这一批序列的位置编码，进行了对齐。
#         """
#         # 这里range从1开始也是因为要避开PAD(0)的位置
#         input_pos = torch.tensor(
#           [list(range(1, input_len + 1)) for input_len in input_len_list])
#         input_pos = input_pos.cuda() if input_len_list.is_cuda else input_pos
#         return self.position_encoder(input_pos)

# def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
#     ''' Sinusoid position encoding table '''

#     def cal_angle(position, hid_idx):
#         return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

#     def get_posi_angle_vec(position):
#         return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

#     sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#     if padding_idx is not None:
#         # zero vector for padding dimension
#         sinusoid_table[padding_idx] = 0.
#     # sinusoid_table: [n_position, d_hid]
#     return torch.FloatTensor(sinusoid_table)


