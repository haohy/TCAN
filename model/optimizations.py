import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from IPython import embed

class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.3):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout
    
    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x

        mask_matrix = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1-self.dropout)
        with torch.no_grad():
            mask = mask_matrix / (1 - self.dropout)
            mask = mask.expand_as(mask_matrix)
        
        return mask * x


class WeightDropout(nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDropout, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()

    def _setup(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path)-1):
                if isinstance(path[i], int):
                    module = module[path[i]]
                elif isinstance(path[i], str):
                    module = getattr(module, path[i])
            try:
                w = getattr(module, name_w)
            except:
                continue
                # embed(header="WeightDropout")
            del module._parameters[name_w]
            module.register_parameter(name_w+'_raw', nn.Parameter(w.data))
        
    def _setweights(self):
        for path in self.weights:
            module = self.module
            name_w = path[-1]
            for i in range(len(path)-1):
                if isinstance(path[i], int):
                    module = module[path[i]]
                elif isinstance(path[i], str):
                    module = getattr(module, path[i])
            raw_w = getattr(module, name_w + '_raw')
            w = F.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(module, name_w, w)
    
    def forward(self, *args, **kwargs):
        self._setweights()
        return self.module.forward(*args, **kwargs)

class VariationalHidDropout(nn.Module):
    def __init__(self, vhdropout):
        super(VariationalHidDropout, self).__init__()
        self.vhdrop = vhdropout

    def reset_mask(self, x):
        # x: [N, C, L]
        m = x.data.new(x.size(0), x.size(1), 1)._bernoulli(self.vhdrop)
        with torch.no_grad():
            mask = m / (1-self.vhdrop)
            self.mask = mask

        return self.mask

    def forward(self, x):
        if not self.training or self.vhdrop == 0:
            return x
        assert self.mask is not None, "You need to reset mask before using VariationalHidDropout"
        mask = self.mask.expand_as(x)
        return mask * x
