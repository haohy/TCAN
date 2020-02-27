import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
sys.path.append(os.getcwd())

from model.tcanet import TCANet
from model.tcn_block import TemporalConvNet
# from model.pe import PositionEmbedding
# from model.optimizations import VariationalDropout, VariationalHidDropout, WeightDropout