import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions

FLAGS = None
class MultiLayerConvNet(nn.Module):
    def __init__(self, *kwargs):
        super().__init__()

        # get kwargs

