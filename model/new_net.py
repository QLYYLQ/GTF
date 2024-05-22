import torch
import torch.nn as nn

from new_lsk import LSKencoder

class model(nn.Module):
    def __init__(self):
        self.encoder = LSKencoder()
        