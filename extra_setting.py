import torch
import torch.nn.functional as F
import sklearn
import numpy as np


class Sigmas(torch.nn.Module):
    def __init__(self, n_channel, image_size, **kwargs):
        torch.nn.Module.__init__(self)
        self.sigmas = torch.nn.Parameter(torch.randn(n_channel, image_size, image_size))

    def forward(self):

        sigmas = self.sigmas
        sigmas = sigmas.cuda()
        sigmas = F.sigmoid(sigmas)

        return sigmas