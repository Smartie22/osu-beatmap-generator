'''
This module will be where all training is structured and done. Will import from the other modules.
'''

import torch
from torch.utils.data import DataLoader
from preprocessing import collate_batch_selector
import preprocessing


#TODO: do we need to prepend/append beginning of map for the timestamps??
def get_accuracy(encoder, decoder, dataset, max=1000):
    """
    Calculate the accuracy of our model
    """
    num_cor, total = 0, 0
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_batch_selector)

    for i, (x, t) in enumerate(dataloader):
        pass # TODO: Consider what to do with collate batch. Consider what to do with encoder and decoder


def train_models():
    pass