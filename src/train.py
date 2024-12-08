'''
This module will be where all training is structured and done. Will import from the other modules.
'''

import torch
from torch.utils.data import DataLoader
from preprocessing import collate_batch_selector
import preprocessing


#TODO: do we need to prepend/append beginning of map for the timestamps??
def get_accuracy(encoder, decoder, mapping_enc, mapping_dec, dataset, max=1000):
    """
    Calculate the accuracy of our model
    """
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_batch_selector)

    for i, (x, t) in enumerate(dataloader):
        # x represents the tokenized time-stamp input sequence, 
        # t represents the tokenized target hitobject sequence

        encoder_out, encoder_hd = encoder(x)                          #TODO: test
        decoder_out, decoder_hd, _ = decoder(encoder_out, encoder_hd) #TODO: test
        
        #TODO: determine the accuracy across all tokens generated and their respective targets
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        
        #NOTE: decoder_out is a list containing:
        #   list of probabilities
        #i.e. each element in decoder_out is a list of probabilities
        num_total = 0
        num_correct = 0
        pad_idx = 3
        while num_total < len(t) and num_total < len(decoder_out):
            _, prediction_idx = decoder_out[i].topk(1) #determine the index of the highest probability token
            if prediction_idx == pad_idx: #NOTE: idk if we should break upon finding a padding token (I think this is fine probably, unless our model generates a padding token in the middle of a map for some reason -seb)
                break 
            if prediction_idx == t[i]:
                num_correct += 1
            num_total += 1

        #return accuracy
        return num_correct / num_total 
        


def train_models():
    pass