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
        #x represents the time-stamp sequence, t represents the target hitobject sequence
        #x is pre-tokenized, t is pre-tokenized
        input_token = preprocessing.time_tok_convert(x, mapping_enc, encoder.num_buckets)
        input_token.insert(0, 0) #prepend start of map token
        #TODO: x is padded, we can't just append end of map to the end.. 

        #(NOTE: lab10 avoids this by having the dataset already contain the indices instead of the tokens themselves, 
        # may need to reconsider how we initialize the beatmap-dataset)
        input_token.append(1) #append end of map token 

        target_token = preprocessing.hitobject_tok_convert(t, mapping_dec) #TODO: test 
#        target_token.append(1) #TODO: if t is padded, then we can not just append to the end (same problem as above)
        

        encoder_out, encoder_hd = encoder(input_token)                #TODO: test
        decoder_out, decoder_hd, _ = decoder(encoder_out, encoder_hd) #TODO: test
        
        #TODO: determine the accuracy across all tokens generated and their respective targets
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        
        #look at what the contents of decoder_out actually is (is it the token or the probability i forgot)
        #iterate over each element in decoder_out and compare it to target_token

        num_total = 0
        num_correct = 0
        pad_idx = 3
        while num_total < len(target_token) and num_total < len(decoder_out):
            #TODO: double check if this is right
            if decoder_out[i] == pad_idx: #idk if we should break upon finding a padding token
                break 
            if decoder_out[i] == target_token[i]:
                num_correct += 1
            num_total += 1
        #return accuracy
        return num_correct / num_total # TODO: Consider what to do with collate batch. Consider what to do with encoder and decoder

        


def train_models():
    pass