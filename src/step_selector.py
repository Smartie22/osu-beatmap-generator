'''
This module will contain the RNN models (encoder + decoder) used to generate 
a sequence of gameplay elements when provided a sequence of song time stamps
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class StepSelector(nn.Module):
    def __init__(self, pattern_set_size, emb_size, hidden_size):
        '''
        pattern_set_size  - number of patterns 
        emb_size    - size of vector which represents a pattern
        hidden_size - size of hidden layers within the LSTM's
        '''
        super().__init__()
        #equivalent to vocab size in NLP
        self.pattern_set_size = pattern_set_size
        #size of vectors which represent a pattern
        self.emb_size = emb_size 
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(pattern_set_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, X):
        #look up the embedding
        wordemb = self.emb(X)
        #we are only interested in the hidden layer for the decoder
        h, out = self.lstm(wordemb) 
        features = torch.cat([torch.amax(h, dim=1), 
                              torch.mean(h, dim=1)], axis=-1)
        return features



if __name__ == "__main__":
    model = StepSelector()
    print(model)
