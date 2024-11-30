'''
This module will contain the RNN models (encoder + decoder) used to generate 
a sequence of gameplay elements when provided a sequence of song time stamps
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class StepSelectorEncoder(nn.Module):
    def __init__(self, pattern_set_size, emb_size, hidden_size):
        '''
        pattern_set_size  - number of patterns 
        emb_size    - size of vector which represents a pattern
        hidden_size - size of hidden layers within the LSTM's
        '''
        super(StepSelectorEncoder, self).__init__()
        #equivalent to vocab size in NLP
        self.pattern_set_size = pattern_set_size
        #size of vectors which represent a pattern
        self.emb_size = emb_size 
        #size of hidden vector which represents the information at a given time step of LSTM execution
        self.hidden_size = hidden_size

        #TODO: should padding_idx param be set??
        self.emb = nn.Embedding(pattern_set_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, X):
        '''
        X - shape is (N, s) where
                N = batch size ?
                S = length of longest sequence in the batch
        '''
        #look up the embedding
        wordemb = self.emb(X)
        #wordemb is shape (N, S, emb_size)

        #h is shape (N,S,hidden_size)
        #out is shape (1, N, hidden_size)
        h, out = self.lstm(wordemb) 
        #h is shape (S, hidden_size) where S is the sequence length
        
        #amax squishes to shape (1, hidden_size)
        #mean squishes to shape (1, hidden_size) 
        features = torch.cat([torch.amax(h, dim=1), 
                              torch.mean(h, dim=1)], axis=-1)
        #features is shape (2 * hidden_size)
        return features, out #features is the context vector, i.e. initial hidden state for decoder?
    
class StepSelectorDecoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        '''
        output_size - the size of the set of tokens (?)
        hidden_size - size of info vector related to all information we have encountered so far

        note: hidden_size is equal to the size of the <features> vector outputted by the encoder 
        '''
        super(StepSelectorDecoder, self).__init__(self)
        self.token_output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True) # Batch_size x Sequence Length x Hidden Size
        #layer to compute probabilities across all tokens
        self.fc = nn.Linear(hidden_size, output_size) # out: batch x seq x output_size

    def forward(self, encoder_out, encoder_hidden, target=None):
        max_seq_len = encoder_out.size(1) #TODO: check if this is right 
        batch_size = encoder_out.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        i = 0 
        #each output sequence should be as long as the input sequence
        while i < max_seq_len:
            #call forward_step to generate the next token and hidden state



            #update the decoder's input by...
            #1 - if a target vector is given, apply teacher-forcing 
            #2 - use the highest probability token as the next input 
            pass
            i += 1

        #turn the decoder outputs into tensor??
        #return outputs and hidden state

    def forward_step(self):
        pass






if __name__ == "__main__":
    #
    a = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    b = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
    print(a.shape)
    c = torch.amax(a, dim=0)
    d = torch.mean(b, dim=1)
    print(c.shape)
    print(d.shape)
    print(torch.cat((c,d),axis=-1).shape)

    #model = StepSelectorEncoder()
    #print(model)

