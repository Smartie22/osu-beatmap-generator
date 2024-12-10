from step_selector import StepSelectorEncoder
from step_selector import StepSelectorDecoder
from torch import tensor
import torch
import os
from json import load

def load_weights(path_hyper, path_encoder, path_decoder):
    encoder = None
    decoder = None
    with (open(path_hyper)) as hyper_param_file:
        hyper_params = load(hyper_param_file)
        encoder = StepSelectorEncoder(hyper_params['n_buckets'], hyper_params['emb_size'], hyper_params['hidden_size_e'])
        decoder = StepSelectorDecoder(hyper_params['output_size_d'], hyper_params['hidden_size_d'])
        encoder.load_state_dict(torch.load(path_encoder, weights_only=True))
        decoder.load_state_dict(torch.load(path_decoder, weights_only=True))
        encoder.eval()
        decoder.eval()
    return encoder, decoder

def evaluate_selector(X, encoder, decoder):
    '''
    X - tensor of size (1, L), representing the sequence of timestamps to be converted into hit objects 
        L represents the sequence length
    '''
    #perform forward pass on our models
    out_hd_e, out_e = encoder(X)
    out_d, _, _ = decoder(out_e, out_hd_e)

    #TODO convert logits into token prediction
    
    #TODO convert tokens into hit objects
    
    #TODO return sequence of hitobjects
    return None 



if __name__ == "__main__":
    #NOTE: maybe we want to pass in command line arguments which are paths to a song??

    #example usage
    currpath = os.path.dirname()
    path_hyper = os.path.join(currpath, 'hyper-params-selector')
    path_encoder = os.path.join(currpath, 'encoder.pt')
    path_decoder = os.path.join(currpath, 'decoder.pt')
    encoder, decoder = load_weights(path_hyper, path_encoder, path_decoder)

    if encoder == None and decoder == None:
        raise Exception("Both encoder AND decoder failed to load properly")
    elif encoder == None:
        raise Exception("Encoder failed to load properly")
    elif decoder == None:
        raise Exception("Decoder failed to load properly")
     
    input = tensor([])
    sequence = evaluate_selector(input, encoder, decoder)