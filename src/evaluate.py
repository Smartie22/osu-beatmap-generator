from step_selector import StepSelectorEncoder
from step_selector import StepSelectorDecoder
from torch import tensor
from json import load
import torch
import os
import preprocessing

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

    
    _, predictions = torch.topk(out_d, 1) #TODO convert logits into token prediction
    
    hit_objs = preprocessing.index_hitobject_convert(predictions.view(-1)) #.view() call is used to flatten into 1-D tensor
    
    return hit_objs 



if __name__ == "__main__":
    #NOTE: maybe we want to pass in command line arguments which are paths to a song??

    #example usage
    testing = tensor([[2, 3, 1], [4, 7, 6], [10, 9, 8]])
    vals, indices = torch.topk(testing, 1)
    print(indices.shape)
    indices = indices.view(-1)
    print(indices)
    print(indices.shape)
#    currpath = os.path.dirname()
#    path_hyper = os.path.join(currpath, 'hyper-params-selector')
#    path_encoder = os.path.join(currpath, 'encoder.pt')
#    path_decoder = os.path.join(currpath, 'decoder.pt')
#    encoder, decoder = load_weights(path_hyper, path_encoder, path_decoder)
#
#    if encoder == None and decoder == None:
#        raise Exception("Both encoder AND decoder failed to load properly")
#    elif encoder == None:
#        raise Exception("Encoder failed to load properly")
#    elif decoder == None:
#        raise Exception("Decoder failed to load properly")
#     
#    input = tensor([])
#    sequence = evaluate_selector(input, encoder, decoder)