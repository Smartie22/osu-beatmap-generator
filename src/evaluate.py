from step_selector import StepSelectorEncoder
from step_selector import StepSelectorDecoder
from torch import tensor
from json import load
import torch
import os
import preprocessing
import librosa
import numpy as np

def load_weights(path_hyper, path_encoder, path_decoder):
    encoder = None
    decoder = None
    hyper_params = None
    with (open(path_hyper)) as hyper_param_file:
        hyper_params = load(hyper_param_file)
        encoder = StepSelectorEncoder(hyper_params['n_buckets'], hyper_params['emb_size'], hyper_params['hidden_size_e'])
        decoder = StepSelectorDecoder(hyper_params['output_size_d'], hyper_params['hidden_size_d'])
        encoder.load_state_dict(torch.load(path_encoder, weights_only=True))
        decoder.load_state_dict(torch.load(path_decoder, weights_only=True))
        encoder.eval()
        decoder.eval()
    return encoder, decoder, hyper_params

def load_convert_dict(path_time_ind, path_ind_obj):
    time_ind = None
    ind_obj = None
    with open(path_time_ind) as dictfile:
        time_ind = load(dictfile)
    with open(path_ind_obj) as dictfile:
        ind_obj = load(dictfile)
    return time_ind, ind_obj

def evaluate_selector(X, encoder, decoder, ind_obj_d):
    '''
    X - tensor of size (1, L), representing the sequence of timestamps to be converted into hit objects 
        L represents the sequence length
    '''
    #perform forward pass on our models
    out_hd_e, out_e = encoder(X)
    out_d, _, _ = decoder(out_e, out_hd_e)

    
    _, predictions = torch.topk(out_d, 1) #TODO convert logits into token prediction
    
    hit_objs = preprocessing.index_hitobject_convert(predictions.view(-1), ind_obj_d) #.view() call is used to flatten into 1-D tensor
    
    return hit_objs 

def eval(path_song, path_hyper, path_encoder, path_decoder, path_encoder_dict, path_decoder_dict):
    '''
    
    '''
    #preprocessing and data loading steps
    encoder, decoder, hyper_params = load_weights(path_hyper, path_encoder, path_decoder)
    num_buckets = hyper_params['n_buckets']
    time_ind_e, ind_obj_d = load_convert_dict(path_encoder_dict, path_decoder_dict)
#    melfilter = preprocessing.process_audio(path_song) #NOTE: needed if we use our own CNN model
    y, sr = librosa.load(path_song)

    timestamp_seq = librosa.onset.onset_detect(y=y, sr=sr, units='time') #TODO: use librosa or call our own CNN model to determine offsets and generate a sequence of timestamps to place notes
    timestamp_seq = (timestamp_seq * 1000).round()
    timestamp_seq = np.ndarray.tolist(timestamp_seq)

    #TODO: convert timestamp_seq into tokens as input for the encoder
    timestamp_seq_idx = preprocessing.time_tok_convert(timestamp_seq, num_buckets, time_ind_e)
    #convert from seconds to milliseconds
    hitobj_seq = evaluate_selector(timestamp_seq_idx, encoder, decoder, ind_obj_d)

    return timestamp_seq, hitobj_seq

if __name__ == "__main__":
    #NOTE: maybe we want to pass in command line arguments which are paths to a song??
    pass
    #example usage
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