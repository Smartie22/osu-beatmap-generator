import evaluate
import postprocessing
import os
import json
import train

def make_hyper_params():
    curr_dir = os.path.dirname(__file__)
    name = "hyper-params"
    name = os.path.join(curr_dir, name)
    datapath = os.path.join(curr_dir, '..', 'data')

    n_buckets = 1000
    emb_size = 200
    hidden_size_d = 200
    hidden_size_e = 200
    dct = {}
    dct['n_buckets'] = n_buckets 
    dct['emb_size'] = emb_size 
    dct['hidden_size_d'] = hidden_size_d
    dct['hidden_size_e'] = hidden_size_e
    with open(name, 'w') as outfile:
        json.dump(dct, outfile)

    # You still have to create the encoder dictionary again if the num buckets are not the same
    train.create_vocab_open_token_files(curr_dir, datapath, n_buckets)


def run():
    #path of this file
    currdir = os.path.dirname(__file__)

    #create path to song
    usrinput = input("Please type the path to the song:\n")
    filtered_path = os.path.normpath(os.path.join(currdir, usrinput))
    print("filtered path is:", filtered_path)
    song_path = filtered_path

    print("Creating paths...")
    #create all paths to model related stuff
    encoder_path = os.path.join(currdir, 'encoder.pt')
    decoder_path = os.path.join(currdir, 'decoder.pt')
    encoder_dict_path = os.path.join(currdir, 'test_encoder_tokens_to_idx.json')
    decoder_dict_path = os.path.join(currdir, 'test_indices.json')
    hyper_path = os.path.join(currdir, 'hyper-params') #NOTE: PLEASE PUT HYPER PARAMS IN SRC DIRECTORY WITH THIS NAME !!!!!!!!!!!!!!

    #load our model, evaluate, then perform post processing to create the map
    print("Running model...")
    timestamps, hitobjs = evaluate.eval(song_path, hyper_path, encoder_path, decoder_path, encoder_dict_path, decoder_dict_path)
    print("Generating your map...")
    postprocessing.create_map(song_path, timestamps, hitobjs)
    print("Map generation completed!")

if __name__ == "__main__":
    make_hyper_params()
    run()
