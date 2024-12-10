import evaluate
import postprocessing
import os

def run():
    #path of this file
    currdir = os.path.dirname(__file__)

    #create path to song
    usrinput = input("Please type the path to the song:\n")
    filtered_path = os.path.normpath(usrinput) #TODO: check this over and see if it works lol
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
    run()