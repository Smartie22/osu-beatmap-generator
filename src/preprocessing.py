'''
This module will contain the preprocessers that will preprocess all over the data.
'''
import json
import math
import os
import asyncio

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import tensor
from torch.nn.functional import normalize
import librosa
import numpy as np

import timeit
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

class BeatmapDataset(Dataset):
    '''
    Preprocesses the data while loading them into the dataset
    '''
    def __init__(self, root_path, time_ind_e, ind_time_e, obj_ind_d, ind_obj_d, num_buckets, num_points=10000, fresh_load=True, num_workers=100):
        '''
        NOTE:
            For the note selection model, we want the map described as follows: 
                {Audio, General, Difficulty, Events, TimingPoints, TimeStamps, Objects},
            where 
                <Audio> - the preprocessed song the beatmap is created for, 
                <General> and <Difficulty> are maps which contain their respective information in the osu file
                <Events> is a list of breaks in the map
                <TimingPoints> is a list of timing sections within the osu beatmap. 
                <TimeStamps> is a list of time stamps in the song (in milliseconds from the beginning), 
                <HitObjects> is a list of hitobjects corresponding to the given time stamp, 
        NOTE2:
            selection model encoder wants the sequence of time stamps and the sequence of corresponding hit objects (for teacher-forcing training)
        
        params:
            <root_path> - path to root directory of the training dataset
            <num_points> - number of training points to load from the training set found in <root_path>, default: 10,000
        '''
        super().__init__()

        # Assign the obj_ind and the ind_obj mappings
        self.time_ind_e = time_ind_e
        self.ind_time_e = ind_time_e
        self.obj_ind_d = obj_ind_d # Seems like we already had some ideas of passing in the mappings before...
        self.ind_obj_d = ind_obj_d
        # Need the number of buckets for translation in the encoder mappings.
        self.num_buckets = num_buckets

        #NOTE: this only imports for training set, idk if thats relevant or a problem or anything

        #create list which stores each tuple described above
        # data = []

        #iterate over each beatmap in the training data from <root_path>
        # num_parsed = 1
        # for currdir, dirnames, filenames in os.walk(root_path):
        #     if num_parsed > num_points:
        #         break
        #     for name in filenames:
        #         if name.endswith(".opus"): # always process .osu before .mp3
        #             #apply preprocessing on the audio
        #             audio = self.process_audio(currdir + "/{0}".format(name))

        #             for name2 in filenames: # directories for specific maps are not very large, this should be fine
        #                 if name2.endswith(".osu"):
        #                     #apply preprocessing on the beatmap, combine results
        #                     dct = self.process_beatmap(currdir + "/{0}".format(name2))
        #                     dct["Audio"] = audio
        #                     data.append(dct)
        #                     num_parsed += 1
        #                 if num_parsed > num_points:
        #                     break
        #             break

        #convert list into pytorch tensor
        # data_t = tensor(data)
        self.data = asyncio.run(self.load_data_parallel(root_path, num_points)) if fresh_load else self.load_json()
        self.data = np.array(self.data)

    def load_json(self):
        pass
    
    def process_files(self, opus_file, osu_file):
        with ThreadPoolExecutor() as executor:
            # future_audio = executor.submit(process_audio, opus_file)
            future_beatmap = executor.submit(self.process_beatmap, osu_file)

        # Collect results
        # audio = future_audio.result()
        beatmap = future_beatmap.result()
        if beatmap:
            # beatmap["Audio"] = audio
            return beatmap

    async def load_data_parallel(self, root_path, num_points):
        data = []
        tasks = []
        files = []
        batch = 24
        os.environ["OPENBLAS_NUM_THREADS"] = "24"
        

        # with ProcessPoolExecutor(max_workers=61) as executor:
        for currdir, _, filenames in os.walk(root_path):
            opus_files = [os.path.join(currdir, f) for f in filenames if f.endswith(".opus")]
            osu_files = [os.path.join(currdir, f) for f in filenames if f.endswith(".osu")]

            files.extend([(o, s) for o in opus_files for s in osu_files][:num_points])

        for i in range(0, len(files), batch):
            tasks = [asyncio.to_thread(self.process_files, *pair) for pair in files[i:i+batch]]

            results = await asyncio.gather(*tasks, return_exceptions=True) # needs to be true else exceptions will not cancel and will be stuck in hell
            data.extend(r for r in results if not isinstance(r, Exception) and r is not None)
        return data

    def __len__(self):
        '''
        return the size of the dataset
        '''
        return self.data.size

    def __getitem__(self, i):
        '''
        retrieve the ith data sample in the dataset
        '''
        return self.data[i]

    def process_beatmap(self, path):
        '''
        Processes a .osu file into a dictionary containing
        '''
        dct = {}
        with open(path, 'r', encoding='utf-8') as f:
            contents = f.readlines()[2:]
            i = 0
            while i < len(contents):
                line = contents[i].strip('\n')
                match line:
                    case "[General]":
                        (lines_parsed, parsed_contents) = self.parse_gen_diff(contents[i+1:])
                        dct["General"] = parsed_contents
                        i += lines_parsed
                        try:
                            if dct["General"]["Mode"] != "0":
                                return None
                        except Exception:
                            pass
                        finally:
                            pass
                    case "[Difficulty]":
                        (lines_parsed, parsed_contents) = self.parse_gen_diff(contents[i+1:])
                        dct["Difficulty"] = parsed_contents
                        i += lines_parsed
                    case "[Events]":
                        (lines_parsed, parsed_contents) = self.parse_events(contents[i+1:])
                        dct["Events"] = parsed_contents
                        i += lines_parsed
                    case "[TimingPoints]":
                        (lines_parsed, parsed_contents) = self.parse_timingPoints_hitObjects(contents[i+1:])
                        dct["TimingPoints"] = parsed_contents
                        i += lines_parsed
                    case "[HitObjects]":
                        (lines_parsed, parsed_contents) = self.split_hitObjects(contents[i+1:])
                        dct["TimeStamps"] = parsed_contents[0]
                        dct["HitObjects"] = parsed_contents[1]
                        i += lines_parsed   # Parse for tokens?
                    case _:
                        i += 1
        return dct

    def parse_gen_diff(self, contents):
        '''
        parse through the [General] or [Difficulty] Section of the .osu! file
        '''
        lines_parsed = 1 # account for the section header line we already skipped
        parsed_contents = {}
        for line in contents:
            if line == "\n": #reached the end of the section
                break
            line = line.strip()
            parts = line.split(':')
            parsed_contents[parts[0].strip()] = parts[1].strip()
            lines_parsed += 1
        return (lines_parsed, parsed_contents)

    def parse_events(self, contents):
        '''
        parse through the [Events] Section of the .osu! file, storing only the break events in chronological order
        '''
        lines_parsed = 1
        parsed_contents = []
        for line in contents:
            if line == "\n":
                break
            line = line.strip()
            if line[0] == "2" or line[0] == "B":
                parsed_contents.append(line)
            lines_parsed += 1
        return (lines_parsed, parsed_contents)


    def parse_timingPoints_hitObjects(self, contents):
        '''
        parse through the [TimingPoints] Section of the .osu! file
        '''
        lines_parsed = 1
        parsed_contents = []
        for line in contents:
            if line == "\n":
                break
            line = line.strip()
            parsed_contents.append(line)
            lines_parsed += 1
        return (lines_parsed, parsed_contents)


    def split_hitObjects(self, contents):
        lines_parsed = 1
        TimeStamps = []
        HitObjects = []
        for line in contents:
            try:
                if line == "\n":
                    break
                line = line.strip()
                info = line.split(",") # contains stuff about the hitobjects
                # x @ index 0, y @ idx 1, type @ idx 3, obj_param @ idx 5
                type = ''
                bitstring = bin(int(info[3]))
                if bitstring[-1] == '1':
                    type = 'c'
                elif bitstring[-2] == '1':
                    type = 'l'
                elif bitstring[-4] == '1':
                    type = 's'

                if type == 'c':
                    obj_key = (info[0], info[1], type, '-1')
                    obj_key = ','.join(obj_key)
                elif type == 's':
                    obj_key = ('-1', '-1', type, '-1')
                    obj_key = ','.join(obj_key)
                else:
                    obj_key = (str(info[0]), str(info[1]), str(type), ','.join(info[5:]))
                    obj_key = ','.join(obj_key)

                # If the obj_key is in the index, we assign it
                index = 2
                if obj_key in self.obj_ind_d:
                    index = self.obj_ind_d[obj_key]
            except Exception:
                lines_parsed += 1
                continue

            TimeStamps.append(int(info.pop(2))) #remove the time stamp, append to timestamps
            HitObjects.append(index) #reconstruct the original line without whitespace, and without the timestamps
            lines_parsed += 1
            

        # Append and prepend start and end tokens.
        HitObjects.append(1)

        TimeStamps_indices = time_tok_convert(TimeStamps, self.num_buckets, self.time_ind_e)

        return (lines_parsed, (TimeStamps_indices, HitObjects)) # Previously (TimeStamps, HitObjects)

def time_tok_convert_helper(element, num_buckets, time_ind_e):
    """
    Take an element of the vector and replace/return it with a token
    """
    if element == 1:
        return time_ind_e[f"{1 - (1 / num_buckets)}, 1.0"]
    i = math.floor(element * num_buckets)
    k = f"{i * 1 / num_buckets}, {(i + 1) * 1 / num_buckets}"
    return time_ind_e[k]

def time_tok_convert(timestamps, num_buckets, time_ind_e):
    """ Given a sequence of <timestamps> we convert it to a sequence of tokens. Assume that the obj to token file is
    made.

    note that <timestamps> is a python list!!, TODO: should we do checks to ensure we arn't making a tensor of tensors..??
    """
    norm_stamps = tensor(timestamps, dtype=torch.float64)
    # normalize
    maxval = torch.max(norm_stamps)
    minval = torch.min(norm_stamps)
    norm_stamps = (norm_stamps - minval)/ (maxval - minval)
    # Put the stamps into buckets
    func = lambda x: (time_tok_convert_helper(x))
    tokens = norm_stamps.apply_(lambda x: (time_tok_convert_helper(x, num_buckets, time_ind_e)))

    #TODO: reconsider whether we need both start AND end of song tokens pre+appended, or just the end of song token appended
    return torch.cat((tensor([0]), tokens, tensor([1]))).tolist() # Prepend and append the start and end tokens


def process_audio(path):
    '''
    Compute mel-spectrogram using the librosa library

    params:
        path - relative path to the mp3 audio file for which we apply filtering on
        dct - dictionary to store the result of filter application
        key - the corresponding key our filter application should be mapped to in <dct> (SHOULD ALWAYS USE DEFAULT!!!!)
    '''
    audio, sr = librosa.load(path, sr=None)
        
    # use a more efficient STFT and Mel filter computation
    stft = librosa.stft(audio, n_fft=1024, hop_length=512, win_length=1024, window='hann')
    melfilter = librosa.feature.melspectrogram(
        y=audio, sr=sr, S=np.abs(stft)**2, n_fft=1024, hop_length=512, n_mels=128, power=2.0
    )

    return melfilter

def convert_index_hitobject(element, ind_obj_d):
    """
    helper to convert a token back to it's corresponding hit object
    """
#    return ind_obj_d[f"{element}"]
#    print("element is", element)
    retval = ind_obj_d[str(element)]
    return retval 

def index_hitobject_convert(indices, ind_obj_d):
    """
    converts a sequence of <tokens> into a sequence of hit objects
    """
    tokens_lst = indices
    if type(indices) == torch.Tensor:
        tokens_lst = indices.tolist()
    i = 0
    while i < len(tokens_lst):
        tokens_lst[i] = convert_index_hitobject(tokens_lst[i], ind_obj_d)
        i += 1

    return tokens_lst 
    

def convert_hitobject_token(self, element):
    """
    Helper to convert a single hitobject element given by <element> into it's corresponding token via <mapping>
    """
    return self.obj_ind_d[f"{element}"]

# TODO: please testing !!! also note that these functions might not even get used
def hitobject_tok_convert(self, hitobjects):
    """
    Convert the sequence of <hitobjects> into a sequence of tokens through the conversion found in <mapping>
    """
    objects = tensor(hitobjects)
    helper = lambda x: (self.convert_hitobject(x))
    tokens = objects.apply_(helper)
    return torch.cat((tokens, tensor([1]))) # Prepend and append the start and end tokens
def create_tokens_decoder(path, tok_index_name, index_tok_name):
    '''
    Creates a mapping between gameplay object and number,

    NOTE: 0 will be the <bom> (beginning of map) token
          1 will be the <eom> (end of map) token
          2 will be the <unk> (unknown pattern) token
          3 will be the <pad> (padding) token

    NOTE 2: indices correspond to the one hot vector representation,
            i.e. the index given by the dictionary represents the index in the
            one hot vector that is set as 1
    '''
    # (<x>, <y>, <type>, <object_params>)
    mapping = {'<bom>': 0, '<eom>': 1, '<pad>': 2}
    indices = {0: '<bom>', 1: '<eom>', 2: '<pad>'}
    idx = [3]  # poor man's pointer (global index which needs to be mutated)
    with open(tok_index_name, 'w') as outfile:
        for currdir, dirnames, filenames in os.walk(path):
            for name in filenames:
                if name.endswith('.osu'):
                    #parse file
                    parse_objects(currdir, name, mapping, indices, idx)

        json.dump(mapping, outfile)

    with open(index_tok_name, 'w') as outfile:
        json.dump(indices, outfile)

def parse_objects(currdir, name, dct, dct2, glob_idx):
    '''
    takes a path to a .osu file, parses the hitobjects section, and
    updates the dictionary given by <dct> with key (<x> <y> <type> <object_params>),
    where...
        x             - x position on gameplay field, '-1' for 's' types
        y             - y position on gameplay field, '-1' for 's' types
        type          - one of {'c', 'l', 's'}
        object_params - unique to sliders ('l' type), '-1' for objects of type 'c' and 's'
    '''
    path = os.path.join(currdir, name)
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != '[HitObjects]\n':
            line = f.readline()
        #we are in the hitobjects section
        line = f.readline() #skip the [HitObjects] header
        while(line != '\n' or line):
            try:
                #   split line to obtain x,y,type,object_params
                if not line:
                    break
                object = line.strip().split(',')
                # x @ index 0, y @ idx 1, type @ idx 3, obj_param @ idx 5
                type = ''
                bitstring = bin(int(object[3]))
                if bitstring[-1] == '1':
                    type = 'c'
                elif bitstring[-2] == '1':
                    type = 'l'
                elif bitstring[-4] == '1':
                    type = 's'
                
                if type == 'c':
                    obj_key = (object[0], object[1], type, '-1')
                    obj_key = ','.join(obj_key)
                elif type == 's':
                    obj_key = ('-1', '-1', type, '-1')
                    obj_key = ','.join(obj_key)
                else:
                    obj_key = (str(object[0]), str(object[1]), str(type), ','.join(object[5:]))
                    obj_key = ','.join(obj_key)

                #   store in dictionary
                if obj_key not in dct and type != '': #NOTE: type != '' is a bandaid fix
                    dct[obj_key] = glob_idx[0]     
                    dct2[glob_idx[0]] = obj_key
                    glob_idx[0] = glob_idx[0] + 1
                line = f.readline()
            except Exception:
                line = f.readline()

def create_tokens_encoder(tok_index_name, index_tok_name, num_buckets=10000):
    """ Tokenizer for the encoder. Each bucket: represents a period of time that is of length 1/10000, with the whole
    field being normalized to fit into those 10000 buckets.
    TODO: Make num_buckets dynamic based on the song's bpm instead of having it fixed.
    """
    mapping = {'<bom>': 0, '<eom>': 1, '<unk>': 2, '<pad>': 3}
    indices = {0: '<bom>', 1: '<eom>', 2: '<unk>', 3: '<pad>'}
    for i in range(num_buckets):
        obj = f"{i * 1/num_buckets}, {(i + 1) * 1/num_buckets}"
        index = i + 4
        mapping[obj] = index
        indices[index] = obj

    # Dump it into a json
    with open(tok_index_name, 'w') as outfile:
        json.dump(mapping, outfile)
    with open(index_tok_name, 'w') as outfile:
        json.dump(indices, outfile)



def collate_batch_selector(batch):
    """ Taking lab10 as inspiration
    X - (N, L) batch of data, input to the encoder, includes padding.
    t - a (N, L) target vector with padding.

        N represents batch size
        L represents length of the longest sequence
    """
    time_seq_list = []
    label_list = []
    for d in batch:
        #obtain labels
        label = d['HitObjects'].copy() # No need to prepend/append start/end tokens. It's already done for you!
        label_list.append(tensor(label))

        #obtain timestamps
        time_seq = d['TimeStamps'].copy()
        time_seq_list.append(tensor(time_seq))

    #pad the sequences
    X = pad_sequence(time_seq_list, padding_value=3).transpose(0, 1).type(torch.LongTensor)
    t = pad_sequence(label_list, padding_value=3).transpose(0, 1).type(torch.LongTensor)


#    #torch.set_printoptions(profile="full")
#    #z=2
#    #print("in collate_fn, t at index", z, "is:", t[z])
    
    
    return X, t



if __name__ == "__main__": # This is used to ensure some unnecessary code does NOT execute...
    # example usage
#    dir = os.path.dirname(__file__)
#    filename = os.path.join(dir, '..', 'data', '2085341 BUTAOTOME - Street Journal', 'BUTAOTOME - Street Journal (Djulus) [Extra].osu')
#    process_beatmap(filename 
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, '..', 'data')

    path_tok_ind_e = os.path.join(dir, "test_encoder_tokens_to_idx.json")
    path_ind_tok_e = os.path.join(dir, "test_encoder_idx_to_token.json")
    path_tok_ind_d = os.path.join(dir, "test_tokenizer.json")
    path_ind_tok_d = os.path.join(dir, "test_indices.json")

    fd_tok_ind_e = open(path_tok_ind_e,)
    fd_ind_tok_e = open(path_ind_tok_e,)
    fd_tok_ind_d = open(path_tok_ind_d,)
    fd_ind_tok_d = open(path_ind_tok_d,)
    tok_ind_e = json.load(fd_tok_ind_e)
    ind_tok_e = json.load(fd_ind_tok_e)
    tok_ind_d = json.load(fd_tok_ind_d)
    ind_tok_d = json.load(fd_ind_tok_d)

    create_tokens_encoder(os.path.join(dir, "test_encoder_tokens_to_idx.json"),
                          os.path.join(dir, "test_encoder_idx_to_token.json"))
    create_tokens_decoder(path, os.path.join(dir, "test_tokenizer.json"), os.path.join(dir, "test_indices.json"))

    bm = BeatmapDataset(path, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, 10000)

    # NOTE: testing conversion func
    # load mapping
#     tok_to_idx_path = os.path.join(dir, 'test_encoder_tokens_to_idx.json')
#     mapping = {}
#     with open(tok_to_idx_path) as jfile:
#        mapping = json.load(jfile)
#     print(mapping)
#     lst = [2, 5, 14, 18]
#     print(time_tok_convert_helper(2/18,10000,mapping))
#     print(time_tok_convert_helper(5/18,10000,mapping))
#     print(time_tok_convert_helper(14/18,10000,mapping))
#     print(time_tok_convert_helper(18/18,10000,mapping))
#    
#     expected = [time_tok_convert_helper(2/18,10000,mapping), time_tok_convert_helper(5/18,10000,mapping), time_tok_convert_helper(14/18,10000,mapping), time_tok_convert_helper(18/18,10000,mapping)]
#     print("expected result is:", expected)
#     res = time_tok_convert(lst, mapping, 10000)
#     print("actual result is:", res)

    #NOTE: testing what hitobjects looks like
    #print("bm looks like:")
#    for z in range(36):
#        print("z is", z, "beatmap is", bm[z])

    fd_tok_ind_e.close()
    fd_ind_tok_e.close()
    fd_tok_ind_d.close()
    fd_ind_tok_d.close()