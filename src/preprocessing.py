'''
This module will contain the preprocessers that will preprocess all over the data.
'''
import json
import os

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch import tensor
import librosa
import numpy as np

import timeit

class BeatmapDataset(Dataset):
    '''
    Preprocesses the data while loading them into the dataset
    '''
    def __init__(self, root_path, obj_ind, ind_obj, num_points=10000):
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

        # Assign the obj_ind and the ind_obj
        self.obj_ind = obj_ind
        self.ind_obj = ind_obj

        #NOTE: this only imports for training set, idk if thats relevant or a problem or anything

        #create list which stores each tuple described above
        data = []

        #iterate over each beatmap in the training data from <root_path>
        num_parsed = 1
        for currdir, dirnames, filenames in os.walk(root_path):
            if num_parsed > num_points:
                break
            for name in filenames:
                if name.endswith(".opus"): # always process .osu before .mp3
                    #apply preprocessing on the audio
                    audio = self.process_audio(currdir + "/{0}".format(name))

                    for name2 in filenames: # directories for specific maps are not very large, this should be fine
                        if name2.endswith(".osu"):
                            #apply preprocessing on the beatmap, combine results
                            dct = self.process_beatmap(currdir + "/{0}".format(name2))
                            dct["Audio"] = audio
                            data.append(dct)
                            num_parsed += 1
                        if num_parsed > num_points:
                            break
                    break

        #convert list into pytorch tensor
        # data_t = tensor(data)
        self.data = data

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

    def process_audio(self, path):
        '''
        Compute mel-spectrogram using the librosa library

        params:
            path - relative path to the mp3 audio file for which we apply filtering on
            dct - dictionary to store the result of filter application
            key - the corresponding key our filter application should be mapped to in <dct> (SHOULD ALWAYS USE DEFAULT!!!!)
        '''
        audio, sr = librosa.load(path)
        stft = librosa.stft(audio)
        melfilter = librosa.feature.melspectrogram(S=stft)
        return melfilter

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
        parse through the [TimingPoints] or [HitObjects] Section of the .osu! file
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
        #TODO: should these be left as lists, or converted to tensors later ??
        TimeStamps = []
        HitObjects = []
        for line in contents:
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
                obj_key = (str(info[0]), str(info[1]), str(type), str(info[5]))
                obj_key = ','.join(obj_key)

            # If the obj_key is in the index, we assign it
            index = 2
            if obj_key in self.obj_ind:
                index = self.obj_ind[obj_key]

            TimeStamps.append(info.pop(2)) #remove the time stamp, append to timestamps
            HitObjects.append(index) #reconstruct the original line without whitespace, and without the timestamps
            lines_parsed += 1
        return (lines_parsed, (TimeStamps, HitObjects))

def create_tokens(path, tok_index_name, index_tok_name):
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
    mapping = {'<bom>': 0, '<eom>': 1, '<unk>': 2, '<pad>': 3}
    indices = {0: '<bom>', 1: '<eom>', 2: '<unk>', 3: '<pad>'}
    idx = [4]  # poor man's pointer (global index which needs to be mutated)
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
                obj_key = (str(object[0]), str(object[1]), str(type), str(object[5]))
                obj_key = ','.join(obj_key)

            #   store in dictionary
            if obj_key not in dct:
                dct[obj_key] = glob_idx[0]     
                dct2[glob_idx[0]] = obj_key
                glob_idx[0] = glob_idx[0] + 1
            line = f.readline()


def collate_batch_selector(batch):
    """ Taking lab10 as inspiration
    X - (N, L) batch of data.
    t - a (N, L) target vector.
    """
    hitobj_list = []
    label_list = []
    for d in batch:
        indices = d['HitObjects'].copy()
        label = indices.copy()
        indices.insert(0, 0) # prepend bom
        label.append(1)
        indices.append(1) # append eom
        hitobj_list.append(tensor(indices))
        label_list.append(tensor(label))

    X = pad_sequence(hitobj_list, padding_value=3).transpose(0, 1)
    t = tensor(label_list)
    return X, t


#example usage
# dir = os.path.dirname(__file__)
# filename = os.path.join(dir, '..', 'data', '2085341 BUTAOTOME - Street Journal', 'BUTAOTOME - Street Journal (Djulus) [Extra].osu')
# process_beatmap(filename)

dir = os.path.dirname(__file__)
path = os.path.join(dir, '..', 'data')
#bm = BeatmapDataset(path)

create_tokens(path, "test_tokenizer.txt", "test_indices.txt")
