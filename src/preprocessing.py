'''
This module will contain the preprocessers that will preprocess all over the data.
'''
import os
from torch.utils.data import Dataset
from torch import tensor
import librosa
import numpy as np

class BeatmapDataset(Dataset):
    '''
    Preprocesses the data while loading them into the dataset
    '''
    def __init__(self, root_path, num_points=10000):
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

        #TODO: check that this works lol
        #NOTE: this only imports for training set, idk if thats relevant or a problem or anything

        #create list which stores each tuple described above
        data = []

        #iterate over each beatmap in the training data from <root_path>
        i = 0
        for currdir, dirnames, filenames in os.walk(root_path):
            if i > num_points:
                break
            for name in filenames:
                if name.endswith(".osu"): # always process .osu before .mp3
                    #apply preprocessing on the beatmap
                    dct = process_beatmap(currdir + "/{0}".format(name))
                    for name2 in filenames: # directories for specific maps are not very large, this should be fine
                        if name2.endswith(".mp3"):
                            #apply preprocessing on the audio, then combine results
                            dct["Audio"] = process_audio(currdir + "/{0}".format(name2))
                            data[i] = dct
                            i+=1
                break

        #convert list into pytorch tensor
        data_t = tensor(data)
        self.data = data_t

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



def process_audio(path):
    '''
    Compute mel-spectrogram using the librosa library

    params:
        path - relative path to the mp3 audio file for which we apply filtering on
        dct - dictionary to store the result of filter application
        key - the corresponding key our filter application should be mapped to in <dct> (SHOULD ALWAYS USE DEFAULT!!!!)
    '''
    audio, sr = librosa.load(path)
    stft = librosa.stft(audio)
    melfilter = librosa.feature.melspectrogram(y=stft)
    return melfilter

def process_beatmap(path):
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
                    (lines_parsed, parsed_contents) = parse_gen_diff(contents[i+1:]) 
                    dct["General"] = parsed_contents
                    i += lines_parsed
                case "[Difficulty]":
                    (lines_parsed, parsed_contents) = parse_gen_diff(contents[i+1:]) 
                    dct["Difficulty"] = parsed_contents
                    i += lines_parsed
                case "[Events]":
                    (lines_parsed, parsed_contents) = parse_events(contents[i+1:])
                    dct["Events"] = parsed_contents
                    i += lines_parsed
                case "[TimingPoints]":
                    (lines_parsed, parsed_contents) = parse_timingPoints_hitObjects(contents[i+1:])
                    dct["TimingPoints"] = parsed_contents
                    i += lines_parsed
                case "[HitObjects]":
                    (lines_parsed, parsed_contents) = parse_timingPoints_hitObjects(contents[i+1:])
                    dct["TimeStamps"] = parsed_contents[0]
                    dct["HitObjects"] = parsed_contents[1]
                    i += lines_parsed
                case _:
                    i += 1
    return dct
                    
def parse_gen_diff(contents):
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

def parse_events(contents):
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


def parse_timingPoints_hitObjects(contents):
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


def split_hitObjects(contents):
    lines_parsed = 1
    TimeStamps = []
    HitObjects = []
    for line in contents:
        if line == "\n":
            break
        line = line.strip()
        info = line.split(",")
        TimeStamps.append(info.pop(2)) #remove the time stamp, append to timestamps
        HitObjects.append(",".join(info)) #reconstruct the original line without whitespace, and without the timestamps
        lines_parsed += 1
    return (lines_parsed, (TimeStamps, HitObjects))
























#NOTE: below functions may be unnecessary
def parse_hitObjects(contents):
    '''
    parse through the [HitObjects] Section of the .osu! file
    
    note that the way an element is parsed depends on the note type
    '''
    lines_parsed = 1
    parsed_contents = []

    for line in contents:
        if line == "\n":
            break
        line = line.strip()
        info = line.split(',') #type is stored as third element

        #convert value into bit string, determine the type of the current note
        type = 'c' if format(info[3], '08b')[0] == '1' else ('l' if format(info[3], '08b')[1] == '1' else 's')
        note_content = {}

        match type:
            case 'c': #normal circle notes
               note_content = parse_hitobj_circle(info) 
            case 'l': #sliders (osu! std long note equivalent)
               note_content = parse_hitobj_slider(info) 
            case 's': #spinners
               note_content = parse_hitobj_spinner(info) 

        parsed_contents.append(note_content)
    return (lines_parsed, parsed_contents)

def parse_hitobj_circle(info):
    '''

    '''
    #NOTE: may not be necessary
    pass

def parse_hitobj_slider(info):
    '''

    '''
    #NOTE: may not be necessary

def parse_hitobj_spinner(info):
    '''

    '''
    #NOTE: may not be necessary



#example usage
dir = os.path.dirname(__file__)
filename = os.path.join(dir, '..', 'data', '2085341 BUTAOTOME - Street Journal', 'BUTAOTOME - Street Journal (Djulus) [Extra].osu')
process_beatmap(filename)
