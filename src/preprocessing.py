'''
This module will contain the preprocessers that will preprocess all over the data.
'''
import os
from torch.utils.data import Dataset
import librosa
import numpy as np

class BeatmapDataset(Dataset):
    '''
    Preprocesses the data while loading them into the dataset
    '''
    def __init__(self):
       super().__init__()

    def __len__(self):
        '''
        return the size of the dataset
        '''
        #TODO: override len method 
        pass

    def __getitem__(self, i):
        '''
        retrieve the ith data sample in the dataset
        '''
        #TODO: override getitem method so array style indexing is possible
        pass
def preprocess_audio():
    '''
    use librosa to compute mel-spectrograms
    '''
    pass

def process_beatmap(path):
    map = {}
    with open(path, 'r', encoding='utf-8') as f:
        contents = f.readlines()[2:]
        i = 0 
        while i < len(contents):
            line = contents[i].strip('\n')
            match line:
                case "[General]":
                    (lines_parsed, parsed_contents) = parse_gen_diff(contents[i+1:]) 
                    map["General"] = parsed_contents
                    i += lines_parsed
                case "[Difficulty]":
                    (lines_parsed, parsed_contents) = parse_gen_diff(contents[i+1:]) 
                    map["Difficulty"] = parsed_contents
                    i += lines_parsed
                case "[Events]":
                    (lines_parsed, parsed_contents) = parse_events(contents[i+1:])
                    map["Events"] = parsed_contents
                    i += lines_parsed
                case "[TimingPoints]":
                    (lines_parsed, parsed_contents) = parse_timingPoints_hitObjects(contents[i+1:])
                    map["TimingPoints"] = parsed_contents
                    i += lines_parsed
                case "[HitObjects]":
                    (lines_parsed, parsed_contents) = parse_timingPoints_hitObjects(contents[i+1:])
                    map["HitObjects"] = parsed_contents
                    i += lines_parsed
                case _:
                    i += 1
    print(map)
                    
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
    return (lines_parsed, parsed_contents)

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
    #TODO
    pass

def parse_hitobj_slider(info):
    '''

    '''
    #TODO

def parse_hitobj_spinner(info):
    '''

    '''
    #TODO

#example
dir = os.path.dirname(__file__)
filename = os.path.join(dir, '..', 'data', '2085341 BUTAOTOME - Street Journal', 'BUTAOTOME - Street Journal (Djulus) [Extra].osu')
process_beatmap(filename)
