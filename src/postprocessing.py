#TODO: define functions which convert a sequence output from our model and format a .osu file with relevant info
import os
import random
import librosa
import torch


def create_map(song_path, timestamps:torch.Tensor, hitobjects:torch.Tensor):

    song = librosa.load(song_path)[0]
    bpm = int(librosa.beat.beat_track(y=song)[0].item())

    slider_mult = 1.2 #hard coded for now
    beat_len = 60000/bpm
    slider_veloc = 1 #because we only use one timing point (and it is uninherited), this is fixed to 1

    #hitobjects = hitobjects.tolist()
    #timestamps = timestamps.tolist()

    with open("osu_generated_beatmap.osu", 'w') as map:
        # create the general section
        map.writelines(['osu file format v9\n',
                        '\n',
                        "[General]\n",
                        f'AudioFilename: {os.path.basename(song_path)}\n',
                        "AudioLeadIn: 2000\n",
                        "PreviewTime: 100\n",
                        "Countdown: 0\n",
                        "SampleSet: Soft\n",
                        "StackLeniency: 0.7\n",
                        "Mode: 0\n",
                        "LetterboxInBreaks: 0\n"])
        
        # create the metadata section
        map.writelines(["\n", "[Metadata]\n",
                        f'Title:{os.path.splitext(os.path.basename(song_path))[0]}\n',
                        "Artist:\n",
                        "Creator: Some Broken Neural Network\n",
                        "Version: beatmap generation diff\n",
                        "Source:\n",
                        "Tags:\n"])

        # create the difficulty section
        map.writelines(["\n", "[Difficulty]\n",
                        "HPDrainRate:3\n",
                        "CircleSize:5\n",
                        "OverallDifficulty:8\n",
                        "ApproachRate:9\n",
                        "SliderMultiplier:1.2\n", #keep the same as hard coded value above, too lazy to change this rn
                        "SliderTickRate:2\n"]) 

        # create the timing points section
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        map.writelines(["\n", "[TimingPoints]\n",
                       f"0,{60000/bpm},4,0,0,100,1,0\n"]) #TODO: consider rounding instead of taking the floor


        # create the hitobject section
        map.writelines(["\n", "[HitObjects]\n"])
        i = 0
        end = 0 # used to store the ending timestamp of the last spinner object
        pairs = list(zip(timestamps, hitobjects))
        while i < len(pairs):
            pair = pairs[i]
            ms = pair[0] + 2000
            ho = pair[1]

            line = ho.split(',')
            x, y = line[0], line[1]
            types = {'c': 0b00000001, 'l': 0b00000010, 's': 0b00001000}
            t = types[line[2]]
            #BANDAID FIX BELOW FOR SLIDERS (so we dont have other objects being displayed during a slider)
            if t == 0b00001000: #spinner
                x, y = 256,192
                if end < ms: # if the last spinner object already ended place down the spinner
                    end = ho[3]
                    map.write(f"{x},{y},{int(ms)},{t},0,{end}\n")
            if t == 0b00000001: #circle note
                map.write(f"{x},{y},{int(ms)},{t},0\n")
            if t == 0b00000010: #slider
                #TODO: determine how long the slider is on the screen for, remove all future objects that exist within the duration of the slider
                slider_info = ho.split('|')[1].split(',')
#                print("line is", line)
                slider_slides = int(line[4]) #integer
                slider_len = float(line[5]) #decimal
                slider_time = slider_slides * (slider_len / (slider_mult*100*slider_veloc)*beat_len)

                j = i+1
                count = 0
                stop_search = False
                #find all the following hitobjects that 
                while j < len(pairs) and not stop_search:
                    nextpair = pairs[j]
                    nextms = nextpair[0]
                    if nextms > ms+slider_time:
                        stop_search = True
                    count+=1
                    j+=1

                i+=count
                map.write(f"{x},{y},{int(ms)},{t},0{',' + ','.join(line[3:]) if line[2] == 'l' else ''}\n")\

            i+=1

                
#            map.write(f"{x},{y},{int(ms)},{t},0{',' + ','.join(line[3:]) if line[2] == 'l' else ''}\n")
#            break #temp

#example test
#o = random.sample(["46,43,c,-1", "44,108,c,-1", "44,172,c,-1", "44,237,c,-1", "67,297,l,B|139:270|161:184,1,130", "193,82,l,B|256:53|322:81,1,130", "315,209,l,B|387:209,2,65", "255,236,c,-1", "196,208,l,B|123:208,2,65", "402,136,c,-1", "360,86,c,-1"], 10)
#t = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#file = os.path.join(os.path.dirname(__file__), "../test/audio.opus")
#create_map(file, t, o)