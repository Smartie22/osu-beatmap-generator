#TODO: define functions which convert a sequence output from our model and format a .osu file with relevant info
import os
import random

def create_map(song_path, song_name, bpm, timestamps, hitobjects):
    with open("osu_generated_beatmap.osu", 'w') as map:
        # create the general section
        map.writelines(['osu file format v9\n',
                        '\n',
                        "[General]\n",
                        f'AudioFilename: {song_name}\n',
                        "AudioLeadIn: 0\n",
                        "PreviewTime: 100\n",
                        "Countdown: 0\n",
                        "SampleSet: Soft\n",
                        "StackLeniency: 0.7\n",
                        "Mode: 0\n",
                        "LetterboxInBreaks: 0\n"])
        
        # create the metadata section
        map.writelines(["\n", "[Metadata]\n",
                        f'Title:{song_name}\n',
                        "Artist:\n",
                        "Creator: Some Broken Neural Network\n",
                        "Version:0\n",
                        "Source:\n",
                        "Tags:\n"])

        # create the difficulty section
        map.writelines(["\n", "[Difficulty]\n",
                        "HPDrainRate:3\n",
                        "CircleSize:4\n",
                        "OverallDifficulty:3\n",
                        "ApproachRate:5\n",
                        "SliderMultiplier:1.2\n",
                        "SliderTickRate:2\n"])

        # create the timing points section
        map.writelines(["\n", "[TimingPoints]\n"])


        # create the hitobject section
        map.writelines(["\n", "[HitObjects]\n"])
        for ms, ho in zip(timestamps, hitobjects):
            line: tuple[str] = ho.split(',')
            x, y = line[0], line[1]
            types = {'c': 0b00000001, 'l': 0b00000010, 's': 0b00001000}
            t = types[line[2]]
            map.write(f'{x},{y},{ms},{t},0,{",".join(line[3:])}\n')

o = random.sample(["46,43,c,-1", "44,108,c,-1", "44,172,c,-1", "44,237,c,-1", "67,297,l,B|139:270|161:184", "193,82,l,B|256:53|322:81", "315,209,l,B|387:209", "255,236,c,-1", "196,208,l,B|123:208", "402,136,c,-1", "360,86,c,-1"], 10)
t = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
create_map('~\\..\\data\\0a7a8f8b7126cf33605db69928adcb48\\audio.opus', 'audio.opus',
           100, t, o)