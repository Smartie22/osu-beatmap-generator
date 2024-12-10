#TODO: define functions which convert a sequence output from our model and format a .osu file with relevant info
import os
import random
import librosa

def create_map(song_path, timestamps, hitobjects):

    song = librosa.load(song_path)[0]
    bpm = int(librosa.beat.beat_track(y=song)[0].item())

    with open("osu_generated_beatmap.osu", 'w') as map:
        # create the general section
        map.writelines(['osu file format v9\n',
                        '\n',
                        "[General]\n",
                        f'AudioFilename: {os.path.basename(song_path)}\n',
                        "AudioLeadIn: 0\n",
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
                        "Version:\n",
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
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        map.writelines(["\n", "[TimingPoints]\n",
                       f"0,{int(60000/bpm)},4,0,0,100,0,0\n"])


        # create the hitobject section
        map.writelines(["\n", "[HitObjects]\n"])
        for ms, ho in zip(timestamps, hitobjects):
            line = ho.split(',')
            x, y = line[0], line[1]
            types = {'c': 0b00000001, 'l': 0b00000010, 's': 0b00001000}
            t = types[line[2]]
            map.write(f"{x},{y},{ms},{t},0{',' + ','.join(line[3:]) if line[2] == 'l' else ''}\n")

o = random.sample(["46,43,c,-1", "44,108,c,-1", "44,172,c,-1", "44,237,c,-1", "67,297,l,B|139:270|161:184,1,130", "193,82,l,B|256:53|322:81,1,130", "315,209,l,B|387:209,2,65", "255,236,c,-1", "196,208,l,B|123:208,2,65", "402,136,c,-1", "360,86,c,-1"], 10)
t = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
file = os.path.join(os.path.dirname(__file__), "../test/audio.opus")
create_map(file, t, o)