import librosa
import numpy as np

def loadFile(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)



    # Get tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    print("Tempo:", tempo)
    print("Beat times:", beat_times)
    return y, sr


    print(loadFile("/Users/teaguesangster/Code/Python/CS450/DataSetup/downloads/Just Dance Hits： Only Girl (In The World) by Rihanna [12.9k]_audio.mp3"))



