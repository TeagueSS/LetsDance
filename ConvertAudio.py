import librosa
import numpy as np

class AudioHandler:

    """/
        Meant to  process audio files using the librosa library. ->

        It takes an audio file path as input, loads the audio data,
        and then has methods for getting audio data at given times or
        frames. getAudioDataAt should return data for a frame time, and
        getCorrespondingAudioData should use threading to get audio data
        for all of the provided frame timings
    """

    # Creating a constructor
    def __init__(self, audio_path: str):
        # Storing our Audio path for later
        self.audio_path = audio_path
        # Saving our y and SR
        self.audio, self.sampleRate = self.loadFile(self.audio_path)

    def loadFile(self, audio_path):
        # Load the audio file
        #audio, sampleRate
        audio, sampleRate = librosa.load(audio_path, sr=None)

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sampleRate)
        beat_times = librosa.frames_to_time(beat_frames, sr=sampleRate)
        # Sharing the info we grabbed
        print("Tempo:", tempo)
        print("Beat times:", beat_times)
        return audio, sampleRate


    ###TODO:
    #   get the audio data at a point in the audio file, from
    def getAudioDataAt(self , time:float):
        # Convert time to frame index
        frame_index = librosa.time_to_frames(time, sr=self.sampleRate)

        # Get the audio data at that point in time
        audio_segment = self.audio[frame_index:]
        # Printing that we got the data ->
        print(f"Audio data at {time} seconds:", audio_segment)
        return audio_segment


    ##TODO
    #   Write a method to take in an array of times
    #   and return an array of frame data at those times
    #   (Type can be a tuple or just a 2d Array of audio_segments)
    def getCorrespondingAudioData(self, arrayOfTimings):
        #TODO: If possible use threading:
        # Look in ConvertAudio -> convertVideoIntoSyncedFrames()
        # That method uses multi threading

        # Creating a list to add audio segments back to ->
        audio_segments = []





        # Remove this return
        return 0
    
    def convertAudioFrame(self, time: float , windowSize: int = 1024):
        
        #Load the audio file
        #audio, sampleRate = librosa.load(songPath, sr=None)

        #Convert time to sample index
        sampleIndex = int(time * self.sampleRate)

        #Extract the audio around desired time
        startIndex = max(0, sampleIndex- windowSize //2)
        endIndex = min(len(self.audio), sampleIndex + windowSize //2)
        audioSegment = self.audio[startIndex:endIndex]

        # Ensure the audio segment is not empty or too short
        if len(audioSegment) == 0:
            raise ValueError("Extracted audio segment is empty.")
        if len(audioSegment) < windowSize:
            audioSegment = np.pad(audioSegment, (0, windowSize - len(audioSegment)), mode="constant")


        #Convert the audio to a frequency
        hop_length = windowSize // 4  # Explicitly set hop length
        spectrumData = np.abs(librosa.stft(audioSegment, n_fft=windowSize, hop_length=hop_length))

        #Return the data as an array
        return spectrumData
    
    
        


