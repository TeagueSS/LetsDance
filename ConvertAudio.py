import logging

import librosa
import numpy as np
import librosa.feature


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
        self.audio, self.sampleRate = librosa.load(self.audio_path)  # Use the actual file path

        self.loadFile()

    def loadFile(self):
        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sampleRate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sampleRate)
        # Sharing the info we grabbed
        print("Tempo:", tempo)
        print("Beat times:", beat_times)
        # Time to get all of our other Data!

        # Building our TempoGram
        self.tempogram = librosa.feature.tempogram(y=self.audio, sr=self.sampleRate)
        # Getting our Tempogram ratio ->
        self.tempogram_ratio = librosa.feature.tempogram_ratio(tg=self.tempogram, sr=self.sampleRate)
        #
        librosa.onset.onset_detect(y=self.audio, sr=self.sampleRate, units='time')
        self.onset_strength = librosa.onset.onset_strength(y=self.audio, sr=self.sampleRate)
        # self.times = librosa.times_like(self.onset_strength, sr=self.sampleRate)
        # self.onset_frames = librosa.onset.onset_detect(onset_envelope=self.onset_strength, sr=self.sampleRate)
        # Getting the cleaned chromaGraph Data
        self.chromagram_stft = librosa.feature.chroma_stft(y=self.audio, sr=self.sampleRate)


    #def build superflexOnset(self):

    #def build cleanedAudioChromagram(self):
    #

    ###TODO:
    #   get the audio data at a point in the audio file, from
    def getAudioDataAt(self, time: float):
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

    def create_audio_map(self, start_ms: int, end_ms: int):
        """
        Create an audio map (e.g., spectrogram) for a specific segment of an audio file.

        Args:
            audio_path (str): Path to the audio file.
            start_ms (int): Start time in milliseconds.
            end_ms (int): End time in milliseconds.

        Returns:
            np.ndarray: A spectrogram or other frequency-domain representation of the audio segment.
        """

        # Load the audio file
        audio = self.audio
        sr = self.sampleRate

        # Convert millisecond timings to sample indices
        start_sample = int(float(start_ms) / 1000 * sr)
        end_sample = int(float(end_ms) / 1000 * sr)

        # Ensure the indices are valid
        if start_sample < 0 or end_sample > len(audio) or start_sample >= end_sample:
            raise ValueError(f"Invalid audio indices: start={start_sample}, end={end_sample}, len(audio)={len(audio)}")

        # Extract the audio segment
        audio_segment = audio[start_sample:end_sample]

        # Ensure the segment is not empty or too short
        if len(audio_segment) == 0:
            raise ValueError("Extracted audio segment is empty.")
        if len(audio_segment) < 1024:  # Minimum window size for STFT
            audio_segment = np.pad(audio_segment, (0, 1024 - len(audio_segment)), mode="constant")

        # Generate a spectrogram using Short-Time Fourier Transform (STFT)
        n_fft = 1024  # Window size
        hop_length = n_fft // 4  # Hop length for STFT
        spectrogram = np.abs(librosa.stft(audio_segment, n_fft=n_fft, hop_length=hop_length))

        return spectrogram

    def convertAudioFrame(self, time: float, windowSize: int = 1024):

        #Load the audio file
        #audio, sampleRate = librosa.load(songPath, sr=None)

        #Convert time to sample index
        sampleIndex = int(time * self.sampleRate)

        #Extract the audio around desired time
        startIndex = max(0, sampleIndex - windowSize // 2)
        endIndex = min(len(self.audio), sampleIndex + windowSize // 2)
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
