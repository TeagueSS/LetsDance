import logging

import librosa
import numpy as np
import librosa.feature
import librosa.display
from matplotlib import pyplot as plt


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
        self.times = librosa.times_like(self.onset_strength, sr=self.sampleRate)
        self.onset_frames = librosa.onset.onset_detect(onset_envelope=self.onset_strength, sr=self.sampleRate)
        # Getting the cleaned chromaGraph Data
        self.chromagram_stft = librosa.feature.chroma_stft(y=self.audio, sr=self.sampleRate)



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

    def view_audio_map(self, start_ms: int, end_ms: int):
        """
        Visualizes raw audio feature data for the given time range using separate graphs.

        Parameters:
            start_ms (int): Start time in milliseconds.
            end_ms (int): End time in milliseconds.
        """
        # Convert milliseconds to seconds
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        # Create our frame slices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)

        # Slice each feature
        tempogram_section = self.tempogram[start_frame, end_frame]
        chromagram_stft_section = self.chromagram_stft[start_frame, end_frame]
        #onset_strength_section = self.onset_strength[start_frame, end_frame]

        # Setting up our graph:
        fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

        # Make Graphs of our Features ->
        librosa.display.specshow(tempogram_section, y_axis='Tempogram', x_axis='time', ax=ax[0])
        ax[0].set(title='tempogram_section-STFT')

        librosa.display.specshow(chromagram_stft_section, y_axis='chroma', x_axis='time', ax=ax[1])
        ax[1].set(title='chromagram_stft_section')

        #librosa.display.specshow(onset_strength_section, y_axis='Strength', x_axis='time', ax=ax[2])
        #ax[2].set(title='onset_strength_section')

        #TODO we need to fix the plotting here















        # Fitting our Data
        plt.tight_layout()
        # Showing our Graph ->
        plt.show()


    def create_audio_map(self, start_ms: int, end_ms: int):
        """
        Extracts and stacks audio feature subsections for the given time range.

        Parameters:
            start_ms (int): Start time in milliseconds.
            end_ms (int): End time in milliseconds.

        Returns:
            tuple: A dictionary of individual audio features and a concatenated numpy array for TensorFlow.
        """

        # Convert milliseconds to seconds
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        # Convert seconds to frame indices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)

        # Slice each feature
        tempogram_section = self.tempogram[:, start_frame:end_frame]  # (n_features, time_frames)
        tempogram_ratio_section = self.tempogram_ratio[start_frame:end_frame, 0]  # Extract 1D data
        onset_strength_section = self.onset_strength[start_frame:end_frame]  # (time_frames,)
        chromagram_stft_section = self.chromagram_stft[:, start_frame:end_frame]  # (n_features, time_frames)

        # Ensure compatible shapes
        min_frames = min(
            tempogram_section.shape[1],
            tempogram_ratio_section.shape[0],
            onset_strength_section.shape[0],
            chromagram_stft_section.shape[1],
        )

        # Trim all sections to the minimum frame length
        tempogram_section = tempogram_section[:, :min_frames]
        tempogram_ratio_section = tempogram_ratio_section[:min_frames]
        onset_strength_section = onset_strength_section[:min_frames]
        chromagram_stft_section = chromagram_stft_section[:, :min_frames]

        # Create our dictonary before stacking
        featuresPreStack = {
            "tempogram": tempogram_section,  # (n_features, min_frames)
            "tempogram_ratio": tempogram_ratio_section,  # (1, min_frames)
            "onset_strength": onset_strength_section,  # (1, min_frames)
            "chromagram_stft": chromagram_stft_section,  # (n_features, min_frames)
        }


        # Reshape 1D arrays to 2D for stacking
        tempogram_ratio_section = tempogram_ratio_section[np.newaxis, :]  # Shape (1, min_frames)
        onset_strength_section = onset_strength_section[np.newaxis, :]  # Shape (1, min_frames)

        # Create dictionary of features
        features = {
            "tempogram": tempogram_section,  # (n_features, min_frames)
            "tempogram_ratio": tempogram_ratio_section,  # (1, min_frames)
            "onset_strength": onset_strength_section,  # (1, min_frames)
            "chromagram_stft": chromagram_stft_section,  # (n_features, min_frames)
        }

        # Concatenate all features along the first axis for TensorFlow
        stacked_features = np.vstack([
            tempogram_section,
            tempogram_ratio_section,
            onset_strength_section,
            chromagram_stft_section
        ])  # Shape: (total_features, min_frames)

        return featuresPreStack, stacked_features

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
