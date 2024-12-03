import logging

import librosa
import numpy as np
import librosa.feature
import librosa.display
from matplotlib import pyplot as plt
    #Hi here
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
        # Getting all of our information once
        # Loading the audio and sample rate
        #self.audio, self.sample_rate = librosa.load(self.audio_path, sr=None)

        # Getting the duration using audio and sample rate
        self.duration = librosa.get_duration(y=self.audio, sr=self.sampleRate)
        print("Hi")
        self.loadFile()
        self.build_beat_based_audio_map()

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

    def getFrameStart(self , start_ms:int):
        start_sec = start_ms / 1000
        # Convert seconds to frame indices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        return start_frame

    def getFrameEnd(self, end_ms:int):
        end_sec = end_ms / 1000
        # Convert seconds to frame indices
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)
        return end_frame
    def get_number_of_frames_in(self, start_ms: int, end_ms: int):
        # Convert milliseconds to seconds
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        # Convert seconds to frame indices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)

        # Return the number of frames in our section
        return end_frame - start_frame

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

    def view_audio_map(self, start_sec, end_sec, tempogram_section, chromagram_stft_section,
                       chromagram_cqt_section,
                       chromagram_cens_section, onset_strength_section):
        # Create frame slices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)

        # Calculate time axis for subsections
        times = librosa.frames_to_time(np.arange(start_frame, end_frame), sr=self.sampleRate)

        # Detect onsets in the full audio
        onset_frames = librosa.onset.onset_detect(onset_envelope=self.onset_strength, sr=self.sampleRate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sampleRate)
        onset_times_section = onset_times[(onset_times >= start_sec) & (onset_times <= end_sec)]

        # Plot feature subsections
        fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)

        # Checking we have all our data
        if chromagram_stft_section is not None:
            # Plot Chromagram-STFT
            librosa.display.specshow(chromagram_stft_section, y_axis='chroma', x_axis='time', ax=ax[0], sr=self.sampleRate,
                                     cmap='inferno')
            ax[0].set(title='Chroma-STFT')
        if chromagram_cqt_section is not None:
            # Plot Chromagram-CQT
            librosa.display.specshow(chromagram_cqt_section, y_axis='chroma', x_axis='time', ax=ax[1], sr=self.sampleRate,
                                     cmap='inferno')
            ax[1].set(title='Chroma-CQT')
        if chromagram_cens_section is not None:
            # Plot Chromagram-CENS
            librosa.display.specshow(chromagram_cens_section, y_axis='chroma', x_axis='time', ax=ax[2], sr=self.sampleRate,
                                     cmap='inferno')
            ax[2].set(title='Chroma-CENS')

        plt.tight_layout()
        plt.show()

        if onset_times_section is not None:
            # Plot the waveform and onset envelope
            plt.figure(figsize=(12, 6))

            # Plot the waveform subsection
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(self.audio[int(start_sec * self.sampleRate):int(end_sec * self.sampleRate)],
                                     sr=self.sampleRate, alpha=0.6)
            plt.vlines(onset_times_section, ymin=-1, ymax=1, color='r', linestyle='dashed', label='Onsets')
            plt.title("Waveform with Onset Detection (Subsection)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude")
            plt.legend()

        # Plot the onset strength envelope subsection
        if onset_times_section is not None:
            plt.subplot(2, 1, 2)
            plt.plot(times, onset_strength_section, label='Onset Strength Envelope')
            plt.vlines(onset_times_section, ymin=0, ymax=max(onset_strength_section), color='r', linestyle='dashed',
                       label='Onsets')
            plt.title("Onset Strength Envelope (Subsection)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Strength")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def build_beat_based_audio_map(self):
        """/
        Our previous implementation woudln't have provided any information on beat
        so here we are going to make an implementation just for audio beats ->
        """
        # Load audio
        y = self.audio
        sr = self.sampleRate

        # Onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # BPM (scalar feature)
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Beat positions (sequence feature)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Periodicity (Fourier Transform of onset envelope)
        fft_result = np.abs(np.fft.rfft(onset_env))
        freqs = np.fft.rfftfreq(len(onset_env), d=1 / sr)

        # Choose dominant rhythm frequencies (optional)
        dominant_rhythm = freqs[np.argmax(fft_result)]

        # Saving all of these to our class
        self.tempo = tempo
        self.beat_times = beat_times
        self.dominant_rhythm = dominant_rhythm
        self.onset_env = onset_env
        #return features
        #print(features)

    def create_and_view_subsection_audio_map(self, start_ms: int, end_ms: int):
        """
        Visualizes subsections of audio features for the given time range.
        Matches the provided example but only uses feature subsections.

        Parameters:
            start_ms (int): Start time in milliseconds.
            end_ms (int): End time in milliseconds.
        """
        import librosa.display
        import numpy as np
        import matplotlib.pyplot as plt

        # Convert milliseconds to seconds
        start_sec = start_ms / 1000
        end_sec = end_ms / 1000

        # Create frame slices
        start_frame = librosa.time_to_frames(start_sec, sr=self.sampleRate)
        end_frame = librosa.time_to_frames(end_sec, sr=self.sampleRate)

        # Slice features
        tempogram_section = self.tempogram[:, start_frame:end_frame]  # Tempogram subsection
        chromagram_stft_section = self.chromagram_stft[:, start_frame:end_frame]  # Chromagram-STFT subsection
        chromagram_cqt_section = librosa.feature.chroma_cqt(y=self.audio, sr=self.sampleRate)[:,
                                 start_frame:end_frame]  # Chromagram-CQT subsection
        chromagram_cens_section = librosa.feature.chroma_cens(y=self.audio, sr=self.sampleRate)[:,
                                  start_frame:end_frame]  # Chromagram-CENS subsection
        onset_strength_section = self.onset_strength[start_frame:end_frame]  # Onset strength subsection

        # Calculate time axis for subsections
        times = librosa.frames_to_time(np.arange(start_frame, end_frame), sr=self.sampleRate)

        self.view_audio_map(start_sec, end_sec, tempogram_section, chromagram_stft_section,
                            chromagram_cqt_section, chromagram_cens_section, onset_strength_section)

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

        # Detect onsets in the full audio for onset_times
        onset_frames = librosa.onset.onset_detect(onset_envelope=self.onset_strength, sr=self.sampleRate)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sampleRate)
        onset_times_section = onset_times[(onset_times >= start_sec) & (onset_times <= end_sec)]

        # Create our dictonary before stacking
        features = {
            "tempogram": tempogram_section,
            "tempogram_ratio": tempogram_ratio_section,
            "onset_strength": onset_strength_section,
            "chromagram_stft": chromagram_stft_section,
            "onset_times_section": onset_times_section
        }

        return features



    #TODO: Seperate the tensor flow mapping into a seperate function
    # -> tesor flow mapping should take in the dictonary
    #    and return a flattened out version for Tensor flow to use
    #    Create Audio map should just make a little subsection
    def create_audio_map_for_tensorflow(self, start_ms: int, end_ms: int, max_frames: int = 128):
        """
        Prepares audio feature subsections for TensorFlow, ensuring fixed sizes and normalized values.

        Parameters:
            start_ms (int): Start time in milliseconds.
            end_ms (int): End time in milliseconds.
            max_frames (int): Maximum number of frames for padding or truncation.

        Returns:
            tuple: A dictionary of processed audio features and a concatenated numpy array for TensorFlow.
        """

        # Step 1: Extract features using existing logic
        features = self.create_audio_map(start_ms, end_ms)

        # Step 2: Normalize features
        def normalize(feature):
            if feature.ndim > 1:  # For 2D features like tempogram or chromagram
                return (feature - np.mean(feature, axis=1, keepdims=True)) / (
                            np.std(feature, axis=1, keepdims=True) + 1e-8)
            else:  # For 1D features like onset strength
                return (feature - np.mean(feature)) / (np.std(feature) + 1e-8)

        for key in ["tempogram", "chromagram_stft", "onset_strength"]:
            features[key] = normalize(features[key])

        # Step 3: Pad or truncate features to fixed size
        def pad_or_truncate(feature, max_frames):
            if feature.ndim == 1:  # 1D array
                if len(feature) > max_frames:
                    return feature[:max_frames]
                return np.pad(feature, (0, max_frames - len(feature)))
            else:  # 2D array
                if feature.shape[1] > max_frames:
                    return feature[:, :max_frames]
                return np.pad(feature, ((0, 0), (0, max_frames - feature.shape[1])))

        for key in features:
            if key != "onset_times_section":  # Ignore onset times, as they are not fixed-size tensors
                features[key] = pad_or_truncate(features[key], max_frames)

        # Step 4: Combine features into a single tensor for TensorFlow
        combined_tensor = np.concatenate(
            [
                features["tempogram"].flatten(),
                features["chromagram_stft"].flatten(),
                features["onset_strength"].flatten(),
            ]
        )

        return features, combined_tensor
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
