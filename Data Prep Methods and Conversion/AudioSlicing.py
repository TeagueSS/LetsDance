import numpy as np
import tensorflow as tf
import librosa

class AudioFrameProcessor:
    def __init__(self, fps, beat_times, onset_env=None):
        """
        Initialize the audio frame processor.
        :param fps: Frames per second of the video.
        :param beat_times: Array of beat times in seconds.
        :param onset_env: Optional onset envelope array for finer slicing.
        """
        self.fps = fps
        self.beat_times = np.array(beat_times)
        self.onset_env = onset_env
        self.normalized_features = None  # To hold normalized features

    def process_audio_features(self, audio_duration):
        """
        Process and slice audio features to match the frame timing.
        Normalize them using TensorFlow's Normalization layer.
        :param audio_duration: Total duration of the audio in seconds.
        """
        num_frames = int(audio_duration * self.fps)
        frame_times = np.linspace(0, audio_duration, num=num_frames, endpoint=False)
        feature_matrix = []

        expected_frames = int(audio_duration * self.fps)
        print("***************************************************************")
        print("***************************************************************")

        print(f"Expected frames: {expected_frames}, Actual frames: {len(frame_times)}")

        for start_time in frame_times:
            end_time = start_time + (1 / self.fps)

            # Calculate beat countdown
            remaining_time = self._calculate_beat_countdown(start_time)

            # Features for this frame
            beat_in_frame = (self.beat_times >= start_time) & (self.beat_times < end_time)
            beat_count = np.sum(beat_in_frame)

            # Aggregate onset envelope if provided
            if self.onset_env is not None:
                onset_in_frame = self._slice_onset_env(start_time, end_time)
                avg_onset_strength = np.mean(onset_in_frame) if len(onset_in_frame) > 0 else 0
            else:
                avg_onset_strength = 0

            # Append beat countdown and onset strength as features
            feature_matrix.append([
                float(remaining_time),
                int(beat_count),
                float(avg_onset_strength)
            ])

        feature_matrix = np.array(feature_matrix)

        # Use TensorFlow Normalization layer
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(feature_matrix)  # Compute the mean and variance
        self.normalized_features = normalizer(feature_matrix).numpy()

    def _calculate_beat_countdown(self, current_time):
        """
        Calculate the time until the next beat for the current frame.
        :param current_time: The current time in seconds.
        :return: Time until the next beat in seconds.

        ARGUIBLY THE MOST IMPORTANT METHOD
        Here we need to
        """
        # Find the time until the next beat
        # Here we are doing a few things ->
        """
        1. Subtract all times from our current time 
        2. Make anything negetive 0 using the Maximum method 
        3. Use the minimum method to get the smallest non-zero entry 
            that was left. 
        """
        remaining_time = np.min(np.maximum(self.beat_times - current_time, 0))  # Only consider future beats
        return remaining_time

    def _slice_onset_env(self, start_time, end_time):
        """
        Helper to slice the onset envelope for a given time window.
        Assumes onset_env is time-aligned to the audio.
        """
        if self.onset_env is None:
            return []

        # Compute sampling rate directly from audio duration
        total_duration = len(self.onset_env) / self.fps
        sr = len(self.onset_env) / total_duration  # Sampling rate of the onset envelope

        # Compute start and end indices
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)

        return self.onset_env[start_idx:end_idx]

    def get_frame_features(self, frame_index):
        """
        Retrieve normalized audio features for a specific frame.
        :param frame_index: The index of the frame.
        :return: Normalized audio features as a TensorFlow tensor.
        """
        if self.normalized_features is None:
            raise ValueError("No features processed. Run `process_audio_features` first.")
        if 0 <= frame_index < len(self.normalized_features):
            return tf.convert_to_tensor(self.normalized_features[frame_index], dtype=tf.float32)
        else:
            raise IndexError("Frame index out of range")

    def get_all_features(self):
        """
        Retrieve all normalized audio features as a TensorFlow tensor.
        """
        if self.normalized_features is None:
            raise ValueError("No features processed. Run `process_audio_features` first.")
        return tf.convert_to_tensor(self.normalized_features, dtype=tf.float32)
