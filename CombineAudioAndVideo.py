import argparse
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import librosa
import pandas as pd
import numpy as np

from AudioHandler import AudioHandler
from ConvertVideo import convertFrameIntoPose
from Training.AudioSlicing import AudioFrameProcessor
from Training.TensorFlowProcessing import TensorFlowDataPrep
from encode import SyncedSkeletonDataAndAudio
# Initialize a lock for writing to the HDF5 file
lock = threading.Lock()

class CombineAudioAndVideo:
    def __init__(self, songNameLink: str):
        # Here we need to store all of our data so
        self.youtubeLink = songNameLink

        # Our methods can interact

    def main(songYoutubeLink: str, savePath: str, timeCropFront: int, timeCropBack: int):
        parser = argparse.ArgumentParser(description='Combine audio files and video files')
        # Importing our
        # This method will call of the methods we all made

        #map_audio_to_video_frames

    #entry, file_handler,audio_slicer
    def process_audio_and_video_entry(self, entry,  file_handler : TensorFlowDataPrep,
                                      audio_slicer: AudioFrameProcessor):
        """
        Process a single mapping entry.
        Args:
            entry (dict): A single mapping entry containing frame data.
            song_path (str): The path to the audio file.
            fps (int): Frames per second of the video.

        Returns:
            dict or None: Processed data if successful, otherwise None.
        """
        try:
            # Extract information from the entry
            frame_path = entry["Frame File Path"]
            frame_number = entry["Frame #"]

            # Seeing if we can process our Skeletal Frames,
            # This will determine if our information can be saved ->

            logging.info("Processing landmarks for frame" + str(frame_number))
            # Process landmarks
            landmarks = convertFrameIntoPose(frame_path, True)
            # See if we actually got landmarks
            if landmarks is None:
                # If we have no landmarks we failed and need to skip
                return

            logging.info("Landmarks for frame" + str(frame_number) + "Found")
            # If we have our landmarks we need to save them in tensor format:
            with lock:
                # Here procesing our skeleton frames
                processed_skeleton = file_handler.process_skeltal_features(landmarks)

            logging.info("Starting our Audio Conversion -> ")
            print("Processing Audio")

            with lock:
                # Process audio with our audio processor instance
                audio_frame = audio_slicer.get_frame_features(frame_number)

            with lock:
                # Saving our processed information ->
                #audio, skelton,frame_number
                file_handler.add_frame(audio = audio_frame, skelton= processed_skeleton , frame_number= frame_number)

        # Exception logging to catch any errors ->
        except Exception as e:
            # Print the exception type, message, and the full traceback
            print(f"Error processing entry {entry}:")
            print(f"Exception Type: {type(e).__name__}")
            print(f"Exception Message: {str(e)}")
            print("Full Traceback:")
            traceback.print_exc()
            traceback.print_exc()  # Print the traceback
            return None
            print(f"Error processing entry {entry}:" +"Due to" +str(e))
            return None

    def process_audio_and_video_frames_Multi_Threaded(self, mapping, song_name, song_path, fps):
        """
        Process audio and video frames concurrently and save them into a SyncedSkeletonDataAndAudio object.

        Args:
            mapping (list): A list of dictionaries with frame mappings.
            song_name (str): The name of the song.
            song_path (str): The path to the audio file.
            fps (int): Frames per second of the video.

        Returns:
            SyncedSkeletonDataAndAudio: An object containing synchronized skeleton and audio data.
        """
        # Here we are actually going

        #TODO get the audio and data handler to hold everything ->


        # Object to save our audio and skeletal and Audio data:
        file_handler = TensorFlowDataPrep()
        # Audio Handler to prep our audio stream
        audio_handler = AudioHandler(song_path)
        # Passing our audio slicer our audio handler
        audio_slicer = AudioFrameProcessor(fps, beat_times=audio_handler.beat_times, onset_env=audio_handler.onset_env)
        # Slicing all of our Audio Frames ->
        audio_slicer.process_audio_features(audio_handler.duration)
        # Print out our audio length
        print("************************************************")
        print("************************************************")
        print("************************************************")

        print(len(audio_slicer.normalized_features))
        print("************************************************")
        print("************************************************")
        print("************************************************")

        # Now that all of our audio is sliced we can start Handling our frames

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor() as executor:
            # Process all entries concurrently

            futures = []
            while mapping:
                # Removing our top entry
                entry = mapping.pop(0)

                # Here we are passing it all of the objects we created to process
                # And save all of our frames :)
                future = executor.submit(self.process_audio_and_video_entry,entry = entry, file_handler = file_handler, audio_slicer=audio_slicer)
                #Adding this entry to the future so that it is completed:
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Here we pass our file handler so we can save or do whatever other processing
        # we still need to do
        return file_handler

    def map_csv_of_frames(self, csv_path):
        # Load the video frame data
        frame_data = pd.read_csv(csv_path)

        # Ensure correct column names (update based on your CSV)
        if 'Frame File Path' not in frame_data.columns or 'Timestamp (s)' not in frame_data.columns:
            raise ValueError("Expected columns 'Frame File Path' and 'Timestamp (s)' not found in the CSV")

        # Create mapping directly
        mapping = frame_data[['Timestamp (s)', 'Frame File Path']].rename(
            columns={'Timestamp (s)': 'Frame #', 'Frame File Path': 'Frame File Path'}
        )

        # Convert to list of dictionaries if needed
        return mapping.to_dict(orient='records')

    def map_audio_to_video_frames(self, csv_path, audio_path, fps):
        """
                    Maps audio frames to video frames, skipping incomplete frames and maintaining a buffer.

                    Args:
                        csv_path (str): Path to the CSV file containing video frame data.
                        audio_path (str): Path to the audio file.
                        fps (int): Frames per second of the video.

                    Returns:
                        list: A list of dictionaries, each containing:
                              - Frame #
                              - Frame File Path
                              - Audio Start Index
                              - Audio End Index
                    """
        # Load the video frame data
        frame_data = pd.read_csv(csv_path)
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=None)
        # Duration of a single video frame in seconds
        frame_duration = 1 / fps
        logging.info("Frame Duration is: " + str(frame_duration))
        # Samples required per frame
        # (How many frames we loose due to the sample rate)
        samples_per_frame = int(frame_duration * sr)
        logging.info("Samples per frame is: " + str(samples_per_frame))

        # Initialize variables
        audio_index = 0  # Current audio sample index
        buffer = 0  # Tracks accumulated leftover samples
        mapping = []  # To store the result

        for frame_number, row in enumerate(frame_data.itertuples(), start=1):
            frame_path = row._1  # Assuming the first column is Frame File Path
            frame_time = row._2  # Assuming the second column is Timestamp (s)

            # Calculate the number of samples available including buffer
            available_samples = len(audio) - audio_index + buffer

            # Seeing if we have enough of a buffer and we can save our frame ->
            if available_samples < samples_per_frame:
                # if we don't have enough time
                # Skip this frame and accumulate audio in the buffer
                buffer += available_samples
                audio_index = len(audio)  # Mark as end of audio
                #TODO We need to delete the extra frame
                continue

            # Determine audio start and end indices
            audio_start_index = audio_index
            audio_end_index = audio_index + samples_per_frame

            # Add frame information to the mapping
            mapping.append({
                "Frame #": frame_number,
                "Frame File Path": frame_path,
                "Audio Start Index": audio_start_index,
                "Audio End Index": audio_end_index
            })

            # Update indices
            audio_index = audio_end_index
            buffer = 0  # Reset buffer after using it

        print("Succesfully mapped " + str(len(mapping)) + " audio frames to video frames")
        logging.info("Succesfully mapped " + str(len(mapping)) + " audio frames to video frames")
        return mapping
