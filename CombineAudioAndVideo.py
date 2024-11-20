import argparse
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import librosa
import pandas as pd
import numpy as np

from ConvertAudio import AudioHandler
from ConvertVideo import convertFrameIntoPose
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

    def process_audio_and_video_entry(self, entry, song_path: str, fps: int, frame_holder : SyncedSkeletonDataAndAudio):
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
            logging.info(f'Processing {song_path}')
            # Extract information from the entry
            frame_path = entry["Frame File Path"]
            frame_number = entry["Frame #"]
            audio_start_index = entry["Audio Start Index"]
            audio_end_index = entry["Audio End Index"]

            logging.info("Processing landmarks for frame" + str(frame_number))
            # Process landmarks
            landmarks = convertFrameIntoPose(frame_path, True)
            if landmarks is None:
                return  # Skip frames with no landmarks

            logging.info("Starting our Audio Conversion -> ")
            print("Processing Audio")
            # Process audio


            audio_handler = AudioHandler(song_path)
            # Getting Librosa
            audio_frame = audio_handler.create_audio_map(audio_start_index, audio_end_index)

            # Return processed data
            with lock:
                frame_holder.add_frame_data(frame_number, audio_start_index, audio_frame, landmarks)


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
        # Object to save all frames
        frame_holder = SyncedSkeletonDataAndAudio(song_name)

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor() as executor:
            # Process all entries concurrently

            futures = []
            while mapping:
                # Removing our top entry
                entry = mapping.pop(0)

                #Queing our task
                future = executor.submit(self.process_audio_and_video_entry, entry, song_path, fps, frame_holder)
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()

        return frame_holder


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
