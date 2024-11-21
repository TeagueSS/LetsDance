import pandas as pd
import os
import librosa
import numpy as np


class SyncedSkeletonDataAndAudio:
    def __init__(self, song_name: str):
        """
        Initialize the SyncedSkeletonDataAndAudio object.

        Args:
            song_name (str): Name of the song.
        """
        # Audio data
        self.audio_data = AudioData(song_name)

        # Skeleton data
        self.bodyParts = [
            "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye", "Right Eye Outer",
            "Left Ear", "Right Ear", "Mouth Left", "Mouth Right", "Left Shoulder", "Right Shoulder",
            "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Ring",
            "Right Ring", "Left Middle", "Right Middle", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
            "Left Heel", "Right Heel", "Left Foot Index", "Right Foot Index", "Mid Hip"
        ]
        self.columns = ["frame_number", "frame_time"] + [f"{part}_{axis}" for part in self.bodyParts for axis in ["x", "y", "z"]]
        self.skeleton_data = pd.DataFrame(columns=self.columns)

    def add_frame_data(self, frame_number: int, frame_time: float, landmarks: dict, audio_frame: dict):
        """
        Adds synchronized data for a single frame.

        Args:
            frame_number (int): The frame number.
            frame_time (float): The time of the frame in seconds.
            landmarks (dict): A dictionary with keys for each body part and values as (x, y, z) tuples.
            audio_frame: The audio data associated with this frame.
        """
        # Add audio frame data
        self.audio_data.add_frame_data(frame_number, frame_time, audio_frame)
        print(f"Type of landmarks: {type(landmarks)}, Content: {landmarks}")

        # Prepare skeleton frame data
        skeleton_row = {"frame_number": frame_number, "frame_time": frame_time}
        for part, coords in landmarks.items():
            skeleton_row[f"{part}_x"] = coords["x"]
            skeleton_row[f"{part}_y"] = coords["y"]
            skeleton_row[f"{part}_z"] = coords["z"]

        # Append to skeleton data
        self.skeleton_data = pd.concat([self.skeleton_data, pd.DataFrame([skeleton_row])], ignore_index=True)

    def get_frame(self, frame_number: int):
        """
        Retrieve synchronized data for a specific frame.

        Args:
            frame_number (int): The frame number to retrieve.

        Returns:
            dict: A dictionary containing both skeleton and audio data for the frame.
        """
        # Get audio frame
        audio_frame = self.audio_data.get_frame(frame_number)

        # Get skeleton frame
        skeleton_frame = self.skeleton_data[self.skeleton_data["frame_number"] == frame_number]
        if skeleton_frame.empty:
            return None

        return {
            "audio_data": audio_frame,
            "skeleton_data": skeleton_frame.iloc[0].to_dict()
        }

    def __repr__(self):
        return f"<SyncedSkeletonDataAndAudio: {len(self.skeleton_data)} frames, {len(self.audio_data.frames)} audio frames>"






class SkeletonData:

    def __init__(self):
        # Define column names
        self.bodyParts = [
            "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye", "Right Eye Outer",
            "Left Ear", "Right Ear", "Mouth Left", "Mouth Right", "Left Shoulder", "Right Shoulder",
            "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Ring",
            "Right Ring", "Left Middle", "Right Middle", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
            "Left Heel", "Right Heel", "Left Foot Index", "Right Foot Index", "Mid Hip"]

        # Defining our columns, and adding depth (X,Y,Z) to each of our points ->
        self.columns = ["Frame #", "Frame Time"] + [f"{part}_{axis}" for part in self.bodyParts for axis in
                                                    ["x", "y", "z"]]

        # Initialize an empty DataFrame
        self.data = pd.DataFrame(columns=self.columns)

    def get_coordinates_for_part(self, part_name):
        # Getting our body parts from our name
        return self.bodyParts[part_name]

    def add_frame_data(self, frame_number, frame_time, landmarks):
        """
        Adds data for a single frame to the DataFrame.

        Parameters:
        - frame_number (int): The frame number.
        - frame_time (str): The time of the frame in "00:00:00" format.
        - landmarks (dict): A dictionary with keys for each body part and values as (x, y, z) tuples.
        """
        # Prepare row data with frame number and time

        # Define columns for your data (for the skeletal points + frame time)
        # Here we are using pd to define it as a Pandas Data Frame ->
        frame_data = pd.DataFrame({
            "Frame #": frame_number,
            "Frame Time": frame_time,
            "Body Parts" : self.bodyParts,
        })

        # Adding our data to our body parts ->
        for part in frame_data["Body Parts"]:
            # Here we have to use a counter to move through our points
            # as they are numbered 0-34 rather than being organized
            # by their dictonary name
            counter = 0
            # Example: assuming `get_coordinates_for_part()` fetches the (x, y, z) coordinates for each body part
            frame_data[f"{part}_x"] = landmarks[counter]['x']
            frame_data[f"{part}_y"] = landmarks[counter]['y']
            frame_data[f"{part}_z"] = landmarks[counter]['z']
            # Increasing our counter
            counter += 1
        # Append row to the DataFrame
        self.data = self.data._append(frame_data, ignore_index=True)


class AudioData:

    def __init__(self, song_name: str):
        """
        Initialize the AudioData object.

        Args:
            song_name (str): Name of the song.
        """
        self.song_name = song_name
        # Create an empty DataFrame to store frames and their timings
        #self.frames = pd.DataFrame(columns=["frame_number", "frame_time", "data_frame"])
        self.frames = []  # Each frame will be a dictionary

    def add_frame_data(self, frame_number: int, frame_time: float, features: dict):
        """
        Add a frame and its timing to the AudioData object.

        Args:
            frame_number (int): The frame's number (index).
            frame_time (float): The time (in seconds) associated with the frame.
            data_frame: The data associated with the frame (e.g., frequency or amplitude data).
        """
        # Append the frame data to the DataFrame
        self.frames.append({
            "frame_number": frame_number,
            "frame_time": frame_time,
            "features": features
        })
        # Ensure the frames are sorted by frame_number
        self.frames = sorted(self.frames, key=lambda x: x["frame_number"])

    def get_frame(self, frame_number: int):
        """
        Retrieve frame data by frame number.

        Args:
            frame_number (int): The frame number to retrieve.

        Returns:
            dict: A dictionary with frame data (or None if not found).
        """
        frame = self.frames[self.frames["frame_number"] == frame_number]
        if not frame.empty:
            return frame.iloc[0].to_dict()
        return None

    def __repr__(self):
        return f"<AudioData: {self.song_name}, {len(self.frames)} frames>"


class DataSaver:
    def __init__(self, data):
        # Assuming 'data' is a pandas DataFrame passed during initialization
        self.data = data

    def save_to_csv(self, filename="output.csv"):
        """Saves the DataFrame to a CSV file."""
        # Folder path for saving the data
        folderPath = "/Users/teaguesangster/Code/Python/CS450/DataSetup/downloads"

        # Join the folder path and file name to get the full export path
        exportPath = os.path.join(folderPath, filename)

        # Sort the data by 'Frame Time' and reset the index (ignoring old index)
        self.data = self.data.sort_values(by='Frame Time').reset_index(drop=True)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        # Save the DataFrame to a CSV file
        self.data.to_csv(exportPath, index=False)

        # Print confirmation message
        print(f"Data saved to {exportPath}")

    def save_to_parquet(self, filename="output.parquet"):
        """Saves the DataFrame to a CSV file."""
        # Folder path for saving the data
        folderPath = "/Users/teaguesangster/Code/Python/CS450/DataSetup/downloads"

        # Join the folder path and file name to get the full export path
        exportPath = os.path.join(folderPath, filename)

        # Sort the data by 'Frame Time' and reset the index (ignoring old index)
        self.data = self.data.sort_values(by='Frame Time').reset_index(drop=True)

        # Check if the folder exists, if not, create it
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

        # Save the DataFrame to a Parquete file
        self.data.to_parquet(exportPath, index=False)

        # Print confirmation message
        print(f"Data saved to {exportPath}")