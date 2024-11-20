import pandas as pd
import os


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