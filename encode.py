import pandas as pd

class SkeletonData:

    def __init__(self):
        # Define column names
        self.columns =  ["Frame #", "Frame Time"] + [
        "Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye", "Right Eye Outer",
        "Left Ear", "Right Ear", "Mouth Left", "Mouth Right", "Left Shoulder", "Right Shoulder",
        "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Ring", "Right Ring",
        "Left Middle", "Right Middle", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
        "Left Heel", "Right Heel", "Left Foot Index", "Right Foot Index", "Mid Hip"
        ]

        # Initialize an empty DataFrame
        self.data = pd.DataFrame(columns=self.columns)

    def add_frame_data(self, frame_number, frame_time, landmarks):
        """
        Adds data for a single frame to the DataFrame.

        Parameters:
        - frame_number (int): The frame number.
        - frame_time (str): The time of the frame in "00:00:00" format.
        - landmarks (dict): A dictionary with keys for each body part and values as (x, y, z) tuples.
        """
        # Prepare row data with frame number and time
        row_data = {"Frame #": frame_number, "Frame Time": frame_time}

        # Add landmarks to row data
        for part in self.columns[2:]:  # Skip "Frame #" and "Frame Time"
            row_data[part] = landmarks.get(part, (None, None, None))  # Default to (None, None, None) if missing

        # Append row to the DataFrame
        self.data = self.data.append(row_data, ignore_index=True)

    def save_to_csv(self, filename="output.csv"):
        """Saves the DataFrame to a CSV file."""
        # Before we save our data we sort it:
        self.data = self.data.sort_values(by='Frame Time').reset_index(drop=True)
        # Creating a CSV fo our data:
        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def save_to_parquet(self, filename="output.parquet"):
        """Saves the DataFrame to a Parquet file."""
        # Before we save our data we sort it:
        self.data = self.data.sort_values(by='Frame Time').reset_index(drop=True)
        self.data.to_parquet(filename, index=False)
        print(f"Data saved to {filename}")

