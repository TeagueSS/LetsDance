import numpy as np
from tensorboard.compat import tf




#TODO
"""
1. Get the multithreaded saving working 
3. get the saving working 
4. start dumping 
"""
class TensorFlowDataPrep:
    def __init__(self):
        # Here we have an array of our aduio and skeletal frames ->
        self.audio_data = []
        self.frame_data = []
        self.frame_number =[]

    def get_length(self):
        """
        Returns the smallest length among audio_data, frame_data, and frame_number.
        """
        return min(len(self.audio_data), len(self.frame_data), len(self.frame_number))

    def add_frame(self, audio, skelton,frame_number ):
        # Here we add our frames by just appending them back ->
        self.audio_data.append(audio)
        self.frame_data.append(skelton)
        # our frame number is so we can know what is where and then sort it after ->
        self.frame_number.append(frame_number)

    #TODO -> add sorting
    def save(self, file_path):
        """
        Save the stored audio and frame data to a compressed .npz file.

        Parameters:
        file_path (str): The path where the data will be saved.
        """
        # Convert lists to NumPy arrays
        audio_array = np.array(self.audio_data)
        frame_array = np.array(self.frame_data)

        # Save arrays to a compressed .npz file
        np.savez_compressed(file_path, audio=audio_array, frame=frame_array)

    def load(self, file_path):
        """
        Load audio and frame data from a compressed .npz file.

        Parameters:
        file_path (str): The path from where the data will be loaded.
        """
        # Load arrays from the .npz file
        data = np.load(file_path)

        # Convert loaded arrays back to lists
        self.audio_data = data['audio'].tolist()
        self.frame_data = data['frame'].tolist()




    def process_skeltal_features(self, skeleton_features):

        landmarks = {
            "x_coords": [],
            "y_coords": [],
            "z_coords": []
        }


        # Function to append coordinates for a specific index
        def append_coordinates(index):
            landmarks["x_coords"].append(skeleton_features[index]['x'])
            landmarks["y_coords"].append(skeleton_features[index]['y'])
            landmarks["z_coords"].append(skeleton_features[index]['z'])

        #Getting our x,y,x for all of our different points ->
        try:
            # Neck (0) → Right Shoulder (12) → Right Elbow (14) → Right Wrist (16)
            append_coordinates(0)  # Neck
            append_coordinates(12)  # Right Shoulder
            append_coordinates(14)  # Right Elbow
            append_coordinates(16)  # Right Wrist

            # Left Hip (23) → Left Knee (25) → Left Ankle (27)
            append_coordinates(23)  # Left Hip
            append_coordinates(25)  # Left Knee
            append_coordinates(27)  # Left Ankle

            # Right Hip (24) → Right Knee (26) → Right Ankle (28)
            append_coordinates(24)  # Right Hip
            append_coordinates(26)  # Right Knee
            append_coordinates(28)  # Right Ankle


        except KeyError:
            print("Error: Expected dictionary with 'x', 'y', 'z' keys.")


        print(landmarks)
        #Now that we have all of our features we just need to keep our XYZ
        # Filter out only required points
        #filtered_data = {key: skeleton_features[key] for key in required_parts}

        # Combine x, y, z coordinates into a single array
        processed_data = np.array([
            landmarks["x_coords"],
            landmarks["y_coords"],
            landmarks["z_coords"]
        ]).T  # Transpose to make it (num_points, 3)

        # Normalize the data to the range [0, 1] using TensorFlow
        def normalize_to_unit_range_tf(data):
            data_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
            min_vals = tf.reduce_min(data_tensor, axis=0)
            max_vals = tf.reduce_max(data_tensor, axis=0)
            range_vals = tf.maximum(max_vals - min_vals, 1e-8)  # Avoid division by zero
            normalized_tensor = (data_tensor - min_vals) / range_vals
            return normalized_tensor


        # Apply normalization
        processed_data = normalize_to_unit_range_tf(processed_data)

        # Flatten for TensorFlow input
        flattened_data = tf.reshape(processed_data, [-1])  # Flatten the normalized data

        # Prepare TensorFlow tensor
        tensor_input = tf.convert_to_tensor(flattened_data, dtype=tf.float32)
        return tensor_input




    def prepare_for_tensorflow(self):
        # Format and batch the data for TensorFlow
        # (e.g., convert to numpy arrays, apply padding)
        pass
