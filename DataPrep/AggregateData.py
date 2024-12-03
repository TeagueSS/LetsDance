import os
import numpy as np


def combine_npz_files(input_directory, output_file_path):
    """
    Combine multiple .npz files from a directory into a single .npz file.

    Parameters:
    input_directory (str): Path to the directory containing the .npz files.
    output_file_path (str): The path where the combined .npz file will be saved.
    """
    audio_data_list = []
    frame_data_list = []

    # Iterate through all .npz files in the input directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(input_directory, file_name)
            print(f"Loading data from {file_path}...")

            # Load data from the .npz file
            data = np.load(file_path)
            audio_data = data['audio']
            frame_data = data['frame']

            # Append the loaded data to lists
            audio_data_list.append(audio_data)
            frame_data_list.append(frame_data)

    # Concatenate all the loaded arrays into one long array
    combined_audio_data = np.concatenate(audio_data_list, axis=0)
    combined_frame_data = np.concatenate(frame_data_list, axis=0)

    # Save the combined data to a new .npz file
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    np.savez_compressed(output_file_path, audio=combined_audio_data, frame=combined_frame_data)
    print(f"Combined data saved successfully to {output_file_path}")


# Specify the directory where the .npz files are located and the output file path
input_directory = "/Users/teaguesangster/Desktop/ProcessedEntries"
output_file_path = "/Users/teaguesangster/Desktop/combined_data.npz"

# Combine the .npz files
combine_npz_files(input_directory, output_file_path)
