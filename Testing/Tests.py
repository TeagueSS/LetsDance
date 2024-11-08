import os
import logging
from ConvertVideo import *
import pytest


# Defining our test file paths ->
CONVERTED_VIDEO_PATH = "Outputs"
VIDEO_NAME ="Only_Girl_Riannah"
VIDEO_TO_CONVERT = "Test Video and Audio/TestVideo_Only_Girl_Rihanna.mp4"
FRAME_PATH = "Test Video and Audio/frame_0289.png"
H5_FILEPATH = "Outputs/Only_Girl_Rihanna_landmarks.h5"
AUDIO_PATH = "Test Video and Audio/Test Audio Only Girl Riannah.mp3"
OUTPUT_PATH = "Outputs"
OUTPUT_FRAMES_PATH = "Outputs/Only_Girl_Riannah"
# Assuming these functions are defined elsewhere in your project
# from your_module import convertVideoIntoSyncedFrames, convertFrameIntoPose, convertFramesIntoHDF5
current_directory = os.getcwd()
logging.info("Getting our current test Directory: " + current_directory)

# Updating all of our paths
VIDEO_TO_CONVERT = os.path.join(current_directory, VIDEO_TO_CONVERT)
FRAME_PATH = os.path.join(current_directory, FRAME_PATH)
H5_FILEPATH = os.path.join(current_directory, H5_FILEPATH)
AUDIO_PATH = os.path.join(current_directory, AUDIO_PATH)
OUTPUT_PATH = os.path.join(current_directory, OUTPUT_PATH)
OUTPUT_FRAMES_PATH = os.path.join(current_directory, OUTPUT_FRAMES_PATH)



"""/
def test_find_file_in_local_directory():
    '''Test to check if the video file exists and log its directory.'''
    # Use the relative path
    # Check if the file exists
    if os.path.exists(full_path):
        directory_path = os.path.dirname(full_path)
        logging.info(f"Full Directory Path: {directory_path}")
        assert True  # Test passes if file exists
    else:
        logging.error(f"File does not exist: {full_path}")
        assert False  # Test fails if file does not exist
"""


def test_packages():
    logging.info("Testing Installed Packages")

    packages = [
        "yt_dlp",
        "subprocess",  # subprocess is a built-in module, so it should always be available
        "csv",  # csv is also built-in
        "librosa",
        "numpy",
        "plotly",
        "pandas"
    ]

    for package in packages:
        try:
            # Use __import__ to import the package dynamically
            __import__(package)
            logging.info(f"{package} is installed.")
        except ImportError:
            logging.error(f"{package} is NOT installed.")
            assert False, f"Package {package} is not installed."


def test_convert_video_into_frames():
    logging.info(f"Testing conversion of video '{VIDEO_NAME}' into frames...")
    convertVideoIntoSyncedFrames(VIDEO_TO_CONVERT, OUTPUT_PATH, VIDEO_NAME)
    logging.info("Video conversion completed.")

def test_convert_frame_into_pose():
    logging.info(f"Testing pose extraction from frame: {FRAME_PATH}")
    landmarks = convertFrameIntoPose(FRAME_PATH)
    if landmarks:
        logging.info(f"Extracted {len(landmarks)} landmarks.")
    else:
        logging.error("No landmarks were detected.")

def test_convert_frames_into_hdf5():
    logging.info(f"Testing conversion of frames in '{OUTPUT_FRAMES_PATH}' into HDF5 file...")
    convertFramesIntoHDF5(OUTPUT_FRAMES_PATH, H5_FILEPATH)
    logging.info("Frames conversion to HDF5 completed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Run tests
    test_convert_video_into_frames()
    test_convert_frame_into_pose()
    test_convert_frames_into_hdf5()
    test_packages()
    logging.info("Tests Complete.")

