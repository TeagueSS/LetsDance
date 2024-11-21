import os
import logging

from keras.src.utils.module_utils import scipy

from ConvertAudio import AudioHandler
from ConvertVideo import *
import pytest
from visualisation import *
from encode import SkeletonData, DataSaver , AudioData
from CombineAudioAndVideo import *
import matplotlib.pyplot as plt

# Defining our test file paths ->
CONVERTED_VIDEO_PATH = "Outputs"
VIDEO_NAME ="Only_Girl_Riannah"
VIDEO_TO_CONVERT = "Test Video and Audio/TestVideo_Only_Girl_Rihanna.mp4"
FRAME_PATH = "Test Video and Audio/frame_0200.png"
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
    landmarks = convertFrameIntoPose(FRAME_PATH , False)
    if landmarks:
        logging.info(f"Extracted {len(landmarks)} landmarks.")
    else:
        logging.error("No landmarks were detected.")

def test_convert_frames_into_hdf5():
    logging.info(f"Testing conversion of frames in '{OUTPUT_FRAMES_PATH}' into HDF5 file...")
    convertFramesIntoHDF5(OUTPUT_FRAMES_PATH, H5_FILEPATH)
    logging.info("Frames conversion to HDF5 completed.")



def test_view_skeletal_data():
    logging.info(f"Testing saving skeletal data...")
    logging.info(f"Getting skeletal data from: '{FRAME_PATH}' ")
    landmarks = convertFrameIntoPose(FRAME_PATH , False)
    # Visualizing our data
    logging.info("Opening our landmarks in browser -> (please allow 5 seconds) ")
    pointsTo3DSkeleton(landmarks)

def test_save_skeletal_data():
    logging.info(f"Testing saving skeletal data...")
    logging.info(f"Getting skeletal data from: '{FRAME_PATH}' ")
    landmarks = convertFrameIntoPose(FRAME_PATH , False)
    logging.info("Creating an instance of SkeletonData... ")
    skeltonData = SkeletonData()
    logging.info("Attempting to save our Skeletal Data:")
    skeltonData.add_frame_data(0,0,landmarks)
    # Creating a data saving object ->
    data_saver = DataSaver(skeltonData.data)
    # Saving our data
    ###TODO we need to fix the output to take in the folder path as well
    #   And then join it later on
    data_saver.save_to_csv("TestingOutputs/Only_Girl_Rihanna_landmarks.csv")


def test_split_audio_frames():
    # Printing out what we want to test
    logging.info(f"Testing splitting audio frames into {OUTPUT_FRAMES_PATH}...")
    # Creating an instance of our Audio conversion class:
    audio_Handler = AudioHandler("/Users/teaguesangster/Code/Python/CS450/DataSetup/Testing/Test Video and Audio/Test Audio Only Girl Riannah.mp3")
    # Getting our audio frame
    logging.info("Attempting to format our song ")
    audio_frame = audio_Handler.convertAudioFrame(1,10)
    # Displaying our Audio frame ->
    logging.info("Displaying our song")
    showAudioFrames(audio_frame , audio_Handler.sampleRate)

def test_audio_data_storage():
    # Log the start of the test
    logging.info("Testing storing audio frames into AudioData...")

    # Initialize an instance of the AudioHandler class
    audio_Handler = AudioHandler("/Users/teaguesangster/Code/Python/CS450/DataSetup/Testing/Test Video and Audio/Test Audio Only Girl Riannah.mp3")

    # Create an AudioData object to store frames
    audio_data = AudioData("Test Audio Only Girl Riannah")

    # Generate and store 3 audio frames
    logging.info("Attempting to process and store frames...")
    for frame_number in range(1, 3):
        frame_time = frame_number  # Assuming each frame is 1 second apart for simplicity
        audio_frame = audio_Handler.convertAudioFrame(frame_time, 10)
        audio_data.add_frame_data(frame_number, frame_time, audio_frame)

    # Display the saved frames for verification
    logging.info("Displaying stored frames:")
    print(audio_data.frames)

    # Test displaying the first frame
    logging.info("Displaying the first frame's spectrogram:")
    first_frame = audio_data.get_frame(1)
    if first_frame:
        showAudioFrames(first_frame["data_frame"], audio_Handler.sampleRate)
    else:
        logging.error("Failed to retrieve the first frame!")



def test_video_and_audio_linking():
    logging.info("Testing video linking...")

    # Try to open our video
    cap = cv2.VideoCapture(VIDEO_TO_CONVERT)
    # Check if the video was opened successfully
    if not cap.isOpened():
        logging.error("Unable to open video " + VIDEO_TO_CONVERT)
        print("Error: Could not open video.")
        return

    # Getting our FPS ->
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Releasing our cap
    cap.release()
    # Getting Our Video Path ->
    logging.info(f"Testing conversion of video '{VIDEO_NAME}' into frames...")
    csv_Path = convertVideoIntoSyncedFrames(VIDEO_TO_CONVERT, OUTPUT_PATH, VIDEO_NAME)
    logging.info("Video conversion completed.")
    # Now that we have our Video information lets get our audio information
    ActionHandler = CombineAudioAndVideo("No Youtube link needed")
    sycned_frames = ActionHandler.map_audio_to_video_frames(csv_Path, AUDIO_PATH, fps)
    # Printing the first few synced frames ->
    # Assuming `mapping` is a list of dictionaries, as returned by the function
    for i, entry in enumerate(sycned_frames[:5], start=1):
        print(f"Entry {i}:")
        print(f"  Frame #: {entry['Frame #']}")
        print(f"  Frame File Path: {entry['Frame File Path']}")
        print(f"  Audio Start Index: {entry['Audio Start Index']}")
        print(f"  Audio End Index: {entry['Audio End Index']}")
        print()  # Add a blank line for readability


def test_Muti_Threaded_Sync_Conversion():
    logging.info("Testing video linking...")

    # Try to open our video
    cap = cv2.VideoCapture(VIDEO_TO_CONVERT)
    # Check if the video was opened successfully
    if not cap.isOpened():
        logging.error("Unable to open video " + VIDEO_TO_CONVERT)
        print("Error: Could not open video.")
        return

    # Getting our FPS ->
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Releasing our cap
    cap.release()
    # Getting Our Video Path ->
    logging.info(f"Testing conversion of video '{VIDEO_NAME}' into frames...")
    csv_Path = convertVideoIntoSyncedFrames(VIDEO_TO_CONVERT, OUTPUT_PATH, VIDEO_NAME)
    logging.info("Video conversion completed.")
    # Now that we have our Video information lets get our audio information
    ActionHandler = CombineAudioAndVideo("No Youtube link needed")
    sycned_frames = ActionHandler.map_audio_to_video_frames(csv_Path, AUDIO_PATH, fps)

    # Now that we have our frames trying to convert them
    logging.info("Video Frames found, lets convert!")
    print("Initating Frame Processing ")

    frames = ActionHandler.process_audio_and_video_frames_Multi_Threaded(sycned_frames, "Only Girl", AUDIO_PATH, fps)
    # Printing our frames
    logging.info("Frames Processing completed.")
    logging.info("Printing results")
    print(frames.get_frame(0))


def test_spectrogram_and_visualization():
    logging.info("Testing spectrogram and visualization...")
    # Getting our song
    #audio_Handler = AudioHandler("/Users/teaguesangster/Code/Python/CS450/DataSetup/Testing/Test Video and Audio/Test Audio Only Girl Riannah.mp3")
    # Building our Tempogram
    # #audio_Handler.build_audio_tempogram_ratio()
    # # Plotting our entire temporalgram ->
    y, sr = librosa.load("/Users/teaguesangster/Code/Python/CS450/DataSetup/Testing/Test Video and Audio/Test Audio Only Girl Riannah.mp3")
    chroma_orig = librosa.feature.chroma_cqt(y=y, sr=sr)

    chromagram_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chromagram_stft, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chroma-STFT')
    chromagram_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    librosa.display.specshow(chromagram_cqt, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chroma-CQT')
    chromagram_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    librosa.display.specshow(chromagram_cens, y_axis='chroma', x_axis='time', cmap='coolwarm')
    plt.title('Chroma-CENS')

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)

    librosa.display.specshow(chromagram_stft, y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='Chroma-STFT')

    #librosa.display.specshow(chromagram_cqt, y_axis='chroma', x_axis='time', ax=ax[1])
    #ax[1].set(title='Chroma-CQT')

    #librosa.display.specshow(chromagram_cens, y_axis='chroma', x_axis='time', ax=ax[2])
    #ax[2].set(title='Chroma-CENS')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Run tests
    test_convert_video_into_frames()
    test_convert_frame_into_pose()
    test_convert_frames_into_hdf5()
    test_save_skeletal_data()
    test_view_skeletal_data()
    test_packages()
    test_split_audio_frames()
    logging.info("Tests Complete.")

