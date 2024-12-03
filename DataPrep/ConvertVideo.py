import logging
import threading
import numpy as np
import cv2
import os
import mediapipe as mp
import csv
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
# Setting our logging level for documentation ->
logging.basicConfig(level=logging.INFO)
import h5py
import psutil

# Initialize a lock for writing to the HDF5 file
lock = threading.Lock()

'''/
Adding a method for saving to the local disk -> 
'''
def save_frame(frame_data):
    # Creating out frame path ->
    frame_path, frame = frame_data
    # Writing to our local disk ->
    cv2.imwrite(frame_path, frame)


def convertVideoIntoSyncedFrames(videoPath: str, outputFolderPath: str , videoName: str):

    ###TODO
    ### Create the full output directory path
    output_folder = os.path.join(outputFolderPath, videoName)
    # Creating that sub folder ->
    os.makedirs(output_folder, exist_ok=True)

    logging.info("Attempting to import" + videoPath)
    # Here we need to turn a video in to a folder of syncedFrames
    cap = cv2.VideoCapture(videoPath)
    # Check if the video was opened successfully
    if not cap.isOpened():
        logging.error("Unable to open video " + videoPath)
        print("Error: Could not open video.")
        return

    # Checking our FPS (So we can sync our song beats to each frame)
    logging.info("Successfully opened video " + videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info("Video FPS is:" + str(fps))
    # Creating a dictonary to hold our frame path and time
    frameTimings = []
    # Creating a counter for our frame #
    frame_count = 0

    #Starting a timer for Video DataPrep
    logging.info("Starting video frame timings")
    start_time = time.time()
    logging.info("Started Converting Video")

    # Using ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor() as executor:  # Adjust the number of workers as needed

        # Looping until we run out of frames ->
        while True:
            # Reading in our frame
            ret, frame = cap.read()
            if not ret:
                # Breaking if there are no frames left ->
                break

            # Crop the frame: remove 1/3 from the left and right
            height, width = frame.shape[:2]
            crop_width = width // 3  # Calculate 1/3 of the width
            cropped_frame = frame[:, crop_width:width - crop_width]  # Crop left and right

            # Getting our filepath for our frame ->
            framePath = os.path.join(output_folder, f'frame_{frame_count:04d}.png')

            # Submit the frame save task to the executor
            executor.submit(save_frame, (framePath, cropped_frame))

            # Now that we have saved our frame we can update our export CSV ->
            # Calculate the timestamp for the frame
            timestamp = frame_count # Here we just want the frame number ->
            frameTimings.append([framePath, timestamp])  # Append file path and timestamp

            # Increasing our frame count
            frame_count += 1
            del frame, cropped_frame

    # Saving our CSV file with all of our frames
    csv_file_path = os.path.join(outputFolderPath,videoName, '.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame File Path', 'Timestamp (s)'])  # Header
        writer.writerows(frameTimings)  # Write data rows

    # Logging completion of video conversion
    logging.info("DataPrep Completed in " + str(time.time() - start_time) + " seconds")
    logging.info("Converted: " + str(len(frameTimings)) + " frames")
    logging.info("Average Frames converted per second: " + str(frame_count / (time.time() - start_time)) + " frames")
    logging.info("Exported to: " + outputFolderPath)
    logging.info("Frame Sync expored to:" + csv_file_path)

    cap.release()
    print(f"Frames saved to {outputFolderPath}")
    return csv_file_path

def convertFrameIntoPose(imagePath: str, delete: bool):
    # Load the image
    logging.info("Attempting to import" + imagePath)
    image = cv2.imread(imagePath)
    # Seeing if we loaded our image
    if image is None:
        logging.error("Unable to open image " + imagePath)
        print(f"Error: Could not load image at {imagePath}")
        return None
    # Importing our tools:
    logging.info("Successfully opened image " + imagePath)
    #Defining our resources ->
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Initialize MediaPipe Pose with static image mode (since we're using individual frames)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Process the image to extract pose landmarks
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Check if landmarks were detected
        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return None

        # Extract landmarks and their visibility
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            # Each landmark has x, y, z, and visibility attributes
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })

        # Delete the output photo ->

        # For now I wanna see the overlay it creates ->
        """/
        # Draw landmarks on the image for visualization (optional)
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        )

        # Save the annotated image (optional)
        output_path = imagePath.replace('.png', '_pose.png')
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved at: {output_path}")
        """
        if delete:
            # Delete the original image file
            try:
                os.remove(imagePath)
                logging.info(f"Deleted image file: {imagePath}")
            except OSError as e:
                logging.error(f"Error deleting file {imagePath}: {e}")

        # Save these values ->
        print(f"Pose landmarks detected: {len(landmarks)}")
        #print(landmarks)
        return landmarks


# Multithreaded function to process a single frame and save the pose to HDF5
def process_and_save_frame(frame_path, frame_number, h5file):
    try:
        # Retrieve landmarks from the frame
        landmarks = convertFrameIntoPose(frame_path , True)
        # Check if landmarks is None or empty
        if landmarks is None or len(landmarks) == 0:
            logging.error(f"No landmarks found for frame {frame_number} at {frame_path}.")
            return
        # Prepare data in a NumPy array format
        landmark_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        # Write data to the HDF5 file using a lock to ensure thread safety
        with lock:
            # Create or update a dataset for this frame's landmarks
            h5file.create_dataset(f"frame_{frame_number}", data=landmark_array)
        logging.info(f"Successfully processed and saved landmarks for frame {frame_number}.")
    except Exception as e:
        logging.error(f"Error processing frame {frame_number} ({frame_path}): {e}")


# Method to convert folder of frames and turn them into a H5 File ->
def convertFramesIntoHDF5(framesDirectory, h5_filepath):
    # Get a list of all frame files in the specified directory
    logging.info(f"Attempting to import frames from {framesDirectory}")

    # Ensure the directory exists
    if not os.path.isdir(framesDirectory):
        logging.error(f"Directory not found: {framesDirectory}")
        return

    # Collect all frame file paths
    frames = []
    for filename in os.listdir(framesDirectory):
        # Add your specific frame file type checks here
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust file types as needed
            frame_path = os.path.join(framesDirectory, filename)
            frames.append(frame_path)

    logging.info(f"Found {len(frames)} frame files to process.")

    # Open HDF5 file in append mode to allow multiple dataset additions
    with h5py.File(h5_filepath, 'a') as h5file:
        logging.info("Successfully opened h5 file")

        # Thread pool to handle frame processing
        with ThreadPoolExecutor() as executor:
            logging.info("Starting to process frames")
            # Process each frame asynchronously

            futures = []
            for frame_number, frame_path in enumerate(frames):
                # Schedule frame processing and saving
                ###TODO Here we need to change it so that it appaends to an array,
                    # and then after it's all been added, we sort, and save ->
                dataPoints = []
                future = executor.submit(process_and_save_frame, frame_path, frame_number, h5file)
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()



