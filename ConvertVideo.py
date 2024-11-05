import logging

import numpy as np
import cv2
import os
import mediapipe as mp
import csv
import time
# Setting our logging level for documentation ->
logging.basicConfig(level=logging.INFO)

def convertVideoIntoSyncedFrames(videoPath: str, outputFolderPath: str):

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

    #Starting a timer for Video Conversion
    logging.info("Starting video frame timings")
    start_time = time.time()
    logging.info("Started Converting Video")
    # Looping until we run out of frames ->
    while True:
        # Reading in our frame
        ret, frame = cap.read()
        if not ret:
            # Breaking if there are no frames left ->
            break

        # Getting our filepath for our frame ->
        framePath = os.path.join(outputFolderPath, f'frame_{frame_count:04d}.png')
        # Save each frame in the specified output folder
        cv2.imwrite(framePath, frame)

        # Now that we have saved our frame we can update our export CSV ->
        # Calculate the timestamp for the frame
        timestamp = frame_count / fps  # Time in seconds
        frameTimings.append([framePath, timestamp])  # Append file path and timestamp

        # Increasing our frame count
        frame_count += 1

    # Saving our CSV file with all of our frames
    csv_file_path = os.path.join(outputFolderPath, 'frame_timestamps.csv')
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame File Path', 'Timestamp (s)'])  # Header
        writer.writerows(frameTimings)  # Write data rows

    # Logging completion of video conversion
    logging.info("Conversion Completed in " + str(time.time() - start_time) + " seconds")
    logging.info("Converted: " + str(len(frameTimings)) + " frames")
    logging.info("Average Frames converted per second: " + str(frame_count / len(frameTimings)) + " frames")
    logging.info("Exported to: " + outputFolderPath)
    logging.info("Frame Sync expored to:" + csv_file_path)

    cap.release()
    print(f"Frames saved to {outputFolderPath}")



###TODO
def convertFrameIntoPose(inputPath: str, outputFolderPath: str, subFolderPath: str):
    #Defining our resources ->
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Create the full output directory path
    output_folder = os.path.join(outputFolderPath, subFolderPath)
    # Creating that sub folder ->
    os.makedirs(output_folder, exist_ok=True)

    # Taking our
    """/
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for frame_file in os.listdir(input_folder):
            frame_path = os.path.join(input_folder, frame_file)
            image = cv2.imread(frame_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Draw the pose annotation on the image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the annotated image
            cv2.imwrite(os.path.join(output_folder, frame_file), image)
    """


#Creating our testing paths
convertedVideoPath = "/Users/teaguesangster/Code/Python/CS450/DataSetup/VideoFrames"
videoToConvert = "/Users/teaguesangster/Code/Python/CS450/DataSetup/downloads/Just Dance Hits： Only Girl (In The World) by Rihanna [12.9k]_video.mp4"
convertVideoIntoSyncedFrames(videoToConvert , convertedVideoPath)
# Converting our test video to a

