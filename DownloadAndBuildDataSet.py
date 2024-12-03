# Process function for all links
import csv
import logging
import os
import traceback

import cv2

from DataPrep.CombineAudioAndVideo import CombineAudioAndVideo
from DataPrep.ConvertVideo import convertVideoIntoSyncedFrames
from DataPrep.dowload import download_audio, download_and_transcode_video

csv_path = "/Users/teaguesangster/Code/Python/CS450/DataSetup/JustDanceRemaining2.csv"
temporary_folder_path = "/Volumes/Samsung/Temporary"
final_path = "/Volumes/Samsung/Completed_Processing/"
output_csv_path = "/Volumes/Samsung/Completed_Processing/DanceSongsCompleted.csv"
def process_links(csv_file , temporary_path , final_path , output_csv_path ):
    # Read the CSV with YouTube links
    if not os.path.isfile(csv_file):
        print(f"Error: The temporary path '{csv_file}' does not exist or is not a directory.")
        return
    # Check if the temporary path exists and is a directory
    if not os.path.isdir(temporary_path):
        print(f"Error: The temporary path '{temporary_path}' does not exist or is not a directory.")
        return

    # Check if the final path exists and is a directory
    if not os.path.isdir(final_path):
        print(f"Error: The final path '{final_path}' does not exist or is not a directory.")
        return


    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present

        # Prepare list for saving to output CSV
        processed_paths = []
        output_directory = final_path
        counter =450
        for row in csv_reader:
            counter += 1  # Moved here
            title, url = row
            print(f"Processing: {title} | {url}")

            # Download and get paths
            try:
                # Make our audio Download path:
                output_filename = str(counter) + '_transcoded_video.mp3'
                output_path = os.path.join(temporary_path, output_filename)
                # Download our Audio
                audio_path = download_audio(url, output_path)
                # Getting our video ->
                print(f"Audio: {audio_path} Downloaded Correctly")

                # Making our path for our Video file:
                output_filename = str(counter) + '_transcoded_video.mp4'
                output_path = os.path.join(temporary_path, output_filename)
                # Now that we have defined our directory downloading our video ->
                video_download = download_and_transcode_video(url=url, output_path=output_path)

                # Getting our audio and Video for this one
                # Try to open our video
                cap =cv2.VideoCapture(video_download)
                # Check if the video was opened successfully
                if not cap.isOpened():
                    logging.error("Unable to open video " + video_download)
                    print("Error: Could not open video.")
                    continue
                # Getting our FPS ->
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                # checking if our FPS is 0
                if fps == 0:
                    print("Error: FPS is zero.")
                    continue  # Skip to the next video

                # Releasing our cap
                cap.release()
                # Using our FPS to divide our frames ->
                # Getting Our Video Path ->
                logging.info(f"Testing conversion of video '{video_download}' into frames...")
                video_name = "video_" + str(counter)
                csv_of_frames_path = convertVideoIntoSyncedFrames(video_download, output_directory, video_name)
                logging.info("Video conversion completed.")
                # Now that we have our Video information lets get our audio information
                ActionHandler = CombineAudioAndVideo(video_name)
                # map_csv_of_frames to use for our syncing
                print("Video conversion completed.")
                print("Processing frames")
                sycned_frames = ActionHandler.map_csv_of_frames(csv_of_frames_path)
                song_name = str(counter) + "Song"
                frames = ActionHandler.process_audio_and_video_frames_Multi_Threaded(sycned_frames, song_name,
                                                                                audio_path, fps)
                #sorting our frames
                print("Sorting our entries: ")
                frames.sort_frame_entries()
                # Saving our Data:
                print("Saving frames")
                path_to_save = os.path.join("/Volumes/Samsung/Saved", song_name)

                frames.save(path_to_save)
                print("Exported as:")
                print(path_to_save)
                # Now that we have our frames trying to convert them
                logging.info("Successfully converted video " + video_download + " to frames.")
                # Append path for output CSV
                processed_paths.append({"File: Path": path_to_save})

                os.remove(audio_path)
                os.remove(video_download)
                logging.info(f"Deleted temporary files: {audio_path} and {video_download}")


            except Exception as e:
                print(f"Error processing {title}: {e}")
                traceback.print_exc()
                continue  # Proceed to the next video

        # Save paths to output CSV
        with open(output_csv_path, mode='w', newline='') as out_csv:
            fieldnames = ["File: Path"]
            writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(processed_paths)
        print(f"Download paths saved to: /Volumes/Samsung/Completed_Processing/DanceSongsCompleted.csv")

