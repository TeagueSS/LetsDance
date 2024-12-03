# Import Pytube
#https://pytube.io/en/latest/user/install.html
#from pytube import YouTube
import logging
import traceback

import cv2
import subprocess

from CombineAudioAndVideo import CombineAudioAndVideo
from ConvertVideo import convertVideoIntoSyncedFrames


def parseSongs(filePath):
    # Dance Files ->
    songs = dict.fromkeys(['title', 'link'])

    # Open and read the CSV file line-by-line
    with open(filePath, mode='r') as file:
        csv_reader = csv.reader(file)  # Use DictReader for column names
        for row in csv_reader:
            songs['title'] = row[0]
            songs['link'] = row[1]
            print("Title: " + songs['title'] + "    Link: " + songs['link'])
#
# #Parsing our songs from our file path ->
csvPath = "C:/Users/sangs/Code/LetsDance/DanceSongs.csv"
# #parseSongs(csvPath)
#
# downloaded_files = []  # List to store file paths
#
# # Creating a hook to get the download path of our files from our
# # YoutubeDL (We need to be able to edit this stream later)
#
# def my_hook(d):
#     if d['status'] == 'finished':
#         # Print the file path we downloaded
#         print(f"Download completed: {d['filename']}")
#         # Store final file path if it's the merged output
#         downloaded_files.append(d['filename'])  # Store merged file path
#
#
#
# def download(url):
#     # Set the base download directory
#     base_download_dir = '/Volumes/Samsung/CS450'
#
#     #ydl_opts = {'format': 'best'}
#     # Video download options
#     ydl_opts_video = {
#         'format': 'bestvideo/best',  # Download only the best video stream
#         'outtmpl': os.path.join(base_download_dir, '%(title)s_video.%(ext)s'),  # Save video in the specified folder
#         'merge_output_format': 'mp4',  # Set the output format for video to mp4
#         'progress_hooks': [my_hook],
#     }
#
#     # Audio download options
#     ydl_opts_audio = {
#         'format': 'bestaudio/best',  # Download only the best audio stream
#         'outtmpl': os.path.join(base_download_dir, '%(title)s_audio.%(ext)s'),  # Save audio in the specified folder
#         'postprocessors': [{
#             'key': 'FFmpegExtractAudio',
#             'preferredcodec': 'mp3',  # Set the output format for audio to mp3
#             'preferredquality': '192',  # Set audio quality to 192 kbps
#         }],
#     }
#     # Download the audio stream
#     with YoutubeDL(ydl_opts_audio) as ydl:
#         ydl.download([url])
#
#     # Setting our video download options->
#
#     # Download the video stream
#     with YoutubeDL(ydl_opts_video) as ydl:
#         ydl.download([url])
#
#     # Okay so the best version of the video is gonna be webk
#     # Which is a Google created Open Source Library for
#     # storing video -> However to get each of our frames we have to
#     # Convert our video in .mp4 ->
#
#     # Getting the avialable number of CPU cores
#     num_cores = os.cpu_count()
#     print("There are " + str(num_cores) + " CPU cores not in use.")
#     # Giving it cores other than 2
#     freeCores = max(1, num_cores -1)
#     print("Giving it: " + str(freeCores) + "CPU cores -> " )
#
#     # Here we can do this using FFMPG
#     # Creating a subprocess to use the Library
#     # (I have it installed locally) and used it for a different project ->
#     file =  downloaded_files[0]
#     # Seeing if we need to convert our video format ->
#     if file.endswith(".webm") or file.endswith(".mkv"):
#         output_file = os.path.splitext(file)[0] + '.mp4'
#         print(f"Converting {file} to {output_file}")
#         subprocess.run(['ffmpeg', '-i', file, '-c:v', 'libx264', '-c:a', 'aac','-threads', str(freeCores), output_file])
#         downloaded_files.append(output_file)  # Add the new MP4 file to the list
#         os.remove(file)  # Optionally remove the original file
#     else:
#         print(f"File {file} is already in the desired format.")
#         # Delete the original file after processing
#     #print(f"Deleting the original file: {file}")
#     #os.remove(file)
#     #This shit runs like ass
#     #return output_file
#
#
# # Getting one of our video links ->
# url = 'https://www.youtube.com/watch?v=dKR8rIh7aQQ'
# #Downloading that video
# download(url)
#
#

import os
import csv

# Paths
base_download_dir = "C:/Users/sangs/Code/LetsDance/DownloadedVideosForProcessing"
output_csv_path = os.path.join(base_download_dir, "downloaded_paths.csv")

# Initialize lists for storing paths
downloaded_files = []

# Hook for capturing download completion
def my_hook(d):
    if d['status'] == 'finished':
        print(f"Downloaded: {d['filename']}")
        downloaded_files.append(d['filename'])

# Download function
def download_audio_video(title, url):
    audio_path, video_path = None, None

    # Audio options
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(base_download_dir, f"{title}_audio.%(ext)s"),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [my_hook],
    }

    # Video options
    ydl_opts_video = {
        'format': 'bestvideo/best',
        'outtmpl': os.path.join(base_download_dir, f"{title}_video.%(ext)s"),
        'progress_hooks': [my_hook],
    }

    # Download audio
    with YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])
        audio_path = downloaded_files[-1]

    # Download video
    with YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([url])
        video_path = downloaded_files[-1]

    return audio_path, video_path

# Process function for all links
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


    # Once all of our videos are downloaded we need to process them
    #TODO
    """
    We need to redownload all of our videos in a format we can actually use from now on -> 
    1. Download them 
    2. Convert them to .mp4 
    3. Pass them to the DataPrep into our data base
    """


import os
from yt_dlp import YoutubeDL


def download_audio(url, download_dir):
    """
    Downloads the audio from the given URL and saves it to the specified directory.

    Parameters:
        url (str): The URL of the YouTube video.
        download_dir (str): The directory where the audio file will be saved.

    Returns:
        str: The file path of the downloaded audio.
    """
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Audio options for yt_dlp
    ydl_opts_audio = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(download_dir, '%(title)s_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Uncomment the following line if you have a progress hook function
        # 'progress_hooks': [my_hook],
    }

    with YoutubeDL(ydl_opts_audio) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        # Prepare the filename
        file_name = ydl.prepare_filename(info_dict)
        # Adjust the file extension to .mp3
        audio_file_path = os.path.splitext(file_name)[0] + '.mp3'

    return audio_file_path

    #import os
    #from yt_dlp import YoutubeDL


def download_and_transcode_video(url, output_path):
    """
    Downloads a video from the given URL and transcodes it to 1080p at 30 fps.

    Parameters:
        url (str): The URL of the video to download.
        output_path (str): The file path where the transcoded video will be saved.

    Returns:
        str: The file path of the transcoded video.
    """
    # Create a temporary directory to store the downloaded video
    temp_dir = os.path.join(os.getcwd(), 'temp_video')
    os.makedirs(temp_dir, exist_ok=True)

    # Options for yt-dlp to download the best quality video
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(temp_dir, 'downloaded_video.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    # Download the video
    with YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', None)
        downloaded_file = ydl.prepare_filename(info_dict)
        if not downloaded_file.endswith('.mp4'):
            downloaded_file = os.path.splitext(downloaded_file)[0] + '.mp4'

    # Transcode the video to 1080p at 30 fps
    transcoded_file = output_path
    ffmpeg_command = [
        'ffmpeg',
        '-i', downloaded_file,
        '-vf', 'scale=1920:1080,fps=30',
        '-c:a', 'copy',  # Copy the audio without re-encoding
        transcoded_file
    ]

    # Run the ffmpeg command
    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during transcoding: {e}")
        return None
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    return transcoded_file

# Path to the input CSV file containing YouTube titles and links
#input_csv = csvPath

# Run the process
#process_links(input_csv)

# Okay so if we

# Process function for all links
#csv_path = "/Users/teaguesangster/Code/Python/CS450/DataSetup/DanceSongs.csv"
#temporary_folder_path = "/Volumes/Samsung/Temporary/"
#final_path = "/Volumes/Samsung/Completed_Processing/"
#output_csv_path = "/Volumes/Samsung/Completed_Processing/DanceSongsCompleted.csv"
# Function call
process_links(csv_path, temporary_folder_path, final_path, output_csv_path)

