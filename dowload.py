# Import Pytube
#https://pytube.io/en/latest/user/install.html
#from pytube import YouTube
import os

from yt_dlp import YoutubeDL
import subprocess
import csv

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
from yt_dlp import YoutubeDL

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
def process_links(csv_file):
    # Read the CSV with YouTube links
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present

        # Prepare list for saving to output CSV
        download_paths = []

        for row in csv_reader:
            title, url = row
            print(f"Processing: {title} | {url}")

            # Download and get paths
            try:
                audio_path, video_path = download_audio_video(title, url)
                download_paths.append({"Audio Path": audio_path, "Video Path": video_path})
            except Exception as e:
                print(f"Error downloading {title}: {e}")

        # Save paths to output CSV
        with open(output_csv_path, mode='w', newline='') as out_csv:
            writer = csv.DictWriter(out_csv, fieldnames=["Audio Path", "Video Path"])
            writer.writeheader()
            writer.writerows(download_paths)

        print(f"Download paths saved to: {output_csv_path}")

    # Once all of our videos are downloaded we need to process them
    #TODO
    """
    We need to redownload all of our videos in a format we can actually use from now on -> 
    1. Download them 
    2. Convert them to .mp4 
    3. Pass them to the Conversion into our data base
    """

# Path to the input CSV file containing YouTube titles and links
input_csv = csvPath

# Run the process
process_links(input_csv)

# Okay so if we