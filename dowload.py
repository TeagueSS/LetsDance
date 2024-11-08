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

#Parsing our songs from our file path ->
csvPath = "/Users/teaguesangster/Code/Python/CS450/DataSetup/DanceSongs.csv"
#parseSongs(csvPath)

downloaded_files = []  # List to store file paths

# Creating a hook to get the download path of our files from our
# YoutubeDL (We need to be able to edit this stream later)

def my_hook(d):
    if d['status'] == 'finished':
        # Print the file path we downloaded
        print(f"Download completed: {d['filename']}")
        # Store final file path if it's the merged output
        downloaded_files.append(d['filename'])  # Store merged file path



def download(url):

    #ydl_opts = {'format': 'best'}
    ydl_opts_video = {
        'format': 'bestvideo/best',  # Download only the best video stream
        'outtmpl': 'downloads/%(title)s_video.%(ext)s',  # Save video with "_video" suffix in 'downloads' folder
        'merge_output_format': 'mp4',              # Set the output format for video to mp4
        'progress_hooks': [my_hook],
    }

    #Downloading the Audio and the Video Seperate

    # Setting our Audio Download Quality->
    ydl_opts_audio = {
        'format': 'bestaudio/best',                # Download only the best audio stream
        'outtmpl': 'downloads/%(title)s_audio.%(ext)s',  # Save audio with "_audio" suffix in 'downloads' folder
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',               # Set the output format for audio to mp3
            'preferredquality': '192',             # Set audio quality to 192 kbps
        }],
    }
    # Download the audio stream
    with YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])

    # Setting our video download options->

    # Download the video stream
    with YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([url])

    # Okay so the best version of the video is gonna be webk
    # Which is a Google created Open Source Library for
    # storing video -> However to get each of our frames we have to
    # Convert our video in .mp4 ->

    # Getting the avialable number of CPU cores
    num_cores = os.cpu_count()
    print("There are " + str(num_cores) + " CPU cores not in use.")
    # Giving it cores other than 2
    freeCores = max(1, num_cores -1)
    print("Giving it: " + str(freeCores) + "CPU cores -> " )

    # Here we can do this using FFMPG
    # Creating a subprocess to use the Library
    # (I have it installed locally) and used it for a different project ->
    file = "/Users/teaguesangster/Code/Python/CS450/DataSetup/" + downloaded_files[0]
    # Seeing if we need to convert our video format ->
    if file.endswith(".webm") or file.endswith(".mkv"):
        output_file = os.path.splitext(file)[0] + '.mp4'
        print(f"Converting {file} to {output_file}")
        subprocess.run(['ffmpeg', '-i', file, '-c:v', 'libx264', '-c:a', 'aac','-threads', str(freeCores), output_file])
        downloaded_files.append(output_file)  # Add the new MP4 file to the list
        os.remove(file)  # Optionally remove the original file
    else:
        print(f"File {file} is already in the desired format.")

    #This shit runs like ass
    #return output_file


# Getting one of our video links ->
#url = 'https://www.youtube.com/watch?v=dKR8rIh7aQQ'
#Downloading that video
#download(url)





# Okay so if we