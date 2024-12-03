import logging
import os

from CombineAudioAndVideo import CombineAudioAndVideo
from ConvertVideo import convertVideoIntoSyncedFrames
from Testing.Tests import OUTPUT_PATH
from dowload import download_audio, download_and_transcode_video



# Here we need to get all of our videos ->

# TODO
# Defining any paths we are going to use ->

# 1. Our Youtube Link File Path
csv_of_youtube_files = "/Users/teaguesangster/Code/Python/CS450/DataSetup/DanceSongs.csv"
# 2. Where we want to write our temporary Videos
video_download_folder = "/Users/teaguesangster/Code/Python/CS450/DataSetup/Temporary"
# 3. Where we will write all of our Processed File paths ->
processed_file_path = "/Temporary"
# 4. Where we will store our .npz Folders ->
csv_of_processed_files = "/Users/teaguesangster/Code/Python/CS450/DataSetup/Temporary"

# Okay so now for the hard part, we need to download our videos one at a time ->

counter = 0


# Lets just get 1 video to test it for now
testVideo = "https://www.youtube.com/watch?v=MJTEt4aPhFQ"
temporary_audio_folder = "/Users/teaguesangster/Code/Python/CS450/DataSetup/Temporary"
temporary_video_folder = "/Users/teaguesangster/Code/Python/CS450/DataSetup/Temporary"

# Assume here we are looping
link = testVideo
# get our audio and our video
song_download = download_audio(url=link, download_dir=temporary_audio_folder)
# For our video we need to define a new directory ->
output_directory = '/Users/teaguesangster/Code/Python/CS450/DataSetup/Temporary'
output_filename = 'transcoded_video.mp4'
output_path = os.path.join(output_directory, output_filename)
# Now that we have defined our dicrectory downloading
video_download = download_and_transcode_video(url=link , output_path = output_path  )





#TODO Step 2 turning our videos into frames ->

# Getting Our Video Path ->
logging.info(f"Converting video '{output_path}' into frames...")
# # Creating a temporary CSV to hold our frame list
# # Here we pass our video, our output folder, and our "song" name
csv_Path = convertVideoIntoSyncedFrames(video_download, OUTPUT_PATH, str(output_path))
# Saying we succeded
logging.info("Video conversion completed for video" + str(output_path))
# Now that we have our Video information lets get our audio information
ActionHandler = CombineAudioAndVideo(str(counter))
# Getting our synced fame list ->
sycned_frames = ActionHandler.map_csv_of_frames(csv_Path)

print("Turning frames into Skeleton")

# Settting our audio to be parsed too ->
frames = ActionHandler.process_audio_and_video_frames_Multi_Threaded(sycned_frames, str(counter), song_download,
                                                                     30)
print("We Processed: ")
print(frames.get_length())

# Sort our frames before saving them
frames.sort_frame_entries()
frames.print_frames_information()
#
# # Make our save file path ->
# frames.save(os.path.join(csv_of_processed_files + "counter"))
#
# base_download_dir = "C:/Users/sangs/Code/LetsDance/DownloadedVideosForProcessing"
# output_csv_path = os.path.join(base_download_dir, "downloaded_paths.csv")
# print("Converted Video! ")

# #TODO add our video path ->