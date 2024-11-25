import os.path
from moviepy.editor import VideoFileClip

def cropVideo(videoPath: str, time_to_remove_from_front, time_to_remove_from_end):
    #load video file
    video = VideoFileClip(videoPath)

    #getting the new start & end times
    start_time = time_to_remove_from_front * 1000
    end_time = video.duration - (time_to_remove_from_end * 1000)

    #crop the video
    trimmedVideo = video.subclip(start_time, end_time)

    #make new filepath for cropped video
    new_file_path = f"{os.path.splitext(videoPath)[0]}_cropped.mp4"
    
    #save cropped video
    trimmedVideo.write_videofile(new_file_path)

    #delete non-cropped original video
    os.remove(videoPath)

    return new_file_path