import os.path
from pydub import AudioSegment

def cropVideo(videoPath: str, time_to_remove_front, time_to_remove_end):
    #load audio file
    audio = AudioSegment.from_file(videoPath, format= "mp3")

    #getting the new start time and end time
    start_time = time_to_remove_front * 1000
    end_time = len(audio) - (time_to_remove_end * 1000)

    #crop the audio
    cropped_audio = audio[start_time:end_time]

    #make a new file path for cropped audio
    new_file_path = f"{os.path.splitext(videoPath)[0]}_cropped.mp3"

    #save new audio file
    cropped_audio.export(new_file_path, format= "mp3")

    # delete original file
    os.remove(videoPath)

    return new_file_path