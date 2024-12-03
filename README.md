
# Lets Dance!

Lets dance is a School Project for CS450 meant to generate dance routines from imported.mp3 files. The code base is sepereated into folders depending on what stage of usage you are in. 
    
    Participants:
        Teague Sangster
        Patrick Huynh
        Simmon Quan
        Nanci Cardenas Martinez



# How to Run
    1. Create a new Anaconda Enviorment and install the packages listed at the bottom: 
        (2 enviorments required, one for processing and one for running)

    2. Pick a List of Videos you wish to train your model off of or use the DanceSongs.csv provided in the path.
     (We are not Liable for the Legal Remifications of downloading, processing and/or training off of youtube Licensed property USE YOUTUBE VIDEOS AT YOUR OWN RISK)

    2. Navagate to the DataPrep Folder and open download.py, Now go to the method called process_links and look at the assigned variabels above. 
    They are what you will pass this function for your list, where to save, as well as any temporary holding folder for videos that will deleted after running.

        ^^^^^ This step will take about a day as Downloading, transcoding, splitting, cropping, and then fianlly encoding for each video is quite Intensive. ^^^^
        This task is optomized for High CPU core counts and usses a ThreadPool based approach. 
        (This program ran on a 14 Core Apple M3 cpu, threading is highlighy optomized and will hit your Cores HARD)
        You can make the opperation Single threaded by editing the code in CombineAudioAndVideo.py (process_audio_and_video_frames_Multi_Threaded method)

    3. Assuming your Data Is downlaoded and formatted to a folder on your Local machine open TrainAndSaveDanceRNN.py and give it the folder path. 
    Once it has the folder path you can run the main method and let it run. This will take awhile. 

    4. Assuming your model is trained you can open BuildDanceFromAudioFile and define your saved paths from your model. Finally at the bottom call the method with your song file and it should generate a dance.
    Restart your song as you press play on the browser it opens and see your buddy come to life!
    

# Folders and their purpose:
## Data Prep Folder: 
Contains all of the needed methods for downloading videos, processing them, and then turning them into .npz arrays to be trained on for an RNN

    Notable Methods:
        Audio Handler: Uses Librosa to import and extract audio features 

        Audio Slicer: Takes in an Audio Handler and a frame rate, then creates little subsections that correspond with those frames

        Encode: Contains multiple methods for saving data without prepping it for tensorflow

            SyncedSkeletonDataAndAudio: Takes in both SkeletonData and AudioData and allows for their respective "Moments" to be enconded together

            SkeletonData:
                Holds Positional Data for a skeleton

            AudioData:
                Holds Audio data from AUDIO_HANDLER

            DataSaver:
                Writes data to hard drive to be 
                imported later 
        
        Convert Video: Contains methods for turning a video into frames

        Download.py **THE MAIN METHOD HERE** 
            CONTAINS METHODS FOR DOWNLAODING, PROCESSING, AND ENCODING ALL IN ONE MULTITHREADED LOOP. 
             This task is optomized for High CPU core counts and usses a ThreadPool based approach. 
        (This program ran on a 14 Core Apple M3 cpu, threading is highlighy optomized and will hit your Cores HARD)
        You can make the opperation Single threaded by editing the code in CombineAudioAndVideo.py (process_audio_and_video_frames_Multi_Threaded method)





## Training 

Contains methods for Training your model with differnt Approaches in each File 
    
    DanceRNNVersion1, a very simple RNN built on just one LTSM layer and a very simple context window of 10 frames. 
    (For Testing Puroses Only)

    TensorFlow Data Prep: For converting Audio and Skeletal Data to saveable numpy Arrays. 
    -> If a frame needs to be trained on it gets formatted by this guy 

## Testing 
Files for testing your imports, that TensorFlow works, your enviorment. 
    
    For Video Conversion: Running the last method at the bottom of tests.py should indicate if packages oare installed for Processing.
    For Model Training: running the script in TestingPerformace.py should see if Tensorflow is running on CPU or GPU. 



## Required Packages:

    Please Look at the attached .yml files, 
        -> one is labeled for Downlaoding (Contains Librosa and Youtube downlaod methdos)
        -> The other is for running this model using TensorFlow
    






