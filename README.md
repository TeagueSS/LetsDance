
# Lets Dance!

Lets dance is a School Project for CS450 meant to generate dance routines from imported.mp3 files. The code base is separated into folders depending on what stage of usage you are in. 
    
    Participants:
        Teague Sangster
        Patrick Huynh
        Simmon Quan
        Nanci Cardenas Martinez



# How to Run
    1. Create a new Anaconda Environment and install the packages listed at the bottom: 
        (2 environments required, one for processing and one for running)

    2. Pick a List of Videos you wish to train your model off of or use the DanceSongs.csv provided in the path.
     (We are not Liable for the Legal Ramifications of downloading, processing and/or training off of youtube Licensed property USE YOUTUBE VIDEOS AT YOUR OWN RISK)

    2. Navigate to the DataPrep Folder and open download.py, Now go to the method called process_links and look at the assigned variables above. 
    They are what you will pass this function for your list, where to save, as well as any temporary holding folder for videos that will be deleted after running.

        ^^^^^ This step will take about a day as Downloading, transcoding, splitting, cropping, and then finally encoding for each video is quite intensive. ^^^^
        This task is optimized for High CPU core counts and uses a ThreadPool based approach. 
        (This program ran on a 14 Core Apple M3 cpu, threading is highly optimized and will hit your Cores HARD)
        You can make the operation Single threaded by editing the code in CombineAudioAndVideo.py (process_audio_and_video_frames_Multi_Threaded method)

    3. Assuming your Data is downloaded and formatted to a folder on your Local machine open TrainAndSaveDanceRNN.py and give it the folder path. 
    Once it has the folder path you can run the main method and let it run. This will take a while. 

    4. Assuming your model is trained you can open BuildDanceFromAudioFile and define your saved paths from your model. Finally at the bottom call the method with your song file and it should generate a dance.
    Restart your song as you press play on the browser it opens and see your buddy come to life!
    

# Folders and their purpose:
## Data Prep Folder: 
Contains all of the needed methods for downloading videos, processing them, and then turning them into .npz arrays to be trained on for an RNN

    Notable Methods:
        Audio Handler: Uses Librosa to import and extract audio features 

        Audio Slicer: Takes in an Audio Handler and a frame rate, then create little subsections that correspond with those frames

        Encode: Contains multiple methods for saving data without prepping it for tensorflow

            SyncedSkeletonDataAndAudio: Takes in both SkeletonData and AudioData and allows for their respective "Moments" to be encoded together

            SkeletonData:
                Holds Positional Data for a skeleton

            AudioData:
                Holds Audio data from AUDIO_HANDLER

            DataSaver:
                Writes data to hard drive to be 
                imported later 
        
        Convert Video: Contains methods for turning a video into frames

        Download.py **THE MAIN METHOD HERE** 
            CONTAINS METHODS FOR DOWNLOADING, PROCESSING, AND ENCODING ALL IN ONE MULTITHREADED LOOP. 
             This task is optimized for High CPU core counts and uses a ThreadPool based approach. 
        (This program ran on a 14 Core Apple M3 cpu, threading is highly optimized and will hit your Cores HARD)
        You can make the operation Single threaded by editing the code in CombineAudioAndVideo.py (process_audio_and_video_frames_Multi_Threaded method)





## Training 

Contains methods for Training your model with different Approaches in each File 
    
    DanceRNNVersion1, a very simple RNN built on just one LTSM layer and a very simple context window of 10 frames. 
    (For Testing Purposes Only)

    TensorFlow Data Prep: For converting Audio and Skeletal Data to saveable numpy Arrays. 
    -> If a frame needs to be trained on it gets formatted by this guy 

## Testing 
Files for testing your imports, that TensorFlow works, your environment. 
    
    For Video Conversion: Running the last method at the bottom of tests.py should indicate if packages are installed for Processing.
    For Model Training: running the script in TestingPerformance.py should see if Tensorflow is running on CPU or GPU. 



## Required Packages:

    Please Look at the attached .yml files, 
        -> one is labeled for Downloading (Contains Librosa and Youtube download methods)
        -> The other is for running this model using TensorFlow
    






