import joblib
from keras.src.saving import load_model
import numpy as np
from DataPrep.AudioHandler import AudioHandler
from DataPrep.AudioSlicing import AudioFrameProcessor
from TrainAndSaveDanceRNN import predict_body_mappings

#Song to generate dances for ->
song_path = "/Users/teaguesangster/Code/Python/CS450/DataSetup/downloads/Just Dance Hits： Only Girl (In The World) by Rihanna [12.9k]_audio.mp3"

# Trained Model Weights ->

# Paths to The saved models ->
model_path = '/Users/teaguesangster/Desktop/saved_audio_to_frame_model2_with_context.keras'
audio_scaler_path = '/Users/teaguesangster/Desktop/audio_scaler.save'
frame_scaler_path = '/Users/teaguesangster/Desktop/frame_scaler.save'


import numpy as np
import plotly.graph_objects as go

def animate_skeleton(frames):
    """
    Animate a sequence of 3D skeletal frames.

    Parameters:
    frames (list of np.ndarray): A list where each element is a numpy array of shape (30,)
                                  representing the x, y, z coordinates of 10 body parts.
    """
    # Define the connections between body parts
    connections = [
        (0, 1), (1, 2), (2, 3),  # Neck → Right Shoulder → Right Elbow → Right Wrist
        (4, 5), (5, 6),          # Left Hip → Left Knee → Left Ankle
        (7, 8), (8, 9),           # Right Hip → Right Knee → Right Ankle
        (7,4) ,(1,7) ,(4,7), (4,1) ,(0,7)
    ]

    # Initialize the figure
    fig = go.Figure(
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[-1, 1], autorange=False),
                yaxis=dict(range=[-1, 1], autorange=False),
                zaxis=dict(range=[-1, 1], autorange=False),
            ),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", method="animate", args=[None])]
            )]
        )
    )

    # Add initial scatter plot
    initial_frame = frames[0]
    x_coords = initial_frame[0::3]
    y_coords = initial_frame[1::3]
    z_coords = initial_frame[2::3]

    fig.add_trace(go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=5, color='blue')
    ))

    # Add lines for each connection
    for start, end in connections:
        fig.add_trace(go.Scatter3d(
            x=[x_coords[start], x_coords[end]],
            y=[y_coords[start], y_coords[end]],
            z=[z_coords[start], z_coords[end]],
            mode='lines',
            line=dict(color='gray', width=2)
        ))

    # Create frames for animation
    animation_frames = []
    for frame in frames:
        x_coords = frame[0::3]
        y_coords = frame[1::3]
        z_coords = frame[2::3]

        data = [go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(size=5, color='blue')
        )]

        for start, end in connections:
            data.append(go.Scatter3d(
                x=[x_coords[start], x_coords[end]],
                y=[y_coords[start], y_coords[end]],
                z=[z_coords[start], z_coords[end]],
                mode='lines',
                line=dict(color='gray', width=2)
            ))

        animation_frames.append(go.Frame(data=data))

    # Add frames to the figure
    fig.frames = animation_frames

    # Show the plot
    fig.show()

# Example usage:
# Assuming `skeletal_frames` is a list of numpy arrays, each of shape (30,)
# representing the x, y, z coordinates of the 10 body parts.
# animate_skeleton(skeletal_frames)



def generateSequnence(song_path):
    #Checking we can import our files ->
    # Load the model
    model = load_model(model_path)

    # Load the scalers
    audio_scaler = joblib.load(audio_scaler_path)
    frame_scaler = joblib.load(frame_scaler_path)

    # Creating an audio slicer of this song
    # Audio Handler to prep our audio stream
    audio_handler = AudioHandler(song_path)
    # Passing our audio slicer our audio handler
    audio_slicer = AudioFrameProcessor(30, beat_times=audio_handler.beat_times, onset_env=audio_handler.onset_env)
    # Slicing all of our Audio Frames ->
    audio_slicer.process_audio_features(audio_handler.duration)


    # Building an array out of those entries ->
    array_of_audio_frames = audio_slicer.get_all_features()

    # Now that we have all of our audio slices we can use our model
    # to build our dance ->
    #predict_body_mappings()

    # Creating an array to hold all of the frames
    all_frames = []

    # Defining all of the variables our method needs ->
    audio_input_array = array_of_audio_frames[0:60]  # NumPy array of shape (time_steps, audio_features)

    audio_sub_amount = 60

    initial_frame = [0.7208858132362366, 0.0, 0.0, 0.4229571521282196, 0.14309634268283844, 0.6275824308395386,
                     0.18961410224437714, 0.2863653600215912, 0.7556061148643494, 0.0, 0.4257323741912842,
                     0.5054376125335693, 0.9413426518440247, 0.49442195892333984, 0.8388641476631165,
                     0.9241803288459778, 0.766042947769165, 0.549893856048584, 1.0, 1.0, 1.0, 0.6355255842208862,
                     0.5008254647254944, 0.9971609711647034, 0.30072176456451416, 0.6921217441558838,
                     0.4080755412578583, 0.3766336739063263, 0.9690350890159607, 0.8619300723075867]

    # Increase the range for every second of requested frames ->
    for i in range(3):

        begining = audio_sub_amount *  i
        print(begining)

        end = audio_sub_amount * (i + 1)
        print(end)
        # Setting our subrange
        audio_input_array =array_of_audio_frames[begining:end]
        # Random starting shape ->
        # A set of audio input arrays
        num_predictions = int(len(audio_input_array))

        # Call the prediction function
        predicted_frames = predict_body_mappings(
            model,
            audio_scaler,
            frame_scaler,
            audio_input_array,
            initial_frame,
            num_predictions
        )

        #Updating our new start position ->
        initial_frame = predicted_frames[0]
        # Process the predictions
        for i, frame in enumerate(predicted_frames):
            print(f"Predicted Frame {i + 1}: {frame}")


    animate_skeleton(predicted_frames)


generateSequnence(song_path)
