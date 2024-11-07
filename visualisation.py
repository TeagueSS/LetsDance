import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ConvertVideo import convertFrameIntoPose
import plotly.graph_objects as go

# This class exists to write methods for
# taking in a given converted file format
# and turn in into something we can actually see



def pointsTo3DSkeleton(skeletalPoints):
    # Looping for all of our points and separtaitng them
    # by their Axis ->
    # Extract x, y, z coordinates if skeletalPoints is a dictionary
    try:
        x_coords = [point['x'] for point in skeletalPoints]
        y_coords = [point['y'] for point in skeletalPoints]
        z_coords = [point['z'] for point in skeletalPoints]
    except KeyError:
        print("Error: Expected dictionary with 'x', 'y', 'z' keys.")

    # Create a scatter plot for the points
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=5, color='blue')
    )])

    # Define the connections (using index pairs) based on skeleton structure
    # Example: [(0,1), (1,2), ...] - this should be customized to your skeleton
    '''/
    Neck (0) → Left Shoulder (11) → Left Elbow (13) → Left Wrist (15)
    Neck (0) → Right Shoulder (12) → Right Elbow (14) → Right Wrist (16)
    Left Hip (23) → Left Knee (25) → Left Ankle (27)
    Right Hip (24) → Right Knee (26) → Right Ankle (28)'''
    connections = [
         (11, 13), (13, 15),  # Example connections
         (12,14), (14, 16),
        (23, 25) , (25,27), (23,24), (11,12), (11,24) ,(12,23),
        (24,26),(26,28)
        # Continue with other connections based on your skeleton
    ]

    # Add lines for each connection
    for start, end in connections:
        fig.add_trace(go.Scatter3d(
            x=[x_coords[start], x_coords[end]],
            y=[y_coords[start], y_coords[end]],
            z=[z_coords[start], z_coords[end]],
            mode='lines',
            line=dict(color='gray', width=2)
        ))

    # Set axis labels and show the plot
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=0)  # Modify 'z' for zoom level as needed
    )
    ))

    fig.show()


#Defining our test file
testPath = "/Users/teaguesangster/Code/Python/CS450/DataSetup/VideoFrames/Only Girl Rihanna/frame_0747.png"
# Running our test to get our data from one of our file paths
pointsTo3DSkeleton(convertFrameIntoPose(testPath))


