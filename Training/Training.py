import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint


# Load and Combine Data
def combine_npz_files(input_directory):
    """
    Combine multiple .npz files from a directory into a single dataset.
    """
    audio_data_list = []
    frame_data_list = []

    # Iterate through all .npz files in the input directory and load their contents
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(input_directory, file_name)
            print(f"Loading data from {file_path}...")

            # Load data from the .npz file
            data = np.load(file_path)
            audio_data = data['audio']
            frame_data = data['frame']

            # Append loaded data to lists
            audio_data_list.append(audio_data)
            frame_data_list.append(frame_data)

    # Concatenate all loaded arrays into one long array
    combined_audio_data = np.concatenate(audio_data_list, axis=0)
    combined_frame_data = np.concatenate(frame_data_list, axis=0)

    return combined_audio_data, combined_frame_data


# Preprocess Data for Training
def preprocess_data(audio_data, frame_data, time_steps=10):
    """
    Preprocess audio and frame data to create time-step sequences for training an RNN.
    """
    # Normalize the audio data only
    # (because our body values will be between 0 and 1 for all indecies ->)

    #Building our frame scaler ->
    frame_scaler = MinMaxScaler()
    frame_data_normalized = frame_scaler.fit_transform(frame_data)

    audio_scaler = MinMaxScaler()
    audio_data_normalized = audio_scaler.fit_transform(audio_data)


    # Then we build our 3 sub sections based off of this ->
    # Create sequences of time steps for audio input, previous frame input, and frame output
    num_samples = audio_data_normalized.shape[0] - time_steps - 1
    audio_sequences, previous_frames, next_frames = [], [], []

    # Looping through all of our samples for this training window ->
    for i in range(num_samples):
        # For our audio window, it's every audio timings :
        audio_sequences.append(audio_data_normalized[i:i + time_steps])
        # For our Frame data it's the same but missing the first
        # Because there's no Guarente it has a starting position ->
        previous_frames.append(frame_data_normalized[i + time_steps - 1])
        # Then here is sequence of next frames it's meant to predict ->
        next_frames.append(frame_data_normalized[i + time_steps])

    # Converting these "Frames" into arrays that can be fed into our model
    X_audio = np.array(audio_sequences)
    X_prev_frame = np.array(previous_frames)
    y_next_frame = np.array(next_frames)

    #Returning our arrays of frames:
    return X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler


# Build the RNN Model
def build_rnn_model(audio_input_shape, frame_input_shape):
    """
    Here we have seperate input layers, as we have different information for our context:
    """
    # Audio Layer ->
    # Here we are passing our audio frame shape and naming it
    audio_input = Input(shape=audio_input_shape, name='audio_input')
    # Using a LSTM for our Nuerons as we want to see the changes in our song
    # And actually keep track of how our audio is changing overtime
    lstm_out = LSTM(100, activation='tanh')(audio_input)

    # Second layer for previous frame (So we know where our model is standing)
    prev_frame_input = Input(shape=frame_input_shape, name='previous_frame_input')

    # Combination Layer (Where our previous Skeletal Stance and CURENT audio get combined)
    concat = Concatenate()([lstm_out, prev_frame_input])

    # Our predicition layer of the RNN
    # we are using a dense layer as we want to represent the entire shape of the body
    # and give each body part it's own neuron to learn from
    output = Dense(frame_input_shape[0], activation='linear')(concat)

    # Finally turning all of our layers into one by "Compling" our model ->
    model = Model(inputs=[audio_input, prev_frame_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    #Returning the model (Layers) we just made
    return model


# Train the RNN Model with Model Checkpointing
def train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, output_model_path, epochs=50, batch_size=32):
    """
    Train the RNN model with checkpointing to save the best model during training.
    """
    # Split data into training and validation sets
    X_audio_train, X_audio_val, X_prev_frame_train, X_prev_frame_val, y_train, y_val = train_test_split(
        X_audio, X_prev_frame, y_next_frame, test_size=0.2, random_state=42
    )

    # Set up checkpointing to save the best model during training
    checkpoint_callback = ModelCheckpoint(filepath=output_model_path,
                                          save_best_only=True,
                                          monitor='val_loss',
                                          mode='min',
                                          verbose=1)

    # Train the model
    history = model.fit(
        [X_audio_train, X_prev_frame_train], y_train,
        validation_data=([X_audio_val, X_prev_frame_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_callback],
        verbose=1
    )

    return history, [X_audio_val, X_prev_frame_val], y_val


# Evaluate and Make Predictions
def evaluate_and_predict(model, X_val, y_val, frame_scaler, num_predictions=5):
    """
    Evaluate the trained model and make predictions on the validation set.
    """
    # Evaluate the model on the validation set
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss: {val_loss}")
    print(f"Validation MAE: {val_mae}")

    # Make predictions on validation set
    predictions = model.predict(X_val)

    # Invert scaling to interpret predictions in the original scale
    predictions_unscaled = frame_scaler.inverse_transform(predictions)
    y_val_unscaled = frame_scaler.inverse_transform(y_val)

    # Print the first few predictions and their corresponding actual values
    print("Predictions vs Actual Values:")
    for i in range(num_predictions):
        print(f"Predicted Frame {i + 1}: {predictions_unscaled[i]}")
        print(f"Actual Frame {i + 1}: {y_val_unscaled[i]}")


# Main Function to Run the Entire Pipeline
if __name__ == "__main__":
    # Define paths for loading and saving data and models
    # Path for all of our data:
    input_directory = "/path/to/your/npz/files"
    # Path for our training weights
    output_model_path = "/path/to/your/saved_audio_to_frame_model_with_context.h5"

    # Load and combine data
    audio_data, frame_data = combine_npz_files(input_directory)

    # Preprocess the data to create sequences for the RNN
    time_steps = 10  # Number of time steps to look back
    X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(audio_data, frame_data,
                                                                                      time_steps)

    # Build the RNN model
    audio_input_shape = (X_audio.shape[1], X_audio.shape[2])  # Shape is (time_steps, audio_features)
    frame_input_shape = (X_prev_frame.shape[1],)  # Shape is (number of frame features,)
    model = build_rnn_model(audio_input_shape, frame_input_shape)

    # Train the RNN model with checkpointing
    epochs = 50
    batch_size = 32
    history, X_val, y_val = train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, output_model_path, epochs,
                                            batch_size)

    # Evaluate the model and make predictions
    evaluate_and_predict(model, X_val, y_val, frame_scaler)

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def predict_body_mappings(model_path, audio_scaler, frame_scaler, audio_input_array, initial_frame, num_predictions):
    """
    Predict body mappings for given audio input sequences using a pre-trained model.

    Parameters:
    - model_path (str): Path to the saved model (.h5 file).
    - audio_scaler (MinMaxScaler): Scaler used for normalizing audio data.
    - frame_scaler (MinMaxScaler): Scaler used for normalizing frame data.
    - audio_input_array (ndarray): Input audio data with shape (num_timesteps, features).
    - initial_frame (ndarray): Initial body frame with shape (features,) to start the prediction.
    - num_predictions (int): Number of body frames to predict.

    Returns:
    - predicted_frames (ndarray): Predicted body frames with shape (num_predictions, features).
    """

    # Load the pre-trained model
    model = load_model(model_path)

    # Normalize the audio input using the scaler
    audio_input_normalized = audio_scaler.transform(audio_input_array)

    # Prepare an empty list to collect predicted frames
    predicted_frames = []

    # Set the initial previous frame to the given initial frame
    previous_frame = initial_frame

    # Predict body frames one by one
    for i in range(num_predictions):
        # Add a batch dimension to the input (since the model expects batched input)
        audio_input_reshaped = np.expand_dims(audio_input_normalized, axis=0)  # Shape (1, num_timesteps, features)
        previous_frame_reshaped = np.expand_dims(previous_frame, axis=0)  # Shape (1, features)

        # Make a prediction
        predicted_frame_normalized = model.predict([audio_input_reshaped, previous_frame_reshaped])

        # Invert the normalization on the predicted frame
        predicted_frame = frame_scaler.inverse_transform(predicted_frame_normalized)

        # Add the predicted frame to the list
        predicted_frames.append(predicted_frame.flatten())

        # Update the previous frame for the next prediction
        previous_frame = predicted_frame.flatten()

        # Optional: Shift the audio input for continuous prediction
        # Assuming new audio information is available, you could shift the audio input to always keep `num_timesteps`.
        # Here, we use the existing sequence as an example.
        if i + 1 < len(audio_input_normalized):
            audio_input_normalized = np.roll(audio_input_normalized, -1, axis=0)  # Shift one time step to the left
            audio_input_normalized[-1] = audio_input_normalized[-1]  # Optional: Add new audio features if available

    # Convert list to ndarray for consistency
    predicted_frames = np.array(predicted_frames)

    return predicted_frames
