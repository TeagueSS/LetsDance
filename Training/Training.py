import os
import numpy as np
import datetime
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorboard.program import TensorBoard
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks

# Define paths for loading and saving data and models
# Replace with your actual paths
input_directory = "/Users/teaguesangster/Desktop/ProcessedEntries"
output_model_path = "/Users/teaguesangster/Desktop/saved_audio_to_frame_model2_with_context.keras"



import os
import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
#print(tf.config.list_physical_devices('GPU'))
# Set environment variables to control threading behavior
os.environ['TENSORFLOW_INTRA_OP_PARALLELISM_THREADS'] = '30'  # Number of threads for operations like matrix multiplication
os.environ['TENSORFLOW_INTER_OP_PARALLELISM_THREADS'] = '14'  # Number of threads for independent operations

# Alternatively, set threading using TensorFlow's configuration methods
tf.config.threading.set_intra_op_parallelism_threads(30)
tf.config.threading.set_inter_op_parallelism_threads(14)

#configure GPU settings
gpus = tf.config.list_physical_devices('GPU')
#Set the thread mode to dedicate threads to GPU operations
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
if gpus:
    try:
        # Set memory growth to avoid TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # If you want to set a specific memory limit (e.g., 4096 MB), uncomment the following lines:
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)




# Set a custom learning rate
learning_rate = 0.001  # You can experiment with this value
optimizer = Adam(learning_rate=learning_rate)

# Log directory for TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Model checkpoint to save the best model
checkpoint_callback = ModelCheckpoint(
    filepath=output_model_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Stop training when a monitored metric has stopped improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Combine callbacks
callbacks = [checkpoint_callback, reduce_lr, early_stopping, tensorboard_callback]


# Load and Combine Data
# Here we have all of our idividual trainings, and if we get more we can find them
# Here
def combine_npz_files(input_directory):
    audio_data_list = []
    frame_data_list = []

    # Iterate through all .npz files in the input directory and load their contents
    for file_name in os.listdir(input_directory):
        # Skip hidden files and system files
        if file_name.endswith('.npz') and not file_name.startswith(('.', '_')):
            file_path = os.path.join(input_directory, file_name)
            print(f"Loading data from {file_path}...")

            try:
                # Load data from the .npz file
                data = np.load(file_path)
                audio_data = data['audio']
                frame_data = data['frame']

                # Append loaded data to lists
                audio_data_list.append(audio_data)
                frame_data_list.append(frame_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue  # Skip this file and continue with the next

    # Check if any data was loaded
    if not audio_data_list or not frame_data_list:
        raise ValueError("No valid data was loaded from the .npz files.")

    # Concatenate all loaded arrays into one long array
    combined_audio_data = np.concatenate(audio_data_list, axis=0)
    combined_frame_data = np.concatenate(frame_data_list, axis=0)

    return combined_audio_data, combined_frame_data

def preprocess_data(audio_data, frame_data, time_steps=60):
    """
    Preprocess audio and frame data to create time-step sequences for training an RNN.
    Here we have to make our valid "Sequences" For our data to use for trianing.

    """
    # Normalzing our Audio data so we can ensure it trains in the correct way
    # and also reverses in the correct format ->
    audio_scaler = MinMaxScaler()
    audio_data_normalized = audio_scaler.fit_transform(audio_data)

    # Normalize the frame data (even if it's between 0 and 1)
    # We normaize it again as we can use our normalizer to reverse our encoding later on ->


    frame_scaler = MinMaxScaler()
    frame_data_normalized = frame_scaler.fit_transform(frame_data)

    # Create sequences of time steps for audio input, previous frame input, and frame output
    # Here our samples are everything except for the first one for our Frames,
    # (We want our model to always know where to start the dance)
    num_samples = audio_data_normalized.shape[0] - time_steps - 1
    audio_sequences, previous_frames, next_frames = [], [], []

    # Loop through all of our samples for this training window
    for i in range(num_samples):
        # For our audio window, it's every audio timing:
        audio_sequences.append(audio_data_normalized[i:i + time_steps])
        # Previous frame (where the model is currently standing)
        previous_frames.append(frame_data_normalized[i + time_steps - 1])
        # Next frame to predict
        next_frames.append(frame_data_normalized[i + time_steps])

    # Convert these "Frames" into arrays that can be fed into our model
    X_audio = np.array(audio_sequences)
    X_prev_frame = np.array(previous_frames)
    y_next_frame = np.array(next_frames)

    # Return our arrays of frames and scalers
    return X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler

# Build the RNN Model
def build_rnn_model(audio_input_shape, frame_input_shape):
    # Audio input layer
    audio_input = Input(shape=audio_input_shape, name='audio_input')

    # First LSTM layer with increased units and return_sequences=True
    lstm_out_1 = LSTM(256, return_sequences=True)(audio_input)
    lstm_out_1 = Dropout(0.5)(lstm_out_1)

    # Second LSTM layer
    lstm_out_2 = LSTM(128)(lstm_out_1)
    lstm_out_2 = Dropout(0.5)(lstm_out_2)

    # Previous frame input layer
    prev_frame_input = Input(shape=frame_input_shape, name='previous_frame_input')

    # Concatenate LSTM output and previous frame input
    concat = Concatenate()([lstm_out_2, prev_frame_input])

    # Dense layer with LeakyReLU activation
    dense_out = Dense(64)(concat)
    dense_out = LeakyReLU(alpha=0.1)(dense_out)

    # Output layer
    output = Dense(frame_input_shape[0], activation='linear')(dense_out)

    # Build and compile the model
    model = Model(inputs=[audio_input, prev_frame_input], outputs=output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Return the model
    return model

def train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, output_model_path, epochs=50, batch_size=32, callbacks=None):
    # Split data into training and validation sets
    X_audio_train, X_audio_val, X_prev_frame_train, X_prev_frame_val, y_train, y_val = train_test_split(
        X_audio, X_prev_frame, y_next_frame, test_size=0.2, random_state=42
    )

    # Train the model
    history = model.fit(
        [X_audio_train, X_prev_frame_train], y_train,
        validation_data=([X_audio_val, X_prev_frame_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history, [X_audio_val, X_prev_frame_val], y_val

# Evaluate and Make Predictions
def evaluate_and_predict(model, X_val, y_val, frame_scaler, num_predictions=5):
    """
    Evaluate the trained model and make predictions on the validation set.
    """
    # Evaluate the model on the validation set
    # Here we predict our loss (How far off) we are from our desired entries
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=1)
    # We pring it out as we train so we can see these numbers hopefully go down overtime ->
    print(f"Validation Loss: {val_loss}")
    print(f"Validation MAE: {val_mae}")

    # Make predictions on validation set
    predictions = model.predict(X_val)

    # Invert scaling to interpret predictions in the original scale
    # Here because we compressed our data so we can train off of it,
    # We let it scale back up so we can actually Visulaize our results ->
    predictions_unscaled = frame_scaler.inverse_transform(predictions)
    y_val_unscaled = frame_scaler.inverse_transform(y_val)

    # Print the first few predictions and their corresponding actual values
    print("Predictions vs Actual Values:")
    for i in range(num_predictions):
        print(f"Predicted Frame {i + 1}: {predictions_unscaled[i]}")
        print(f"Actual Frame {i + 1}: {y_val_unscaled[i]}")




   # Main Function to Run the Entire Pipeline
if __name__ == "__main__":
    # Load and combine data
    audio_data, frame_data = combine_npz_files(input_directory)

    # Preprocess the data to create sequences for the RNN
    time_steps = 60  # Ensure this matches the value in preprocess_data
    X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(
        audio_data, frame_data, time_steps
    )

    # Build the RNN model
    audio_input_shape = (X_audio.shape[1], X_audio.shape[2])  # Shape is (time_steps, audio_features)
    frame_input_shape = (X_prev_frame.shape[1],)  # Shape is (number of frame features,)
    model = build_rnn_model(audio_input_shape, frame_input_shape)

    # Train the RNN model with callbacks
    epochs = 50
    batch_size = 32
    history, X_val, y_val = train_rnn_model(
        model, X_audio, X_prev_frame, y_next_frame, output_model_path, epochs, batch_size, callbacks=callbacks
    )

    # Evaluate the model and make predictions
    evaluate_and_predict(model, X_val, y_val, frame_scaler)



# Prediction Function
# Here we can provide audio and actually see our model return frames
# Which we can then later Visualize through Predict Body mappings
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

    # Normalize the initial frame
    previous_frame_normalized = frame_scaler.transform([initial_frame])[0]

    # Prepare an empty list to collect predicted frames
    predicted_frames = []

    # Predict body frames one by one
    for i in range(num_predictions):
        # Add a batch dimension to the input (since the model expects batched input)
        audio_input_reshaped = np.expand_dims(audio_input_normalized, axis=0)  # Shape (1, num_timesteps, features)
        prev_frame_reshaped = np.expand_dims(previous_frame_normalized, axis=0)  # Shape (1, features)

        # Make a prediction
        predicted_frame_normalized = model.predict([audio_input_reshaped, prev_frame_reshaped])

        # Invert scaling to get the predicted frame in original scale
        predicted_frame = frame_scaler.inverse_transform(predicted_frame_normalized)

        # Add the predicted frame to the list
        predicted_frames.append(predicted_frame.flatten())

        # Update the previous frame for the next prediction
        previous_frame_normalized = predicted_frame_normalized.flatten()

        # Optional: Shift the audio input for continuous prediction
        # Assuming new audio information is available, you could shift the audio input to always keep `num_timesteps`.
        # Here, we use the existing sequence as an example.
        if i + 1 < len(audio_input_normalized):
            audio_input_normalized = np.roll(audio_input_normalized, -1, axis=0)  # Shift one time step to the left
            # Optionally, replace the last element with new audio features
            # audio_input_normalized[-1] = new_audio_features_normalized

    # Convert list to ndarray for consistency
    predicted_frames = np.array(predicted_frames)

    return predicted_frames
