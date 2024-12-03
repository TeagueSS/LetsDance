import os
import numpy as np
import datetime
import tensorflow as tf
from keras.src.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import joblib  # For saving and loading scalers



import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(tf.config.list_physical_devices('GPU'))
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


class WindowSizeAdjuster(Callback):
    def __init__(self, initial_window_size, max_window_size, increment, dataset_function, audio_data, frame_data):
        """
        Custom callback to increase the window size during training.
        - initial_window_size: Starting size of the window.
        - max_window_size: Maximum allowable window size.
        - increment: How much to increase the window size by after each adjustment.
        - dataset_function: Function to regenerate the dataset with the new window size.
        - audio_data: Full audio data used to regenerate the dataset.
        - frame_data: Full frame data used to regenerate the dataset.
        """
        super().__init__()
        self.window_size = initial_window_size
        self.max_window_size = max_window_size
        self.increment = increment
        self.dataset_function = dataset_function
        self.audio_data = audio_data
        self.frame_data = frame_data

    def on_epoch_end(self, epoch, logs=None):
        # Adjust window size every 5 epochs (or your desired interval)
        if epoch > 0 and epoch % 5 == 0 and self.window_size < self.max_window_size:
            self.window_size = min(self.window_size + self.increment, self.max_window_size)
            print(f"\nAdjusting window size to: {self.window_size}")

            # Regenerate dataset with new window size
            X_audio, X_prev_frame, y_next_frame, _, _ = self.dataset_function(self.audio_data, self.frame_data,
                                                                              time_steps=self.window_size)

            # Split the new data
            X_audio_train, X_audio_val, X_prev_frame_train, X_prev_frame_val, y_train, y_val = train_test_split(
                X_audio, X_prev_frame, y_next_frame, test_size=0.2, random_state=42
            )

            # Update the training and validation datasets in the model
            self.model.train_data = ([X_audio_train, X_prev_frame_train], y_train)
            self.model.val_data = ([X_audio_val, X_prev_frame_val], y_val)
            print(f"Dataset updated for new window size: {self.window_size}")


# Define paths for loading and saving data and models
input_directory = "/Users/teaguesangster/Desktop/ProcessedEntries"
output_model_path = "/Users/teaguesangster/Desktop/saved_audio_to_frame_model2_with_context.keras"

# Derive the directory from output_model_path
model_directory = os.path.dirname(output_model_path)

# Construct paths for preprocessed data and scalers within the same directory
preprocessed_data_path = os.path.join(model_directory, 'preprocessed_data.npz')
audio_scaler_path = os.path.join(model_directory, 'audio_scaler.save')
frame_scaler_path = os.path.join(model_directory, 'frame_scaler.save')

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

def preprocess_data(audio_data, frame_data, time_steps=200):
    """
    Preprocess audio and frame data to create time-step sequences for training an RNN.
    """
    # Normalizing our Audio data
    audio_scaler = MinMaxScaler()
    audio_data_normalized = audio_scaler.fit_transform(audio_data)

    # Normalize the frame data
    frame_scaler = MinMaxScaler()
    frame_data_normalized = frame_scaler.fit_transform(frame_data)

    # Create sequences of time steps for audio input, previous frame input, and frame output
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

def build_rnn_model(audio_input_shape, frame_input_shape):
    # Audio input layer
    audio_input = Input(shape=audio_input_shape, name='audio_input')

    # Long Term / Short term Neuerons for our audio layer
    # -> Audio data is continious and requires context so long term memory
    # Is important
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
    dense_out = LeakyReLU(negative_slope=0.1)(dense_out)

    # Output layer
    output = Dense(frame_input_shape[0], activation='linear')(dense_out)

    # Build and compile the model
    model = Model(inputs=[audio_input, prev_frame_input], outputs=output)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Return the model
    return model

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

# Functions to save and load preprocessed data and scalers
def save_preprocessed_data(X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler, filename):
    np.savez_compressed(filename, X_audio=X_audio, X_prev_frame=X_prev_frame, y_next_frame=y_next_frame)
    joblib.dump(audio_scaler, 'audio_scaler.save')
    joblib.dump(frame_scaler, 'frame_scaler.save')

def load_preprocessed_data(filename):
    data = np.load(filename)
    X_audio = data['X_audio']
    X_prev_frame = data['X_prev_frame']
    y_next_frame = data['y_next_frame']
    audio_scaler = joblib.load('audio_scaler.save')
    frame_scaler = joblib.load('frame_scaler.save')
    return X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler




# Functions to save and load preprocessed data and scalers
def save_preprocessed_data(X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler, data_path):
    np.savez_compressed(data_path, X_audio=X_audio, X_prev_frame=X_prev_frame, y_next_frame=y_next_frame)
    joblib.dump(audio_scaler, audio_scaler_path)
    joblib.dump(frame_scaler, frame_scaler_path)

def load_preprocessed_data(data_path):
    data = np.load(data_path)
    X_audio = data['X_audio']
    X_prev_frame = data['X_prev_frame']
    y_next_frame = data['y_next_frame']
    audio_scaler = joblib.load(audio_scaler_path)
    frame_scaler = joblib.load(frame_scaler_path)
    return X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler





if __name__ == "__main__":
    # Check if preprocessed data and scalers exist
    if os.path.exists(preprocessed_data_path) and os.path.exists(audio_scaler_path) and os.path.exists(frame_scaler_path):
        # Load preprocessed data
        print("Loading preprocessed data...")
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = load_preprocessed_data(preprocessed_data_path)
    else:
        # Load and combine data
        print("Combining raw data...")
        audio_data, frame_data = combine_npz_files(input_directory)
        # Preprocess the data to create sequences for the RNN
        print("Preprocessing data...")
        time_steps = 60  # Ensure this matches the value in preprocess_data
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(
            audio_data, frame_data, time_steps
        )
        # Save preprocessed data
        print("Saving preprocessed data...")
        save_preprocessed_data(X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler, preprocessed_data_path)

    # Split the data
    print("Splitting data into training and validation sets...")
    X_audio_train, X_audio_val, X_prev_frame_train, X_prev_frame_val, y_train, y_val = train_test_split(
        X_audio, X_prev_frame, y_next_frame, test_size=0.2, random_state=42
    )

    # Build the RNN model
    audio_input_shape = (X_audio.shape[1], X_audio.shape[2])  # Shape is (time_steps, audio_features)
    frame_input_shape = (X_prev_frame.shape[1],)  # Shape is (number of frame features,)

    # Check if the model exists
    if os.path.exists(output_model_path):
        # Load the saved model
        print("Loading saved model...")
        model = load_model(output_model_path)
    else:
        # Build the RNN model
        print("Building a new model...")
        model = build_rnn_model(audio_input_shape, frame_input_shape)

    # Decide whether to train the model
    train_model = True  # Set to False if you want to skip training when model exists

    if train_model or not os.path.exists(output_model_path):
        # Train the RNN model with callbacks
        print("Training_Methods the model...")
        epochs = 50
        batch_size = 32
        history = model.fit(
            [X_audio_train, X_prev_frame_train], y_train,
            validation_data=([X_audio_val, X_prev_frame_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("Skipping training as the model already exists.")

    # Evaluate the model and make predictions
    print("Evaluating the model and making predictions...")
    X_val = [X_audio_val, X_prev_frame_val]
    evaluate_and_predict(model, X_val, y_val, frame_scaler)



# Prediction Function
def predict_body_mappings(model_path, audio_scaler, frame_scaler, audio_input_array, initial_frame, num_predictions):
    """
    Predict body mappings for given audio input sequences using a pre-trained model.
    """
    # If model_path is a path, load the model
    if isinstance(model_path, str):
        model = load_model(model_path)
    else:
        # Assume model_path is already a loaded model
        model = model_path



    # Normalize the audio input using the scaler (Our encoder)
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
        if i + 1 < len(audio_input_normalized):
            audio_input_normalized = np.roll(audio_input_normalized, -1, axis=0)  # Shift one time step to the left

    # Convert list to ndarray for consistency
    predicted_frames = np.array(predicted_frames)

    return predicted_frames
