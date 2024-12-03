import unittest
import numpy as np
import os
from tensorflow.keras.models import load_model
# Import your functions here
from Training_Methods.Training import combine_npz_files, preprocess_data, build_rnn_model, train_rnn_model, evaluate_and_predict, predict_body_mappings
# Create mock data for testing
import numpy as np
import os


def create_mock_npz_files(test_directory, num_files=2, num_samples=20, audio_features=5, frame_features=3):
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    for i in range(num_files):
        audio_data = np.random.rand(num_samples, audio_features)
        frame_data = np.random.rand(num_samples, frame_features)
        file_path = os.path.join(test_directory, f"test_data_{i}.npz")
        np.savez(file_path, audio=audio_data, frame=frame_data)
    print(f"Created {num_files} mock .npz files in {test_directory}")


# Create mock data in the 'test_data' directory
create_mock_npz_files('test_data')


class TestRNNPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Paths for test data and model
        cls.test_data_dir = 'test_data'
        cls.model_save_path = 'test_model.h5'

        # Create mock data
        create_mock_npz_files(cls.test_data_dir, num_files=2, num_samples=20, audio_features=5, frame_features=3)

    @classmethod
    def tearDownClass(cls):
        # Clean up created files
        import shutil
        shutil.rmtree(cls.test_data_dir)
        if os.path.exists(cls.model_save_path):
            os.remove(cls.model_save_path)

    def test_combine_npz_files(self):
        # Test combining .npz files
        audio_data, frame_data = combine_npz_files(self.test_data_dir)
        self.assertEqual(audio_data.shape, (40, 5))
        self.assertEqual(frame_data.shape, (40, 3))
        print("combine_npz_files passed.")

    def test_preprocess_data(self):
        # Use data from combine_npz_files
        audio_data, frame_data = combine_npz_files(self.test_data_dir)
        time_steps = 5
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(audio_data, frame_data, time_steps)

        # Check shapes
        expected_samples = audio_data.shape[0] - time_steps - 1
        self.assertEqual(X_audio.shape, (expected_samples, time_steps, audio_data.shape[1]))
        self.assertEqual(X_prev_frame.shape, (expected_samples, frame_data.shape[1]))
        self.assertEqual(y_next_frame.shape, (expected_samples, frame_data.shape[1]))
        print("preprocess_data passed.")

    def test_build_rnn_model(self):
        # Build model and check its structure
        time_steps = 5
        audio_features = 5
        frame_features = 3
        audio_input_shape = (time_steps, audio_features)
        frame_input_shape = (frame_features,)

        model = build_rnn_model(audio_input_shape, frame_input_shape)
        self.assertIsNotNone(model)
        self.assertEqual(len(model.inputs), 2)
        self.assertEqual(model.output_shape, (None, frame_features))
        print("build_rnn_model passed.")

    def test_train_rnn_model(self):
        # Prepare data
        audio_data, frame_data = combine_npz_files(self.test_data_dir)
        time_steps = 5
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(audio_data, frame_data, time_steps)
        audio_input_shape = (X_audio.shape[1], X_audio.shape[2])
        frame_input_shape = (X_prev_frame.shape[1],)

        # Build model
        model = build_rnn_model(audio_input_shape, frame_input_shape)

        # Train model
        history, X_val, y_val = train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, self.model_save_path, epochs=1, batch_size=4)

        # Check if the model was saved
        self.assertTrue(os.path.exists(self.model_save_path))
        print("train_rnn_model passed.")

    def test_evaluate_and_predict(self):
        # Prepare data
        audio_data, frame_data = combine_npz_files(self.test_data_dir)
        time_steps = 5
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(audio_data, frame_data, time_steps)
        audio_input_shape = (X_audio.shape[1], X_audio.shape[2])
        frame_input_shape = (X_prev_frame.shape[1],)

        # Build and train model
        model = build_rnn_model(audio_input_shape, frame_input_shape)
        history, X_val, y_val = train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, self.model_save_path, epochs=1, batch_size=4)

        # Evaluate and predict
        evaluate_and_predict(model, X_val, y_val, frame_scaler, num_predictions=2)
        print("evaluate_and_predict passed.")

    def test_predict_body_mappings(self):
        # Prepare data
        audio_data, frame_data = combine_npz_files(self.test_data_dir)
        time_steps = 5
        X_audio, X_prev_frame, y_next_frame, audio_scaler, frame_scaler = preprocess_data(audio_data, frame_data, time_steps)
        audio_input_shape = (X_audio.shape[1], X_audio.shape[2])
        frame_input_shape = (X_prev_frame.shape[1],)

        # Build and train model
        model = build_rnn_model(audio_input_shape, frame_input_shape)
        history, _, _ = train_rnn_model(model, X_audio, X_prev_frame, y_next_frame, self.model_save_path, epochs=1, batch_size=4)

        # Prepare input for prediction
        audio_input_array = X_audio[0]  # Use first sequence
        initial_frame = X_prev_frame[0]  # Use corresponding previous frame
        num_predictions = 2

        predicted_frames = predict_body_mappings(
            self.model_save_path, audio_scaler, frame_scaler, audio_input_array, initial_frame, num_predictions
        )

        self.assertEqual(predicted_frames.shape, (num_predictions, frame_data.shape[1]))
        print("predict_body_mappings passed.")

if __name__ == '__main__':
    unittest.main()
