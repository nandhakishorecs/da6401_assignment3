import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Embedding, Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import wandb
    from wandb.keras import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
class DataProcessor:
    """A class to handle data processing for sequence-to-sequence models."""
    
    BLANK_CHAR = ' '
    START_CHAR = '\t'
    END_CHAR = '\n'
    
    def __init__(self):
        self.input_char_enc = {}
        self.input_char_dec = []
        self.target_char_enc = {}
        self.target_char_dec = []
        self.max_encoder_seq_length = 1
        self.max_decoder_seq_length = 1
    
    def read_data(self, data_path, characters=False):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = [line.split("\t") for line in f.read().split("\n") if line != '']
        
        input_data = [val[1] for val in lines]
        target_data = [val[0] for val in lines]
        
        if characters:
            input_data = [list(inp_str) for inp_str in input_data]
            target_data = [list(tar_str) for tar_str in target_data]
            
        return input_data, target_data
    
    def process_data(self, input_data, enc_timesteps, target_data=None, dec_timesteps=None):
        # Validate inputs
        if target_data is not None and dec_timesteps is not None:
            for string in target_data:
                if not all(ch in self.target_char_enc for ch in string):
                    raise ValueError(f"Found character in target_data not in target_char_enc: {string}")
                if len(string) + 2 > dec_timesteps:
                    raise ValueError(f"Sequence length {len(string) + 2} exceeds dec_timesteps {dec_timesteps} for string: {string}")
        
        # Process encoder input
        encoder_input = np.array([
            [self.input_char_enc[ch] for ch in string] + 
            [self.input_char_enc[self.BLANK_CHAR]] * (enc_timesteps - len(string))
            for string in input_data
        ])
        
        decoder_input, decoder_target = None, None
        if target_data is not None and dec_timesteps is not None:
            # Process decoder input with fixed length
            decoder_input = []
            for string in target_data:
                seq = [self.target_char_enc[self.START_CHAR]] + \
                      [self.target_char_enc[ch] for ch in string] + \
                      [self.target_char_enc[self.END_CHAR]]
                seq += [self.target_char_enc[self.BLANK_CHAR]] * (dec_timesteps - len(seq))
                decoder_input.append(seq[:dec_timesteps])  # Truncate to dec_timesteps
            decoder_input = np.array(decoder_input)
            
            # Process decoder target (one-hot encoded)
            decoder_target = np.zeros(
                (decoder_input.shape[0], dec_timesteps, len(self.target_char_enc)),
                dtype='float32'
            )
            
            for i in range(decoder_input.shape[0]):
                for t, char_ind in enumerate(decoder_input[i]):
                    if t > 0:
                        decoder_target[i, t-1, char_ind] = 1.0
                if t < dec_timesteps - 1:
                    decoder_target[i, t:, self.target_char_enc[self.BLANK_CHAR]] = 1.0
        
        return encoder_input, decoder_input, decoder_target

    def encode_decode_characters(self, train_input, train_target, val_input, val_target):
        for string in train_input + val_input:
            self.max_encoder_seq_length = max(self.max_encoder_seq_length, len(string))
            for char in string:
                if char not in self.input_char_enc:
                    self.input_char_enc[char] = len(self.input_char_dec)
                    self.input_char_dec.append(char)
        if self.BLANK_CHAR not in self.input_char_enc:
            self.input_char_enc[self.BLANK_CHAR] = len(self.input_char_dec)
            self.input_char_dec.append(self.BLANK_CHAR)
        
        self.target_char_enc[self.START_CHAR] = len(self.target_char_dec)
        self.target_char_dec.append(self.START_CHAR)
        
        for string in train_target + val_target:
            self.max_decoder_seq_length = max(self.max_decoder_seq_length, len(string) + 2)  # +2 for START_CHAR and END_CHAR
            for char in string:
                if char not in self.target_char_enc:
                    self.target_char_enc[char] = len(self.target_char_dec)
                    self.target_char_dec.append(char)
        
        self.target_char_enc[self.END_CHAR] = len(self.target_char_dec)
        self.target_char_dec.append(self.END_CHAR)
        
        if self.BLANK_CHAR not in self.target_char_enc:
            self.target_char_enc[self.BLANK_CHAR] = len(self.target_char_dec)
            self.target_char_dec.append(self.BLANK_CHAR)
        
        # print("Number of training samples:", len(train_input))
        # print("Number of validation samples:", len(val_input))
        # print("Number of unique input tokens:", len(self.input_char_dec))
        # print("Number of unique output tokens:", len(self.target_char_dec))
        # print("Max sequence length for inputs:", self.max_encoder_seq_length)
        # print("Max sequence length for outputs:", self.max_decoder_seq_length)
        
        return (
            self.input_char_enc, self.input_char_dec,
            self.target_char_enc, self.target_char_dec,
            self.max_encoder_seq_length, self.max_decoder_seq_length
        )

class Seq2SeqRNN:
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        embed_dim=128,
        hidden_dim=256,
        learning_rate=0.001,
        encoder_cell_type='lstm',
        decoder_cell_type='lstm',
        encoder_bias=True,
        decoder_bias=False,
        optimiser='adam',
        num_encoder_layers=1,
        num_decoder_layers=1,
        encoder_dropout_rate=0.1,
        decoder_dropout_rate=0.1,
        batch_size=64,
        epochs=10,
        log=False
    ):
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.encoder_cell_type = encoder_cell_type.lower()
        self.decoder_cell_type = decoder_cell_type.lower()
        self.optimiser = optimiser.lower()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_dropout_rate = encoder_dropout_rate
        self.decoder_dropout_rate = decoder_dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.encoder_bias = encoder_bias
        self.decoder_bias = decoder_bias
        self.log = log
        self.model = None
        self._validate_params()

    def _validate_params(self):
        if self.encoder_cell_type not in ['rnn', 'lstm', 'gru'] or self.decoder_cell_type not in ['rnn', 'lstm', 'gru']:
            raise ValueError("cell_type must be 'rnn', 'lstm', or 'gru'")
        if self.optimiser not in ['adam', 'sgd']:
            raise ValueError("optimiser must be 'adam' or 'sgd'")
        if self.src_vocab_size < 1 or self.tgt_vocab_size < 1:
            raise ValueError("Vocabulary sizes must be positive integers")
        if self.embed_dim < 1 or self.hidden_dim < 1:
            raise ValueError("Embedding and hidden dimensions must be positive integers")
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError("Number of layers must be at least 1")
        if self.learning_rate < 0:
            raise ValueError("learning rate must be positive")
        if not isinstance(self.log, bool):
            raise ValueError("log must be a boolean")

    def _get_rnn_cell(self, cell_type='encoder'):
        if cell_type == 'encoder':
            if self.encoder_cell_type == 'rnn':
                return SimpleRNN
            elif self.encoder_cell_type == 'lstm':
                return LSTM
            elif self.encoder_cell_type == 'gru':
                return GRU
        elif cell_type == 'decoder':
            if self.decoder_cell_type == 'rnn':
                return SimpleRNN
            elif self.decoder_cell_type == 'lstm':
                return LSTM
            elif self.decoder_cell_type == 'gru':
                return GRU

    def build_model(self):
        encoder_rnn_cell = self._get_rnn_cell(cell_type='encoder')
        decoder_rnn_cell = self._get_rnn_cell(cell_type='decoder')

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.src_vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)

        encoder_outputs = encoder_embedding
        encoder_states = []
        for i in range(self.num_encoder_layers):
            return_sequences = True if i < self.num_encoder_layers - 1 else False
            encoder_layer = encoder_rnn_cell(
                self.hidden_dim,
                return_sequences=return_sequences,
                return_state=True,
                dropout=self.encoder_dropout_rate,
                use_bias=self.encoder_bias,
                name=f'encoder_{self.encoder_cell_type}_{i+1}'
            )
            encoder_outputs = encoder_layer(encoder_outputs)

            if self.encoder_cell_type == 'lstm':
                encoder_outputs, state_h, state_c = encoder_outputs
                encoder_states = [state_h, state_c]
            else:
                encoder_outputs, state_h = encoder_outputs
                encoder_states = [state_h]

        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            self.tgt_vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)

        decoder_outputs = decoder_embedding
        for i in range(self.num_decoder_layers):
            decoder_layer = decoder_rnn_cell(
                self.hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.decoder_dropout_rate,
                use_bias=self.decoder_bias,
                name=f'decoder_{self.decoder_cell_type}_{i+1}'
            )
            if self.decoder_cell_type == 'lstm':
                initial_state = encoder_states if self.encoder_cell_type == 'lstm' else [encoder_states[0], encoder_states[0]]
            else:
                initial_state = encoder_states[0] if self.encoder_cell_type == 'lstm' else encoder_states

            decoder_outputs = decoder_layer(decoder_outputs, initial_state=initial_state)

            if self.decoder_cell_type == 'lstm':
                decoder_outputs, state_h, state_c = decoder_outputs
                encoder_states = [state_h, state_c]
            else:
                decoder_outputs, state_h = decoder_outputs
                encoder_states = [state_h]

        decoder_dense = Dense(self.tgt_vocab_size, activation='softmax', name='decoder_output')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if self.optimiser == 'adam':
            optimizer = Adam(self.learning_rate)
        elif self.optimiser == 'sgd':
            optimizer = SGD(self.learning_rate)
        else:
            raise ValueError("optimiser must be 'adam' or 'sgd'")

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return self.model

    def train(self, encoder_inputs, decoder_inputs, decoder_outputs, validation_data=None, save_model=False, save_model_path=None):
        if self.model is None:
            self.build_model()

        callbacks = []
        if self.log:
            if not WANDB_AVAILABLE:
                print("Warning: wandb is not installed. Logging to Weights & Biases will be disabled.")
            else:
                wandb.init(project="seq2seq_rnn", config={
                    "src_vocab_size": self.src_vocab_size,
                    "tgt_vocab_size": self.tgt_vocab_size,
                    "embed_dim": self.embed_dim,
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "encoder_cell_type": self.encoder_cell_type,
                    "decoder_cell_type": self.decoder_cell_type,
                    "num_encoder_layers": self.num_encoder_layers,
                    "num_decoder_layers": self.num_decoder_layers,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                })
                callbacks.append(WandbCallback(
                    monitor='val_loss',
                    log_weights=False,
                    log_gradients=False,
                    save_model=False
                ))

        history = self.model.fit(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks
        )

        if self.log and WANDB_AVAILABLE:
            for epoch in range(self.epochs):
                metrics = {
                    'epoch': epoch + 1,
                    'loss': history.history['loss'][epoch],
                    'accuracy': history.history['accuracy'][epoch]
                }
                if validation_data is not None:
                    metrics.update({
                        'val_loss': history.history.get('val_loss', [0])[epoch],
                        'val_accuracy': history.history.get('val_accuracy', [0])[epoch]
                    })
                wandb.log(metrics, step=epoch + 1)
            wandb.finish()

        if save_model and save_model_path is not None:
            os.makedirs(save_model_path, exist_ok=True)
            weights_path = os.path.join(save_model_path, "best_vannila_seq2seq_model.h5")
            self.model.save_weights(weights_path)
            print(f"Saved model weights to {weights_path}")

        return history

    def evaluate(self, encoder_inputs, decoder_inputs, decoder_outputs):
        if self.model is None:
            self.build_model()
        metrics = self.model.evaluate(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            verbose=1
        )
        predictions = self.model.predict([encoder_inputs, decoder_inputs])
        return predictions, metrics[1]

    def summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()

    def build_model_for_visualization(self):
        encoder_rnn_cell = self._get_rnn_cell(cell_type='encoder')
        decoder_rnn_cell = self._get_rnn_cell(cell_type='decoder')

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(
            self.src_vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='encoder_embedding'
        )(encoder_inputs)

        encoder_outputs = encoder_embedding
        encoder_states = []
        for i in range(self.num_encoder_layers):
            encoder_layer = encoder_rnn_cell(
                self.hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.encoder_dropout_rate,
                use_bias=self.encoder_bias,
                name=f'encoder_{self.encoder_cell_type}_{i+1}'
            )
            encoder_outputs = encoder_layer(encoder_outputs)

            if self.encoder_cell_type == 'lstm':
                encoder_outputs, state_h, state_c = encoder_outputs
                encoder_states = [state_h, state_c]
            else:
                encoder_outputs, state_h = encoder_outputs
                encoder_states = [state_h]

        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(
            self.tgt_vocab_size,
            self.embed_dim,
            mask_zero=True,
            name='decoder_embedding'
        )(decoder_inputs)

        decoder_outputs = decoder_embedding
        for i in range(self.num_decoder_layers):
            decoder_layer = decoder_rnn_cell(
                self.hidden_dim,
                return_sequences=True,
                return_state=True,
                dropout=self.decoder_dropout_rate,
                use_bias=self.decoder_bias,
                name=f'decoder_{self.decoder_cell_type}_{i+1}'
            )
            initial_state = encoder_states if self.decoder_cell_type == 'lstm' else encoder_states[0]
            decoder_outputs = decoder_layer(decoder_outputs, initial_state=initial_state)

            if self.decoder_cell_type == 'lstm':
                decoder_outputs, state_h, state_c = decoder_outputs
                encoder_states = [state_h, state_c]
            else:
                decoder_outputs, state_h = decoder_outputs
                encoder_states = [state_h]

        decoder_dense = Dense(self.tgt_vocab_size, activation='softmax', name='decoder_output')
        decoder_outputs = decoder_dense(decoder_outputs)

        vis_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if self.model is not None:
            vis_model.set_weights(self.model.get_weights())

        optimizer = Adam(self.learning_rate) if self.optimiser == 'adam' else SGD(self.learning_rate)
        vis_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return vis_model

    def get_lstm_activations(self, encoder_inputs, decoder_inputs):
        """
        Extract LSTM activations using the visualization model to ensure return_sequences=True.
        Returns activations for each LSTM layer in encoder and decoder.
        """
        vis_model = self.build_model_for_visualization()

        activation_outputs = []
        for i in range(self.num_encoder_layers):
            if self.encoder_cell_type == 'lstm':
                activation_outputs.append(vis_model.get_layer(f'encoder_lstm_{i+1}').output[0])  # Hidden state
                activation_outputs.append(vis_model.get_layer(f'encoder_lstm_{i+1}').output[1])  # Cell state
            else:
                activation_outputs.append(vis_model.get_layer(f'encoder_{self.encoder_cell_type}_{i+1}').output[0])  # Hidden state only

        for i in range(self.num_decoder_layers):
            if self.decoder_cell_type == 'lstm':
                activation_outputs.append(vis_model.get_layer(f'decoder_lstm_{i+1}').output[0])  # Hidden state
                activation_outputs.append(vis_model.get_layer(f'decoder_lstm_{i+1}').output[1])  # Cell state
            else:
                activation_outputs.append(vis_model.get_layer(f'decoder_{self.decoder_cell_type}_{i+1}').output[0])  # Hidden state only

        activation_model = Model(inputs=vis_model.inputs, outputs=activation_outputs)
        activations = activation_model.predict([encoder_inputs, decoder_inputs])

        return activations
    def visualize_predictions(self, encoder_inputs, decoder_inputs, target_char_dec, save_path="predictions"):
        """
        Visualize the model's predictions for two random test points, highlighting the output in green.
        
        Args:
            encoder_inputs: Test encoder inputs (numpy array).
            decoder_inputs: Test decoder inputs (numpy array).
            target_char_dec: List mapping indices to target characters (from DataProcessor).
            save_path: Unused parameter (kept for compatibility).
        """
        # Select two random test points
        num_samples = encoder_inputs.shape[0]
        if num_samples < 2:
            print("Error: Test dataset has fewer than 2 samples.")
            return
        
        # Randomly select two indices
        selected_indices = np.random.choice(num_samples, size=20, replace=True)
        
        # Extract the corresponding inputs for the selected indices
        selected_encoder_inputs = encoder_inputs[selected_indices]
        selected_decoder_inputs = decoder_inputs[selected_indices]
        
        # Generate predictions using the model
        predictions = self.model.predict([selected_encoder_inputs, selected_decoder_inputs])
        
        # ANSI escape codes for green text and reset
        GREEN = "\033[32m"
        RESET = "\033[0m"
        
        # Process predictions for each selected sample
        for idx, sample_idx in enumerate(selected_indices):
            print(f"\n## Test Sample {sample_idx + 1}\n")
            
            # Convert predictions to characters
            predicted_sequence = predictions[idx]  # Shape: [dec_timesteps, tgt_vocab_size]
            predicted_indices = np.argmax(predicted_sequence, axis=-1)  # Get the most likely character index at each time step
            
            # Decode the predicted indices into a character sequence
            predicted_chars = []
            for char_idx in predicted_indices:
                if char_idx < len(target_char_dec):
                    char = target_char_dec[char_idx]
                    # Stop at END_CHAR if present
                    if char == '\n':  # END_CHAR as defined in DataProcessor
                        break
                    predicted_chars.append(char)
                else:
                    predicted_chars.append('?')  # Fallback for invalid indices
            
            # Remove START_CHAR if present
            if predicted_chars and predicted_chars[0] == '\t':  # START_CHAR as defined in DataProcessor
                predicted_chars = predicted_chars[1:]
            
            # Join the characters into a string and highlight each character in green
            predicted_text = "".join(predicted_chars)
            colored_text = "".join(f"{GREEN}{char}{RESET}" for char in predicted_text)
            
            print(f"Predicted Output: {colored_text}\n")
        
        print("Predictions for two random test samples printed to console.")

if __name__ == '__main__':
    try:
        os.remove(os.path.expanduser('~/.cache/matplotlib/fontlist-v330.json'))
        print("Cleared Matplotlib font cache.")
    except:
        pass

    processor = DataProcessor()
    base_path = "/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/v1/data"
    train_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    val_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    test_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"

    train_input, train_target = processor.read_data(train_data_path, characters=True)
    val_input, val_target = processor.read_data(val_data_path, characters=True)
    test_input, test_target = processor.read_data(test_data_path, characters=True)

    (
        input_char_enc, input_char_dec,
        target_char_enc, target_char_dec,
        max_encoder_seq_length, max_decoder_seq_length
    ) = processor.encode_decode_characters(
        train_input, train_target,
        val_input, val_target
    )

    enc_timesteps = max_encoder_seq_length
    dec_timesteps = max_decoder_seq_length

    print("Processing training data...")
    train_encoder_input, train_decoder_input, train_decoder_target = processor.process_data(
        train_input, enc_timesteps, train_target, dec_timesteps
    )
    print("Training data shapes:", train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)

    print("Processing validation data...")
    val_encoder_input, val_decoder_input, val_decoder_target = processor.process_data(
        val_input, enc_timesteps, val_target, dec_timesteps
    )
    print("Validation data shapes:", val_encoder_input.shape, val_decoder_input.shape, val_decoder_target.shape)

    print("Processing test data...")
    test_encoder_input, test_decoder_input, test_decoder_target = processor.process_data(
        test_input, enc_timesteps, test_target, dec_timesteps
    )
    print("Test data shapes:", test_encoder_input.shape, test_decoder_input.shape, test_decoder_target.shape)

    seq2seq = Seq2SeqRNN(
        src_vocab_size=len(input_char_enc),
        tgt_vocab_size=len(target_char_enc),
        embed_dim=512,
        hidden_dim=512,
        learning_rate=0.00012180528531698948,
        optimiser='adam',
        encoder_cell_type='lstm',
        decoder_cell_type='rnn',
        encoder_bias=True,
        decoder_bias=False,
        num_encoder_layers=2,
        num_decoder_layers=3,
        encoder_dropout_rate=0.11029084967332052,
        decoder_dropout_rate=0.28297487009133887,
        batch_size=64,
        epochs=1,
        log=True
    )

    print("Starting training...")
    history = seq2seq.train(
        encoder_inputs=train_encoder_input,
        decoder_inputs=train_decoder_input,
        decoder_outputs=train_decoder_target,
        validation_data=([val_encoder_input, val_decoder_input], val_decoder_target)
    )

    print("Evaluating on test data...")
    _, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Visualizing activations for test data...")
    # print("Visualizing predictions for two random test points...")
    subset_size = min(2, test_encoder_input.shape[0])
    seq2seq.visualize_predictions(  # Changed from visualize_activations to visualize_predictions
        test_encoder_input[:subset_size],
        test_decoder_input[:subset_size],
        processor.target_char_dec,  # Pass target_char_dec from DataProcessor
        save_path="predictions_test"
)