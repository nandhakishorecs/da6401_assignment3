import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

from data_loader import *
from network import * 

if __name__ == '__main__':
    # Initialize DataProcessor
    processor = DataProcessor()
    
    # Define file paths
    base_path = "/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/v1/data"
    train_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
    val_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"
    test_data_path = f"{base_path}/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
    
    # Read training, validation, and test data
    train_input, train_target = processor.read_data(train_data_path, characters=True)
    val_input, val_target = processor.read_data(val_data_path, characters=True)
    test_input, test_target = processor.read_data(test_data_path, characters=True)
    
    # Create character encodings
    (
        input_char_enc, input_char_dec,
        target_char_enc, target_char_dec,
        max_encoder_seq_length, max_decoder_seq_length
    ) = processor.encode_decode_characters(
        train_input, train_target,
        val_input, val_target
    )
    
    # Process data
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
    
    # Initialize Seq2SeqRNN model with logging enabled
    seq2seq = Seq2SeqRNN(
        # using best model choosen from sweeps 
        src_vocab_size=len(input_char_dec),
        tgt_vocab_size=len(target_char_dec),
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
    
    # Train the model
    print("Starting training...")
    history = seq2seq.train(
        encoder_inputs=train_encoder_input,
        decoder_inputs=train_decoder_input,
        decoder_outputs=train_decoder_target,
        validation_data=([val_encoder_input, val_decoder_input], val_decoder_target)
    )

    # Save model weights
    # model_save_path = os.getcwd()
    # os.makedirs(model_save_path, exist_ok=True)
    # weights_path = os.path.join(model_save_path, "best_vannila_seq2seq_model.h5")
    # seq2seq.save_weights(weights_path)
    # print(f"Saved model weights to {weights_path}")
    
    # Evaluate on test data
    print("Evaluating on test data...")
    predictions, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # Decode predictions
    predictions, accuracy = seq2seq.evaluate(test_encoder_input, test_decoder_input, test_decoder_target)

    # Decode predictions
    def decode_sequence(pred, char_dec, end_char='\n'):
        decoded = []
        for seq in pred:
            seq_dec = []
            for t in range(seq.shape[0]):
                char_idx = np.argmax(seq[t])
                char = char_dec[char_idx]
                if char == end_char:
                    break
                if char != '\t':  # Skip START_CHAR
                    seq_dec.append(char)
            decoded.append(''.join(seq_dec))
        return decoded

    predicted_sequences = decode_sequence(predictions, target_char_dec)

    # Prepare data for grid (5 data points)
    grid_data = [['Input (English)', 'True Target (Devnagri)', 'Predicted Output']]
    for i in range(5):
        input_str = ''.join(test_input[i])
        true_str = ''.join(test_target[i])
        pred_str = predicted_sequences[i] if i < len(predicted_sequences) else ''
        grid_data.append([input_str, true_str, pred_str])

    # Create matplotlib table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    table = Table(ax, bbox=[0, 0, 1, 1])

    n_rows = len(grid_data)
    n_cols = len(grid_data[0])
    width = 1.0 / n_cols
    height = 1.0 / n_rows

    # Add cells
    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0:
                facecolor = '#2F4F4F'  # Dark slate gray
                text_color = 'white'
                fontweight = 'bold'
            else:
                facecolor = '#E6E6FA' if j < 2 else ('#98FB98' if grid_data[i][j] == grid_data[i][1] else '#FFB6C1')
                text_color = 'black'
                fontweight = 'normal'
            cell = table.add_cell(i, j, width, height, text=grid_data[i][j], loc='center', facecolor=facecolor)
            cell.set_text_props(color=text_color, weight=fontweight, fontsize=12)
            cell.set_edgecolor('black')
            cell.set_linewidth(1)

    # Customize table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Add title and footer
    plt.title('Seq2Seq Model: 5 Test Data Points', fontsize=16, pad=20, color='#483D8B')
    fig.text(0.5, 0.02, f'Prediction Accuracy: {accuracy:.2%}', ha='center', fontsize=12, color='#2F4F4F')

    # Save plot
    plt.savefig('seq2seq_grid.png', bbox_inches='tight', dpi=300)
    plt.close()