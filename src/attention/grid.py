import numpy as np
import pandas as pd
from tabulate import tabulate
import random
from PIL import Image, ImageDraw, ImageFont

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

from data_loader import *
from network import * 

def decode_sequence(prediction, char_dec, max_len):
    """Decode a sequence of token IDs back to characters, assuming char_dec is a list."""
    decoded = ''
    for token in prediction[:max_len]:
        if token == 0:  # Assuming 0 is padding or end token
            break
        try:
            char = char_dec[int(token)] if token < len(char_dec) else ''
        except IndexError:
            char = ''
        decoded += char
    # Clean the decoded text: remove newlines and extra spaces
    return ''.join(decoded.split())

def generate_predictions(seq2seq, test_encoder_input, test_decoder_input):
    print("Generating predictions for test data...")
    test_predictions = seq2seq.model.predict([test_encoder_input, test_decoder_input])
    predicted_ids = np.argmax(test_predictions, axis=-1)
    return predicted_ids

def save_to_csv(latin_refs, devanagari_origs, predicted_texts, csv_path):
    df = pd.DataFrame({
        'Latin Reference': latin_refs,
        'Devanagari Original': devanagari_origs,
        'Prediction': predicted_texts
    })
    df.to_csv(csv_path, index=False, lineterminator='\n')
    print(f"Saved test predictions to {csv_path}")

def save_table_as_txt(table_data, headers, output_path):
    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"Saved table as {output_path}")


def generate_multiple_predictions(seq2seq, test_point_input, test_point_decoder_input, target_char_dec, max_decoder_seq_length, num_predictions=5):
    predictions = []
    accuracies = []
    target_text = decode_sequence(test_point_target[0].argmax(axis=-1), target_char_dec, max_decoder_seq_length)
    
    for _ in range(num_predictions):
        pred_probs = seq2seq.model.predict([test_point_input, test_point_decoder_input])
        pred_ids = []
        for t in range(pred_probs.shape[1]):
            probs = pred_probs[0, t]
            token = np.random.choice(len(probs), p=probs)
            pred_ids.append(token)
        pred_text = decode_sequence(pred_ids, target_char_dec, max_decoder_seq_length)
        predictions.append(pred_text)
        
        correct_chars = sum(1 for a, b in zip(pred_text, target_text) if a == b)
        accuracy = correct_chars / max(len(pred_text), len(target_text), 1)
        accuracies.append(accuracy)
    
    return predictions, accuracies, target_text

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
        src_vocab_size=len(input_char_dec),
        tgt_vocab_size=len(target_char_dec),
        embed_dim=128,
        hidden_dim=512,
        learning_rate=0.0003757444021189912, 
        optimiser='adam',
        encoder_cell_type='lstm',
        decoder_cell_type='rnn',  # Changed to lstm for consistency
        encoder_bias=True, 
        decoder_bias=False, 
        num_encoder_layers=1,
        num_decoder_layers=3,
        encoder_dropout_rate=0.4847177227086345, 
        decoder_dropout_rate=0.2977326987635475, 
        batch_size=64, 
        epochs=1,
        log=True, 
        use_attention=True, 
        num_attention_layers=2
    )
    
    # Train the model
    print("Starting training...")
    history = seq2seq.train(
        encoder_inputs=train_encoder_input,
        decoder_inputs=train_decoder_input,
        decoder_outputs=train_decoder_target,
        validation_data=([val_encoder_input, val_decoder_input], val_decoder_target)
    )

    # Generate predictions
    predicted_ids = generate_predictions(seq2seq, test_encoder_input, test_decoder_input)

    # Decode inputs, targets, and predictions
    latin_refs = [decode_sequence(seq, input_char_dec, max_decoder_seq_length) for seq in test_encoder_input]
    devanagari_origs = [decode_sequence(seq.argmax(axis=-1), target_char_dec, max_decoder_seq_length) for seq in test_decoder_target]
    predicted_texts = [decode_sequence(pred, target_char_dec, max_decoder_seq_length) for pred in predicted_ids]

    # Save to CSV
    save_to_csv(latin_refs, devanagari_origs, predicted_texts, 'test_predictions.csv')

    # Select one random test point
    random_idx = random.randint(0, len(test_input) - 1)
    test_point_input = test_encoder_input[random_idx:random_idx+1]
    test_point_decoder_input = test_decoder_input[random_idx:random_idx+1]
    test_point_target = test_decoder_target[random_idx:random_idx+1]

    # Generate 5 predictions
    print(f"\nGenerating 5 predictions for random test point (index {random_idx})...")
    predictions, accuracies, target_text = generate_multiple_predictions(
        seq2seq, test_point_input, test_point_decoder_input, target_char_dec, max_decoder_seq_length
    )

    # Create table data
    table_data = []
    for i, (pred, acc) in enumerate(zip(predictions, accuracies)):
        row = [str(i + 1), pred, f"{acc:.4f}"]
        if acc == max(accuracies):
            row = [f"**{i + 1}**", f"**{pred}**", f"**{acc:.4f}**"]
        table_data.append(row)

    # Print table to console
    headers = ["Prediction #", "Predicted Text", "Accuracy"]
    print("\nPredictions for random test point:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Print original input and target for reference
    print(f"\nOriginal Latin Input: {latin_refs[random_idx]}")
    print(f"Original Devanagari Target: {devanagari_origs[random_idx]}")

    
    # Save table as TXT
    save_table_as_txt(table_data, headers, 'test_point_predictions.txt')

