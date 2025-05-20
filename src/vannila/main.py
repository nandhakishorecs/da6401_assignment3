import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

import argparse
from data_loader import *
from network import * 

def get_args():
    parser = argparse.ArgumentParser(description = '\033[92m' + '\nTrain a CNN based model with iNaturalist dataset\n' + '\033[0m')
    
    # model architecture 
    parser.add_argument('-ed', '--embed_dim', type = int, default = 128, choices = [128, 256, 512], help = 'Embedding Dimensions')
    parser.add_argument('-hd', '--hidden_simension', type = int, default = 128, choices=[128, 256, 512], help = 'Hidden Dimensions')
    
    # layer parameters 
    parser.add_argument('-eb', '--encoder_bias', type = bool, default = True, help = 'Bias for encoder layer(s)')
    parser.add_argument('-db', '--decoder_bias', type = bool, default = True, help = 'Bias for decoder layer(s)')
    parser.add_argument('-bs', '--batch_size', type = int, default = 16, choices = [16, 32, 64], help = 'Batch size')
    parser.add_argument('-ne', '--n_encoder_layer', type = int, default = 1, choices = [1, 2, 3, 4], help = 'Number of layers in Encoder')
    parser.add_argument('-nd', '--n_decoder_layer', type = int, default = 1, choices = [1, 2, 3, 4], help = 'Number of layers in Decoder')
    parser.add_argument('-et', '--encoder_type', type = str, default = 'rnn', choices = ['lstm', 'rnn', 'gru'], help = 'Encoder cell type')
    parser.add_argument('-dt', '--decoder_type', type = str, default = 'rnn', choices = ['lstm', 'rnn', 'gru'], help = 'Decoder cell type')
    parser.add_argument('-edo', '--encoder_dropout', type = float, default = 0.1, help = 'Encoder dropout')
    parser.add_argument('-ddo', '--decoder_dropout', type = float, default = 0.1, help = 'Decoder dropout')

    # optimiser parameters
    parser.add_argument('-lr', '--learning_rate', type = float, default = 1e-4, help = 'Learning rate')
    parser.add_argument('-o', '--optimiser', type = str, default = 'adam', choices=['sgd', 'adam'], help = 'Optimiser')
    parser.add_argument('-e', '--epochs', type = int, default = 1, help = 'Number of epochs')
    
    # wandb configuration
    parser.add_argument('-log', '--log', type = bool, default = False, help = 'Use wandb')
    parser.add_argument('-wp', '--wandb_project', type = str, default = 'da6401_assignment2', help = 'Use wandb')
    parser.add_argument('-we', '--wand_entity', type = str, default = 'trial1', help = 'Use wandb')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

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
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        optimiser=args.optimiser,
        encoder_cell_type=args.encoder_type,
        decoder_cell_type=args.decoder_type,
        encoder_bias=args.encoder_bias,
        decoder_bias=args.decoder_bias,
        num_encoder_layers=args.n_encoder_layer,
        num_decoder_layers=args.n_decoder_layer,
        encoder_dropout_rate=args.encoder_dropout,
        decoder_dropout_rate=args.decoder_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        log=args.log
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
    _, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Print model summary
    seq2seq.summary()