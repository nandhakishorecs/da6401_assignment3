import tensorflow as tf
import wandb                                # type: ignore
import yaml                                 # type: ignore

import warnings
warnings.filterwarnings('ignore')

import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #

from data_loader import * 
from network import * 

# Set random seed for reproducibility
tf.random.set_seed(42)


SWEEP_NAME = 'da6401_assignment3_attention_v1'
EXPERIMENT_COUNT = 70

with open("sweep_config.yml", "r") as file:
    sweep_config = yaml.safe_load(file)

sweep_id = wandb.sweep(sweep_config,project=SWEEP_NAME)

def do_sweep(): 

    wandb.init(project = 'da6401_assignment3')
    config = wandb.config 

    wandb.run.name = f'ed_{config.embed_dim}_hd_{config.hidden_dim}_lr_{config.learning_rate:.4f}_opt_{config.optimiser}_e_{config.encoder_cell_type}_d_{config.decoder_cell_type}_eb_{config.encoder_bias}_db_{config.decoder_bias}_ne_{config.num_encoder_layers}_nd_{config.num_decoder_layers}_edo_{config.encoder_dropout_rate:.4f}_ddo_{config.decoder_dropout_rate:.4f}_bs_{config.batch_size}_{config.epochs}_na_{config.num_attention_layers}'

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
    
    # print("Processing training data...")
    train_encoder_input, train_decoder_input, train_decoder_target = processor.process_data(
        train_input, enc_timesteps, train_target, dec_timesteps
    )
    # print("Training data shapes:", train_encoder_input.shape, train_decoder_input.shape, train_decoder_target.shape)
    
    # print("Processing validation data...")
    val_encoder_input, val_decoder_input, val_decoder_target = processor.process_data(
        val_input, enc_timesteps, val_target, dec_timesteps
    )
    # print("Validation data shapes:", val_encoder_input.shape, val_decoder_input.shape, val_decoder_target.shape)
    
    # print("Processing test data...")
    test_encoder_input, test_decoder_input, test_decoder_target = processor.process_data(
        test_input, enc_timesteps, test_target, dec_timesteps
    )
    # print("Test data shapes:", test_encoder_input.shape, test_decoder_input.shape, test_decoder_target.shape)
    
    # Initialize Seq2SeqRNN model with logging enabled
    seq2seq = Seq2SeqRNN(
        src_vocab_size=len(input_char_dec),
        tgt_vocab_size=len(target_char_dec),
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate, 
        optimiser=config.optimiser,
        encoder_cell_type=config.encoder_cell_type,
        decoder_cell_type=config.decoder_cell_type,  # Changed to lstm for consistency
        encoder_bias=config.encoder_bias, 
        decoder_bias=config.decoder_bias, 
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        encoder_dropout_rate=config.encoder_dropout_rate, 
        decoder_dropout_rate=config.decoder_dropout_rate, 
        batch_size=config.batch_size, 
        epochs=config.epochs,
        log=True, 
        use_attention=True, 
        num_attention_layers=config.num_attention_layers
    )
    
    # Train the model
    # print("Starting training...")
    try: 
        history = seq2seq.train(
            encoder_inputs=train_encoder_input,
            decoder_inputs=train_decoder_input,
            decoder_outputs=train_decoder_target,
            validation_data=([val_encoder_input, val_decoder_input], val_decoder_target)
        )
    except Exception as error: 
        print(f'Training failed with error:\r{error}')
        raise
    
    # Evaluate on test data
    # print("Evaluating on test data...")
    _, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    wandb.finish()

if __name__ == '__main__': 
    wandb.agent(sweep_id, function = do_sweep, count = EXPERIMENT_COUNT)