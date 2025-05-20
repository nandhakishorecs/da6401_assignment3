import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #


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
    
    # Evaluate on test data
    print("Evaluating on test data...")
    _, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Generating attention heatmaps...")
    save_path = '/home/ee20d064/GAIL_Machine_Learning/Nandhakishore/RNN/v1/predictions_attention'
    seq2seq.generate_attention_heatmaps(
        encoder_inputs_test=test_encoder_input,
        decoder_inputs_test=test_decoder_input,
        input_char_dec=input_char_dec,
        target_char_dec=target_char_dec,
        num_samples=5,
        save_path=save_path
    )
    
    # Print model summary
    # seq2seq.summary()