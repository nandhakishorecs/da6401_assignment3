import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'data')))

import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont


from data_loader import *
from network import * 

def create_prediction_table_png(predictions, inputs, references, output_path, font_path=None):
    """Create a PNG table of predictions with green/red highlights for matches/mismatches."""
    # Default font (system-dependent, prefer Noto Sans Devanagari for Devanagari support)
    if font_path is None:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"  # Adjust if needed
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        print("Warning: Devanagari font not found. Using default font (may not render Devanagari correctly).")
        font = ImageFont.load_default()

    # Table dimensions
    cell_width = 200
    cell_height = 40
    header_height = 50
    num_rows = len(predictions) + 1  # Include header
    img_width = cell_width * 3
    img_height = header_height + cell_height * num_rows

    # Create image
    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    # Draw headers
    headers = ["Input (Latin)", "Reference (Devanagari)", "Prediction (Devanagari)"]
    draw.rectangle((0, 0, img_width, header_height), fill="#D3D3D3")  # Light gray header
    for i, header in enumerate(headers):
        draw.text((i * cell_width + 10, 10), header, font=font, fill="black")

    # Draw rows
    for row, (inp, ref, pred) in enumerate(zip(inputs, references, predictions)):
        y = header_height + row * cell_height
        # Input and Reference columns (white background)
        draw.rectangle((0, y, cell_width, y + cell_height), fill="white")
        draw.rectangle((cell_width, y, 2 * cell_width, y + cell_height), fill="white")
        # Prediction column (green for match, red for mismatch)
        is_match = pred == ref
        pred_color = "#00FF00" if is_match else "#FF0000"  # Green or red
        draw.rectangle((2 * cell_width, y, 3 * cell_width, y + cell_height), fill=pred_color)

        # Draw text
        draw.text((10, y + 10), inp, font=font, fill="black")
        draw.text((cell_width + 10, y + 10), ref, font=font, fill="black")
        draw.text((2 * cell_width + 10, y + 10), pred, font=font, fill="black")

        # Draw grid lines
        draw.line((0, y, img_width, y), fill="black")
        for x in range(0, img_width + 1, cell_width):
            draw.line((x, header_height, x, img_height), fill="black")

    # Draw final lines
    draw.line((0, header_height, img_width, header_height), fill="black")
    draw.line((0, img_height, img_width, img_height), fill="black")

    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Saved prediction table to {output_path}")


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
    
    # Initialize Seq2SeqRNN model with assumed best hyperparameters
    seq2seq = Seq2SeqRNN(
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
    
    seq2seq.train(
        encoder_inputs=train_encoder_input,
        decoder_inputs=train_decoder_input,
        decoder_outputs=train_decoder_target,
        validation_data=([val_encoder_input, val_decoder_input], val_decoder_target),
    )
    
    # Load the best model from W&B or local path
    # Replace with your W&B run path or local model path
    # model_path = f"{os.getcwd()}/best_vannila_seq2seq_model.h5"  # Example: "wandb/run-20250518-xxxx/model-best.h5"
    # seq2seq.build_model()
    # seq2seq.model.load_weights(model_path)
    # print("Loaded best model from:", model_path)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_predictions, test_accuracy = seq2seq.evaluate(
        encoder_inputs=test_encoder_input,
        decoder_inputs=test_decoder_input,
        decoder_outputs=test_decoder_target
    )
    
    print(f"Test Accuracy (Exact Match): {test_accuracy:.4f}")
    
    # Print top 5 predictions in Devanagari
    print("\nTop 5 Test Predictions (Devanagari):")
    for i in range(min(5, len(test_predictions))):
        pred = test_predictions[i]
        ref = ''.join(test_target[i].tolist() if isinstance(test_target[i], np.ndarray) else test_target[i])
        print(f"Prediction {i+1}: {pred} (Reference: {ref})")
    
    # Save all predictions to a file
    os.makedirs("predictions_vanilla", exist_ok=True)
    with open("predictions_vanilla/test_predictions.tsv", "w", encoding="utf-8") as f:
        f.write("Input\tReference\tPrediction\n")
        for inp, ref, pred in zip(test_input, test_target, test_predictions):
            inp_str = ''.join(inp.tolist() if isinstance(inp, np.ndarray) else inp)
            ref_str = ''.join(ref.tolist() if isinstance(ref, np.ndarray) else ref)
            f.write(f"{inp_str}\t{ref_str}\t{pred}\n")
    print("Saved predictions to predictions_vanilla/test_predictions.tsv")
    
    # Prepare top 5 for PNG and text grid
    top_inputs = [
        ''.join(test_input[i].tolist() if isinstance(test_input[i], np.ndarray) else test_input[i])
        for i in range(min(5, len(test_input)))
    ]
    top_references = [
        ''.join(test_target[i].tolist() if isinstance(test_target[i], np.ndarray) else test_target[i])
        for i in range(min(5, len(test_target)))
    ]
    top_predictions = test_predictions[:5]
    
    # Create PNG with highlights
    create_prediction_table_png(
        top_predictions,
        top_inputs,
        top_references,
        "predictions_vanilla/top_5_predictions.png"
    )