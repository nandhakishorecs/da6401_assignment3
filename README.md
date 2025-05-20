# DA6401 Introduction to Deep Learning - Assignment 3
This repository contains all the code for Assignment 3 for the Introduction to Deep Learning Course (DA6401) offered at Wadhwani School of Data Science and AI, Indian Institute of Technology Madras. 

**Course Instructor**: Prof. Mitesh Khapra <br>
**Author**: Nandhakishore C S <br>
**Roll Number**: DA24M011 

This Assignment has three directories (denoted by src, predictions_vannila, predictions_attention) - refer the src/ for code and the remaining for outputs. 

## Wandb Report: 
Wand Report: https://wandb.ai/da6401_assignments/da6401_assignment3_vannila_v1/reports/DA6401-Assignment-3--VmlldzoxMjg0NTkyMQ

## Instructions for run the code: 

1. Create a virtual environment in Python and install the necessary libraries mentioned in the requirements.txt file. 
```console 
$ python3 -m venv env_name 
$ source env_name/bin/activate 
(env_name) $ pip install -r requirements.txt
```

2. The Following packages are needed to use the code in this repository. 
```txt
tensorflow 
numpy 
tqdm 
PyYAML 
wandb
pandas
seaborn
tabulate
```
3. Tensorflow is used to implement RNNs and to create DataLoaders for dataset. 
4. The metrics from different models are logged using wandb. Login to wandb by creating a new project and pasting the API key in CLI. 
```console 
$ wandb login 
```
5. The folder data/ contatins the files to lead the data from .tsv files - change the path as per you preference to load the files
6. To download the dataset execute the following commands
```python
$ python3 data/download_data.py
```
7. The code files named main.py in src/vannilla and src/attention support argparse <br>

Commands to run: 
```consolde
python3 main.py -ed 512 -hd 512 -eb True -db False -bs 16 -ne 1 -nd 3 -et rnn -dt lstm -lr 0.001 -o adam -e 1 -log True  -ddo 0.2 -edo 0.3
```
Lookup Table: 

| Flag Name              | Default Value        | Description                                           |
|------------------------|----------------------|-------------------------------------------------------|
| -ed, --embed_dim           | 128                | Number of embedding dimensions, chose from [128, 256, 512]                        |
| -hd, --hidden_dim      | 128                    | Number of hidden dimensions, chose from [128, 256, 512]                          |
| -eb, --encoder_bias     | True                    | Enable bias in encoder                      |
| -db, --decoder_bias     | True                    | Enable bias in decoder                      |
| -bs, --batch_size     | 16                    | Data Batch size                      |
|-ne, --n_encoder_layer     | 1                    | number of layers in encoder, choose from [1, 2, 3, 4]                      |
|-nd, --n_decoder_layer     | 1                    | number of layers in decoder, choose from [1, 2, 3,4]                      |
|-et, --encoder_type     | rnn                    | cell type, choose from ['rnn', 'lstm', 'gru']                      |
|-dt, --decoder_type     | rnn                    | cell type, choose from ['rnn', 'lstm', 'gru']                      |
|-ddo, --decoder_dropout     | 0.1                    | Decoder dropout                      |
|-edo, --encoder_dropout     | 0.1                    | Encoder dropout                      |
| -lr, --learning_rate   | 0.001                | Learning Rate for the model                           |
| -o, --optimiser        | 'adam'               | Optimiser to minimise the model's loss                |
| -log, --log            | True                 | Use wandb logging                                     |
| -wp, --wandb_project   | 'da6401_assignment2' | Wandb project name                                    |
| -we, --wandb_entity    | 'trail1'             | Wandb entity name                                     |
| -att, --use_attention    | True             | use attention for training                                     |
| -na, --n_attention_layer    | 1             | number of  attention layers for training, choose from [1, 2, 3 ,4]                                     |