# DA6401 Introduction to Deep Learning - Assignment 3
This repository contains all the code for Assignment 3 for the Introduction to Deep Learning Course (DA6401) offered at Wadhwani School of Data Science and AI, Indian Institute of Technology Madras. 

**Course Instructor**: Prof. Mitesh Khapra <br>
**Author**: Nandhakishore C S <br>
**Roll Number**: DA24M011 

This Assignment has three directories (denoted by src, predictions_vannila, predictions_attention) - refer the src/ for code and the remaining for outputs. 


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
 
