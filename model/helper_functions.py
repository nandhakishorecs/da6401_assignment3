import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))

from data.data_loader import * 

import torch 

import numpy as np 
import pandas as pd                     # type: ignore
from tqdm import tqdm                   # type: ignore
import random 


# Training function
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for latin_batch, native_batch in progress_bar:
        latin_batch, native_batch = latin_batch.to(device), native_batch.to(device)
        optimizer.zero_grad()
        output = model(latin_batch, native_batch, teacher_forcing_ratio=0.5)
        output = output[:, 1:, :].reshape(-1, output.size(-1))
        target = native_batch[:, 1:].reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for latin_batch, native_batch in progress_bar:
            latin_batch, native_batch = latin_batch.to(device), native_batch.to(device)
            output = model(latin_batch, native_batch, teacher_forcing_ratio=0.0)
            output = output[:, 1:, :].reshape(-1, output.size(-1))
            target = native_batch[:, 1:].reshape(-1)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Character and word accuracy
            preds = output.argmax(dim=-1).reshape(native_batch.size(0), -1)
            for i in range(native_batch.size(0)):
                pred_seq = preds[i].cpu().numpy()
                target_seq = native_batch[i, 1:].cpu().numpy()
                pred_str = ''.join(dataset.idx2native_char[idx] for idx in pred_seq if idx in dataset.idx2native_char and idx != dataset.native_char2idx['<EOS>'])
                target_str = ''.join(dataset.idx2native_char[idx] for idx in target_seq if idx in dataset.idx2native_char and idx != dataset.native_char2idx['<EOS>'])
                
                # Character accuracy
                min_len = min(len(pred_seq), len(target_seq))
                char_correct += sum(1 for p, t in zip(pred_seq[:min_len], target_seq[:min_len]) if p == t)
                char_total += min_len
                
                # Word accuracy
                if pred_str == target_str:
                    word_correct += 1
                word_total += 1
                
            progress_bar.set_postfix({'loss': loss.item()})
                
    return total_loss / len(dataloader), char_correct / char_total, word_correct / word_total