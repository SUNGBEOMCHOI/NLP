import math
import time
import random
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import BucketIterator

from data import MultiDataset
from model import Encoder, Decoder, Seq2Seq

def train(args, cfg):
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device']=='cuda' and torch.cuda.is_available() else 'cpu')
    train_cfg = cfg['train']
    batch_size = train_cfg['batch_size']
    train_epochs = train_cfg['train_epochs']
    clip = train_cfg['clip'] # gradient clipping


    model_cfg = cfg['model']

    ########################
    #    train settings    #
    ########################
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == torch.device('cuda'):
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    dataset = MultiDataset()
    train_data, valid_data, test_data = dataset.get_dataset()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        device = device
    )
    dataset.build_vocab(train_data)

    model_cfg['encoder']['input_dim'] = len(dataset.SRC.vocab)
    model_cfg['decoder']['output_dim'] = len(dataset.TRG.vocab)

    ########################
    #      Make model      #
    ########################
    enc = Encoder(model_cfg['encoder'])
    dec = Decoder(model_cfg['decoder'])

    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = dataset.TRG.vocab.stoi[dataset.TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    ########################
    #      Train model     #
    ########################
    model.train()
    epoch_loss = 0
    best_valid_loss = float('inf')

    for epoch in range(1, train_epochs+1):
        start_time = time.time()
        for i, batch in enumerate(train_iterator):
            src = batch.src
            trg = batch.trg

            optimizer.zero_grad()

            output = model(src, trg)

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # trg = [(trg len - 1)*batch size]
            # output = [(trg len - 1)*batch size, output dim]

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_iterator)
        valid_loss = evaluate(model, valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            # trg = [(trg len - 1) * batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss/len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(args, cfg)