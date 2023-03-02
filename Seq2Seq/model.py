import random

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.input_dim = model_cfg['input_dim']
        self.emb_dim = model_cfg['emb_dim']
        self.hid_dim = model_cfg['hid_dim']
        self.n_layers = model_cfg['n_layers']
        self.dropout = model_cfg['dropout']

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout)
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.output_dim = model_cfg['output_dim']
        self.emb_dim = model_cfg['emb_dim']
        self.hid_dim = model_cfg['hid_dim']
        self.n_layers = model_cfg['n_layers']
        self.dropout = model_cfg['dropout']

        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=self.dropout)
        self.fc_out = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.apply(self._init_weights) # model initialization

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs
    
    def _init_weights(self, m):
        """
        Initialize model weights corresponding uniform random
        """
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)