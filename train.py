import math
import time

import torch
import torch.nn as nn

from model import TransformerModel
from data import vocab_size, pad_token, token2index, loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, loader, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(loader):
        src_batch, tgt_batch = batch

        optimizer.zero_grad()

        output = model(src_batch, tgt_batch[:,:-1])

        loss = criterion(
            output.reshape(-1, vocab_size),
            tgt_batch[:,1:].reshape(-1)
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src_batch, tgt_batch = batch

            output = model(src_batch, tgt_batch[:, :-1])

            loss = criterion(
                output.reshape(-1, vocab_size),
                tgt_batch[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


padding_idx = token2index(pad_token)
model = TransformerModel(vocab_size=vocab_size, padding_idx=padding_idx).to(device)
# model.load_state_dict(torch.load('transformer_model.pt', device))

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

n_epochs = 10
clip = 1.0

best_valid_loss = float('inf')
best_model = None

for epoch in range(n_epochs):
    start_time = time.time()

    train_loss = train(model, loader, optimizer, criterion, clip)
    valid_loss = evaluate(model, loader, criterion)

    end_time = time.time()

    epoch_time = end_time - start_time
    epoch_mins = int(epoch_time / 60)
    epoch_secs = int(epoch_time) % 60

    print('epoch: {} | time: {}m {}s'.format(
        epoch + 1,
        math.floor(epoch_time / 60),
        math.floor(epoch_secs) % 60
    ))
    print('    train loss: {} | train ppl: {}'.format(
        train_loss,
        math.exp(train_loss)
    ))
    print('    valid loss: {} | valid ppl: {}'.format(
        valid_loss,
        math.exp(valid_loss)
    ))

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = model

torch.save(model.state_dict(), 'transformer_model.pt')
# torch.save(best_model.state_dict(), 'best_transformer_model.pt')
