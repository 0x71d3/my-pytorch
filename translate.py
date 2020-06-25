import heapq

import torch
import torch.nn as nn
from pyknp import Juman

import tokenizer as tok
from model import TransformerModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


vocab_path = 'web_s_bpe.vocab'
vocab_size = 32000

with open(vocab_path, encoding='utf-8') as f:
    bpe_vocab = [line.split()[0] for line in f]

pad_token = '<pad>'
bos_token = '<bos>'
eos_token = '<eos>'
unk_token = '<unk>'

special_tokens = [pad_token, bos_token, eos_token, unk_token]

vocab = (special_tokens + bpe_vocab)[:vocab_size]

def token2index(token):
    return vocab.index(token) if token in vocab else vocab.index(unk_token)

def index2token(index):
    return vocab[index]

def seq2indexes(seq):
    return [token2index(token) for token in seq]

def indexes2seq(indexes):
    return [index2token(index) for index in indexes]

def pad_seq(seq, pad_token, padded_len):
    return (seq + [pad_token] * padded_len)[:padded_len]


padding_idx = token2index(pad_token)
seq_len_max = 128

def greedy_search(model, memory, memory_key_padding_mask):
    tgt_seq = [tok.bos_token]

    for i in range(seq_len_max - 1):
        padded_seq = pad_seq(tgt_seq, pad_token, seq_len_max)
        tgt_indexes = seq2indexes(padded_seq)
        tgt_tensor = torch.tensor(tgt_indexes).to(device)
        tgt_batch = tgt_tensor.unsqueeze(0)

        tgt = model.tgt_embedding(tgt_batch)
        tgt_mask = model.transformer.generate_square_subsequent_mask(
            tgt_batch.size(-1)
        ).to(device)
        tgt_key_padding_mask = (tgt_batch == padding_idx)

        with torch.no_grad():
            output = model.transformer.decoder(
                tgt=tgt.transpose(0, 1),
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            output = model.fc(output.transpose(0, 1))

        output_index = output.argmax(2).squeeze(0)[i].item()
        output_token = index2token(output_index)

        tgt_seq.append(output_token)
        if output_token == tok.eos_token:
            break
    
    return tgt_seq


def beam_search(model, memory, memory_key_padding_mask, beam_width):
    softmax = nn.LogSoftmax(2)

    beam_heap = []

    tgt_prob = 0.0
    tgt_indexes = seq2indexes(pad_seq([bos_token], pad_token, seq_len_max))
    heapq.heappush(beam_heap, (tgt_prob, tgt_indexes))

    for i in range(seq_len_max - 1):
        heap = []

        for tgt_prob, tgt_indexes in beam_heap:
            tgt_tensor = torch.tensor(tgt_indexes).to(device)
            tgt_batch = tgt_tensor.unsqueeze(0)

            tgt = model.tgt_embedding(tgt_batch)
            tgt_mask = model.transformer.generate_square_subsequent_mask(
                tgt_batch.size(-1)
            ).to(device)
            tgt_key_padding_mask = (tgt_batch == padding_idx)

            with torch.no_grad():
                output = model.transformer.decoder(
                    tgt=tgt.transpose(0, 1),
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                output = softmax(model.fc(output.transpose(0, 1)))

            for j in range(output.size(-1)):
                prob_j = -output[0][i][j].item()
                tgt_indexes[i+1] = j

                heapq.heappush(heap, (tgt_prob + prob_j, tgt_indexes[:]))
        
        beam_heap = []
        for j in range(beam_width):
            heapq.heappush(beam_heap, heapq.heappop(heap))

    return indexes2seq(heapq.heappop(beam_heap)[1])


def translate(model, src_seq, beam=False):
    model.eval()

    padded_seq = pad_seq(src_seq, pad_token, seq_len_max)
    src_indexes = tok.seq2indexes(padded_seq)

    src_tensor = torch.tensor(src_indexes).to(device)
    src_batch = src_tensor.unsqueeze(0)

    src = model.src_embedding(src_batch)
    src_key_padding_mask = (src_batch == padding_idx)

    with torch.no_grad():
        memory = model.transformer.encoder(
            src=src.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask
        )
    memory_key_padding_mask = src_key_padding_mask.clone()

    if beam:
        return beam_search(model, memory, memory_key_padding_mask, 5)

    return greedy_search(model, memory, memory_key_padding_mask)
    

import sys

from pyknp import Juman
from subword_nmt.apply_bpe import BPE

jumanpp = Juman()

codes_path = 'web_s_bpe.codes'
bpe = BPE(
    open(codes_path, encoding='utf-8'),
    vocab=open(vocab_path, encoding='utf-8')
)

model = TransformerModel(
    vocab_size=vocab_size,
    padding_idx=padding_idx
).to(device)
model.load_state_dict(torch.load('transformer_model.pt', device))

for line in iter(sys.stdin.readline, ''):
    result = jumanpp.analysis(line.strip())
    wakati = ' '.join(mrph.midasi for mrph in result.mrph_list())

    src_seq = bpe.process_line(wakati).split()

    tgt_seq = translate(model, src_seq, False)
    print(''.join(tgt_seq).replace('@@', ''))
