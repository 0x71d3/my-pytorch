import torch


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

def seq2indexes(seq):
    return [token2index(token) for token in seq]


seq_len_max = 128
batch_size = 32

def pad_seq(seq, pad_token, padded_len):
    return (seq + [pad_token] * padded_len)[:padded_len]

srcs = []
tgts = []

with open('btsj_seq_srcs.BPE.txt', encoding='utf-8') as f:
    for line in f:
        src_seq = line.split() + [eos_token]
        padded_seq = pad_seq(src_seq, pad_token, seq_len_max)
        src_indexes = seq2indexes(padded_seq)
        srcs.append(src_indexes)
with open('btsj_seq_tgts.BPE.txt', encoding='utf-8') as f:
    for line in f:
        tgt_seq = [bos_token] + line.split() + [eos_token]
        padded_seq = pad_seq(tgt_seq, pad_token, seq_len_max)
        tgt_indexes = seq2indexes(padded_seq)
        tgts.append(tgt_indexes)

with open('nucc_seq_srcs.BPE.txt', encoding='utf-8') as f:
    for line in f:
        src_seq = line.split() + [eos_token]
        padded_seq = pad_seq(src_seq, pad_token, seq_len_max)
        src_indexes = seq2indexes(padded_seq)
        srcs.append(src_indexes)
with open('nucc_seq_tgts.BPE.txt', encoding='utf-8') as f:
    for line in f:
        tgt_seq = [bos_token] + line.split() + [eos_token]
        padded_seq = pad_seq(tgt_seq, pad_token, seq_len_max)
        tgt_indexes = seq2indexes(padded_seq)
        tgts.append(tgt_indexes)


src_tensor = torch.tensor(srcs).to(device)
tgt_tensor = torch.tensor(tgts).to(device)

dataset = torch.utils.data.TensorDataset(src_tensor, tgt_tensor)

loader = torch.utils.data.DataLoader(dataset, batch_size, True)
