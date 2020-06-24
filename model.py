import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super(TransformerModel, self).__init__()

        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        self.src_embedding = nn.Embedding(
            self.vocab_size, 512, self.padding_idx
        )
        self.tgt_embedding = nn.Embedding(
            self.vocab_size, 512, self.padding_idx
        )

        self.transformer = nn.Transformer()

        self.fc = nn.Linear(512, self.vocab_size)

    def forward(self, src, tgt):
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(-2)
        ).to(device)

        src_key_padding_mask = (src == self.padding_idx)
        tgt_key_padding_mask = (tgt == self.padding_idx)
        memory_key_padding_mask = src_key_padding_mask.clone()

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        output = self.transformer(
            src.transpose(0, 1),
            tgt.transpose(0, 1),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        output = self.fc(output.transpose(0, 1))

        return output
