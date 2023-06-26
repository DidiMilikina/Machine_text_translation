import torch
import torch.nn as nn

from hyperparameters import *
from preprocessing import *

device = get_device()

class TransformerEncoderBlock(nn.Module):
    """
    Encoder block with one Multihead attention layer, feedforward block(Linear, ReLU, Linear, Dropout)
    """
    def __init__(self):
        super(TransformerEncoderBlock, self).__init__()
        
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). 
        # Default: False (seq, batch, feature).
        self.multi_head_attention = nn.MultiheadAttention(batch_first=True, embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS)

        self.layer_norm1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT_PROB)

        self.feedforward = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)
        )

    def forward(self, x):

        # Multihead attention layer
        attn_output = self.multi_head_attention(x, x, x)[0]
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feedforward block
        ff_output = self.feedforward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x
    
# Encoder
class Encoder(nn.Module):
    """
    Encoder component with embedding layer, encoder block and linear layer.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=BERT_FR_VOCAB_SIZE+3, embedding_dim=EMBEDDING_DIM)
        self.linear = nn.Linear(in_features=EMBEDDING_DIM, out_features=EMBEDDING_DIM)

        transformer_block = []
        for i in range(DEPTH_ENCODER):
            transformer_block.append(TransformerEncoderBlock())

        self.transformer_block = nn.Sequential(*transformer_block)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.transformer_block(embedded)
        enc_out = self.linear(embedded)

        return enc_out


class TransformerDecoderBlock(nn.Module):
    """
    Decoder block with Multihead attention layer, cross attention layer, feedforward block(Linear, ReLU, Linear, Dropout)
    """
    def __init__(self):
        super(TransformerDecoderBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm2 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm3 = nn.LayerNorm(EMBEDDING_DIM)

        self.dropout = nn.Dropout(DROPOUT_PROB)

        self.attention_one = nn.MultiheadAttention(embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS)
        self.attention_two = nn.MultiheadAttention(embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS)
     
        self.feedforward = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM * 4),
            nn.ReLU(),
            nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM),
        )
    
    def forward(self, x, encoder_out):

        # Multihead attention layer
        attn_output_one = self.attention_one(x, x, x)[0]
        x = self.layer_norm1(x + self.dropout(attn_output_one))

        # Cross attention layer
        attn_output_two = self.attention_two(x, encoder_out, encoder_out)[0]
        x = self.layer_norm2(x + self.dropout(attn_output_two))

        # Feedforward block
        ff_output = self.feedforward(x)
        x = self.layer_norm3(x + self.dropout(ff_output))

        return x, encoder_out

# Decoder
class Decoder(nn.Module):
    """
    Decoder component with embedding layer, decoder block and linear layer.
    """
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=BERT_FR_VOCAB_SIZE+3, embedding_dim=EMBEDDING_DIM)
        self.linear = nn.Linear(in_features=EMBEDDING_DIM, out_features=BERT_FR_VOCAB_SIZE+3)

        transformer_block = []
        for i in range(DEPTH_DECODER):
            transformer_block.append(TransformerDecoderBlock())

        self.transformer_block = nn.Sequential(*transformer_block)

    def forward(self, encoder_output):

        zero_tensor = torch.zeros((encoder_output.size(0), encoder_output.size(1)), dtype=torch.long).to(device=device)   
        embedded = self.embedding(zero_tensor)
        
        for block in self.transformer_block:
            embedded, _ = block(embedded, encoder_output)

        dec_out = self.linear(embedded)
        
        return dec_out