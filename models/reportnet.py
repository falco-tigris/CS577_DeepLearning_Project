import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(12)
        self.poolAvg = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 96, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(96)
        self.conv5 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 384, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(384)
        self.conv7 = nn.Conv2d(384, 768, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(768)
        self.conv8 = nn.Conv2d(768, 12, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(12)
        # Input size = 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        x = x.float()
        x = self.poolAvg(F.relu(self.bn1(self.conv1(x))))
        x = self.poolAvg(F.relu(self.bn2(self.conv2(x))))
        x = self.poolAvg(F.relu(self.bn3(self.conv3(x))))
        x = self.poolAvg(F.relu(self.bn4(self.conv4(x))))        
        x = self.poolAvg(F.relu(self.bn5(self.conv5(x))))
        x = self.poolAvg(F.relu(self.bn6(self.conv6(x))))
        x = self.poolAvg(F.relu(self.bn7(self.conv7(x))))
        x = self.poolAvg(F.relu(self.bn8(self.conv8(x))))        
        x = x.view(-1, 768)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        # Info
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-np.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * np.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * np.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(transformer_out)
        
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    
class ReportNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hidden_dim=512):
        super(ReportNet, self).__init__()
        self.cnn = CNN()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, image1, report):
        # Extract features from images using CNN
        image1_features = self.cnn(image1)
        # image2_features = self.cnn(image2)
        # image_features = torch.cat((image1_features, image2_features), dim=1)
        # Pass the image features and text through the transformer
        transformer_output = self.transformer(image1_features.unsqueeze(0), tgt=report)
        # Generate the report
        report = self.fc_out(transformer_output)
        return report