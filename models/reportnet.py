import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self, d_model):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 1, 1)
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
        self.fc2 = nn.Linear(512, d_model)

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

# Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

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

class DecoderGenerator(nn.Module):
    def __init__(self, vocab_size, dim_model, num_heads, num_decoder_layers, dropout_p):
        super(DecoderGenerator, self).__init__()
        
        self.dim_model = dim_model
        self.emb = nn.Embedding(vocab_size, dim_model)
        # max_len determines how far the position can have an effect on a token (window)
        self.pos = PositionalEncoding(dim_model, dropout_p, max_len=5000)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim_model, nhead=num_heads, dropout=dropout_p, activation='gelu', 
            ),
            num_decoder_layers,
        )
        self.dense = nn.Linear(dim_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tgt, tgt_mask, memory):
        # tgt = (batch_size, tgt_seq_len)
        # tgt_mask = (tgt_seq_len, tgt_seq_len)
        # memory = (batch_size, src_seq_len, dim_model)
        
        # Embedding + positional encoding - 
        tgt = self.emb(tgt) * np.sqrt(self.dim_model)
        # Permute to obtain (tgt_seq_len, batch_size, dim_model)
        tgt = torch.permute(tgt, (1, 0, 2))
        tgt = self.pos(tgt)

        # Transformer blocks - Out size = (tgt_seq_len, batch_size, dim_model)
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask) #memory_mask=memory_mask)
        
        # Dense layer - Out size = (tgt_seq_len, batch_size, vocab_size)
        out = self.dense(out)      
        return self.softmax(out)


class ReportNet(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, max_len=60, dropout_p=0.4):
        super(ReportNet, self).__init__()
        self.max_len = max_len
        self.encoder = CNN(emb_dim)
        self.decoder = DecoderGenerator(
            vocab_size=vocab_size,
            dim_model=emb_dim,
            num_heads=2,
            num_decoder_layers=6,
            dropout_p=dropout_p,
        )

    def forward(self, image1, report, tgt_mask):
        # Extract features from images using CNN
        image1_features = self.encoder(image1)
        image1_features = image1_features.unsqueeze(0)

        # Generate the report
        transformer_output = self.decoder(report, tgt_mask, image1_features)
        return transformer_output
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
