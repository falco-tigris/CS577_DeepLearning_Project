import torch
import pandas as pd
from utils import set_seed
from data.preprocessing import create_datasets
from models.reportnet import ImageEncoderReportDecoder, ImageEncoderReportDecoderConfig

# Pretrained CNN models
set_seed(33)
img_enc_resnet = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
img_enc_resnet.fc = torch.nn.Identity()
img_enc_resnet.input_shape = (224, 224)
img_enc_resnet.output_shape = (512, 1)

# Select one of the above
img_enc_name =  "RestNet" # "ResNetAE" "Direct" "ResNet18"
img_enc = img_enc_resnet
img_enc_width, img_enc_height = img_enc.input_shape
img_enc_out_shape = img_enc.output_shape
block_size = img_enc_out_shape[0]

# Load data
projections = pd.read_csv('data/indiana_projections.csv')
projections = projections.set_index('uid')
reports = pd.read_csv('data/indiana_reports.csv')
reports = reports.set_index('uid')

# Create dataloaders
train_data, val_data, test_data = create_datasets(projections, reports, 'data/images/images_normalized', batch_size=32)

# Get vocabulary
word2id = train_data.word2idx.union(val_data.word2idx).union(test_data.word2idx)
id2word = train_data.idx2word.union(val_data.idx2word).union(test_data.idx2word)

# Create model
vocab_size = train_data.dataset.get_vocab_size()
config = ImageEncoderReportDecoderConfig(vocab_size, block_size, 512, True, True, None)
model = ImageEncoderReportDecoder(config, img_enc, img_enc_out_shape, img_enc_name)

# Train model
from train import TrainerConfig, Trainer

epochs = 10
tokens_per_epoch = len(train_data) * block_size

train_config = TrainerConfig(max_epochs=epochs, batch_size=16, learning_rate=1.0e-3,
                          betas = (0.9, 0.95), weight_decay=0, lr_decay=True,
                          warmup_tokens=tokens_per_epoch, 
                          final_tokens= epochs*tokens_per_epoch,
                          ckpt_path='reportnet',
                          num_workers=8,
                          pretrain = False)

trainer = Trainer(model, train_data, val_data, train_config, word2id, id2word)
print("Training model...")
trainer.train()

# Save model
torch.save(model.state_dict(), 'model.pth')