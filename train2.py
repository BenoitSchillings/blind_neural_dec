import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import glob
import os
import numpy as np
import cv2
import random

def crop(x):
    return x[200:-200, 200:-200]

class FlexibleFITSDataset(Dataset):
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle
        
        self.image_numbers = sorted(list(set([
            os.path.basename(f).split('_')[1] 
            for f in glob.glob(os.path.join(data_dir, 'image_*_1.fits'))
        ])))
        
        with fits.open(os.path.join(data_dir, f'image_{self.image_numbers[0]}_1.fits')) as hdul:
            self.image_shape = crop(hdul[0].data).shape

        self.on_epoch_end()

    def __len__(self):
        return len(self.image_numbers)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_numbers)

    def __getitem__(self, index):
        image_number = self.image_numbers[index]
        X = np.empty((*self.image_shape, 7), dtype=np.float32)
        
        for j in range(7):
            with fits.open(os.path.join(self.data_dir, f'image_{image_number}_{j+1}.fits')) as hdul:
                X[:, :, j] = crop(np.array(hdul[0].data.astype(np.float32)))

        with fits.open(os.path.join(self.data_dir, f'image_{image_number}_sharp.fits')) as hdul:
            y = crop(np.array(hdul[0].data.astype(np.float32)))

        X = X / 1e3
        y = y / 1e3

        return torch.from_numpy(X.transpose(2, 0, 1)), torch.from_numpy(y[None, :, :])

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.upsample:
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FlexibleEncoderDecoder(nn.Module):
    def __init__(self):
        super(FlexibleEncoderDecoder, self).__init__()
        
        self.initial_conv = ConvBlock(7, 32, kernel_size=3)
        
        # Encoder path
        self.e1_conv1 = ConvBlock(32, 64, stride=2)
        self.e1_conv2 = ConvBlock(64, 64)
        self.e1_conv3 = ConvBlock(64, 64)
        self.e1_conv4 = ConvBlock(64, 64, kernel_size=1)
        self.e1_skip = ConvBlock(32, 64, stride=2)
        
        self.e2_conv1 = ConvBlock(64, 64)
        self.e2_conv2 = ConvBlock(64, 64)
        self.e2_conv3 = ConvBlock(64, 64)
        self.e2_conv4 = ConvBlock(64, 64, kernel_size=1)
        
        self.e3_conv1 = ConvBlock(64, 128, stride=2)
        self.e3_conv2 = ConvBlock(128, 128)
        self.e3_conv3 = ConvBlock(128, 128)
        self.e3_conv4 = ConvBlock(128, 128, kernel_size=1)
        self.e3_skip = ConvBlock(64, 128, stride=2)
        
        self.e4_conv1 = ConvBlock(128, 256, stride=2)
        self.e4_conv2 = ConvBlock(256, 256)
        self.e4_conv3 = ConvBlock(256, 256)
        self.e4_conv4 = ConvBlock(256, 256, kernel_size=1)
        self.e4_skip = ConvBlock(128, 256, stride=2)
        
        # Decoder path
        self.d1_conv1 = ConvBlock(256, 128, upsample=True)
        self.d1_conv2 = ConvBlock(128, 128)
        self.d1_conv3 = ConvBlock(128, 128)
        self.d1_conv4 = ConvBlock(128, 128)
        
        self.d2_conv1 = ConvBlock(128, 64, upsample=True)
        self.d2_conv2 = ConvBlock(64, 64)
        self.d2_conv3 = ConvBlock(64, 64)
        self.d2_conv4 = ConvBlock(64, 64)
        
        self.d3_conv1 = ConvBlock(64, 64, upsample=True)
        self.d3_conv2 = ConvBlock(64, 64)
        self.d3_conv3 = ConvBlock(64, 16)
        
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Initial convolutional layer
        x_initial = self.initial_conv(x)
        
        # Encoder path
        e1 = self.e1_conv1(x_initial)
        e1 = self.e1_conv2(e1)
        e1 = self.e1_conv3(e1)
        e1 = self.e1_conv4(e1)
        e1 = e1 + self.e1_skip(x_initial)  # Skip connection
        
        e2 = self.e2_conv1(e1)
        e2 = self.e2_conv2(e2)
        e2 = self.e2_conv3(e2)
        e2 = self.e2_conv4(e2)
        e2 = e2 + e1  # Skip connection
        
        e3 = self.e3_conv1(e2)
        e3 = self.e3_conv2(e3)
        e3 = self.e3_conv3(e3)
        e3 = self.e3_conv4(e3)
        e3 = e3 + self.e3_skip(e2)  # Skip connection
        
        e4 = self.e4_conv1(e3)
        e4 = self.e4_conv2(e4)
        e4 = self.e4_conv3(e4)
        e4 = self.e4_conv4(e4)
        e4 = e4 + self.e4_skip(e3)  # Skip connection
        
        # Decoder path
        d1 = self.d1_conv1(e4)
        d1 = d1 + e3  # Skip connection
        d1 = self.d1_conv2(d1)
        d1 = self.d1_conv3(d1)
        d1 = self.d1_conv4(d1)
        
        d2 = self.d2_conv1(d1)
        d2 = d2 + e2  # Skip connection
        d2 = self.d2_conv2(d2)
        d2 = self.d2_conv3(d2)
        d2 = self.d2_conv4(d2)
        
        d3 = self.d3_conv1(d2)
        d3 = self.d3_conv2(d3)
        d3 = self.d3_conv3(d3)
        
        # Final convolution
        outputs = self.final_conv(d3)
        outputs = outputs + x[:, 0:1, :, :]  # Skip connection from input
        
        return outputs

def train_step(model, optimizer, criterion, inputs, targets):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_step(model, criterion, inputs, targets):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    return loss.item()

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train_and_evaluate(model, train_loader, val_loader, num_epochs, checkpoint_dir, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    start_epoch = 0
    
    # Check if there's a checkpoint to restore
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, latest_checkpoint)
        print(f"Restored checkpoint from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_losses = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = train_step(model, optimizer, criterion, inputs, targets)
            train_losses.append(loss)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {loss:.4f}")
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = eval_step(model, criterion, inputs, targets)
                val_losses.append(loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, np.mean(train_losses), checkpoint_dir)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datasets and data loaders
    train_dataset = FlexibleFITSDataset('./training_data/')
    val_dataset = FlexibleFITSDataset('./val_data/')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Initialize model
    model = FlexibleEncoderDecoder().to(device)
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.abspath('./checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the model
    num_epochs = 100
    trained_model = train_and_evaluate(model, train_loader, val_loader, num_epochs, checkpoint_dir, device)
    
    print("Training completed.")

if __name__ == "__main__":
    main()