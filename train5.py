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

        flip = 0
        if random.random() < 0.5:
            flip = 1

        flipa = 0
        if random.random() < 0.5:
            flipa = 1
           
        
        for j in range(7):
            with fits.open(os.path.join(self.data_dir, f'image_{image_number}_{j+1}.fits')) as hdul:
                data = crop(np.array(hdul[0].data.astype(np.float32)))
                if (flip):
                    data = np.flip(data, axis=0)
                if (flipa):
                    data = np.flip(data, axis=1)

                X[:, :, j] = data

        with fits.open(os.path.join(self.data_dir, f'image_{image_number}_sharp.fits')) as hdul:
            y = crop(np.array(hdul[0].data.astype(np.float32)))

        if (flip):
            y = np.flip(y, axis=0)
        if (flipa):
            y = np.flip(y, axis=1)
    
        X = X / 256.0
        y = y / 256.0

        #if random.random() < 0.5:
        #    X = np.flip(X, axis=0)
        #    y = np.flip(y, axis=0)

        #if random.random() < 0.5:
        #    X = np.flip(X, axis=1)
        #    y = np.flip(y, axis=1)

            

        return torch.from_numpy(X.transpose(2, 0, 1)), torch.from_numpy(y[None, :, :])

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3, stride=1, upsample=False):
        super(ConvBlock, self).__init__()

        self.upsample = upsample

        if (upsample):
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=1)
        else:
            self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride)

        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.1)

        self.reflection = nn.ReflectionPad2d(int((kernel_size-1)/2))
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        
        if (self.upsample):
            out = self.upsample(out)

        out = self.reflection(out)
        out = self.conv(out)
            
        return out

class deconv_block(nn.Module):
    def __init__(self, n_frames):
        super(deconv_block, self).__init__()
        self.A01 = ConvBlock(n_frames, 32, kernel_size=3)
        
        self.C01 = ConvBlock(32, 64, stride=2)
        self.C02 = ConvBlock(64, 64)
        self.C03 = ConvBlock(64, 64)
        self.C04 = ConvBlock(64, 64, kernel_size=1)

        self.C11 = ConvBlock(64, 64)
        self.C12 = ConvBlock(64, 64)
        self.C13 = ConvBlock(64, 64)
        self.C14 = ConvBlock(64, 64, kernel_size=1)
        
        self.C21 = ConvBlock(64, 128, stride=2)
        self.C22 = ConvBlock(128, 128)
        self.C23 = ConvBlock(128, 128)
        self.C24 = ConvBlock(128, 128, kernel_size=1)
        
        self.C31 = ConvBlock(128, 256, stride=2)
        self.C32 = ConvBlock(256, 256)
        self.C33 = ConvBlock(256, 256)
        self.C34 = ConvBlock(256, 256, kernel_size=1)
        
        self.C41 = ConvBlock(256, 128, upsample=True)
        self.C42 = ConvBlock(128, 128)
        self.C43 = ConvBlock(128, 128)
        self.C44 = ConvBlock(128, 128)
        
        self.C51 = ConvBlock(128, 64, upsample=True)
        self.C52 = ConvBlock(64, 64)
        self.C53 = ConvBlock(64, 64)
        self.C54 = ConvBlock(64, 64)
        
        self.C61 = ConvBlock(64, 64, upsample=True)
        self.C62 = ConvBlock(64, 64)
        self.C63 = ConvBlock(64, 16)

        self.C64 = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.C64.weight)
        nn.init.constant_(self.C64.bias, 0.1)

    def forward(self, x):        
        A01 = self.A01(x)

        # N -> N/2
        C01 = self.C01(A01)
        C02 = self.C02(C01)
        C03 = self.C03(C02)
        C04 = self.C04(C03)
        C04 += C01
        
        # N/2 -> N/2
        C11 = self.C11(C04)
        C12 = self.C12(C11)
        C13 = self.C13(C12)
        C14 = self.C14(C13)
        C14 += C11
        
        # N/2 -> N/4
        C21 = self.C21(C14)
        C22 = self.C22(C21)
        C23 = self.C23(C22)
        C24 = self.C24(C23)
        C24 += C21
        
        # N/4 -> N/8
        C31 = self.C31(C24)
        C32 = self.C32(C31)
        C33 = self.C33(C32)
        C34 = self.C34(C33)
        C34 += C31
        
        C41 = self.C41(C34)
        C41 += C24
        C42 = self.C42(C41)
        C43 = self.C43(C42)
        C44 = self.C44(C43)
        C44 += C41
        
        C51 = self.C51(C44)
        C51 += C14
        C52 = self.C52(C51)
        C53 = self.C53(C52)
        C54 = self.C54(C53)
        C54 += C51
        
        C61 = self.C61(C54)        
        C62 = self.C62(C61)
        C63 = self.C63(C62)
        C64 = self.C64(C63)
        out = C64 + x[:,0:1,:,:]
        
        return out

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
    
    cv2.namedWindow("Inputs", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Outputs", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Targets", cv2.WINDOW_NORMAL)
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_losses = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Display input, output, and target images
            with torch.no_grad():
                outputs = model(inputs)
                
                if (batch_idx % 20 == 0):
                    # Convert tensors to numpy arrays and scale for display
                    inputs_display = inputs.cpu().numpy()  # This will be (batch_size, 7, height, width)
                    outputs_display = outputs.cpu().numpy()  # This will be (batch_size, 1, height, width)
                    targets_display = targets.cpu().numpy()  # This will be (batch_size, 1, height, width)
                    
                    # Function to normalize and convert to uint8
                    def normalize_for_display(img):
                        return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    # Create a grid of input images for each sample in the batch
                    input_grids = []
                    for sample_inputs in inputs_display:
                        sample_grid = np.hstack([normalize_for_display(img) for img in sample_inputs])
                        input_grids.append(sample_grid)
                    input_display = np.vstack(input_grids)
                    
                    # Create a grid of output images
                    output_display = np.vstack([normalize_for_display(out[0]) for out in outputs_display])
                    
                    # Create a grid of target images
                    target_display = np.vstack([normalize_for_display(tar[0]) for tar in targets_display])
                    
                    # Display images
                    cv2.imshow("Inputs", input_display)
                    cv2.imshow("Outputs", output_display)
                    cv2.imshow("Targets", target_display)
                    cv2.waitKey(1)  # Display for 1ms
            
            # Training step
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
    
    cv2.destroyAllWindows()
    return model

def print_network_structure(model):
    print(model)
    print("\nDetailed layer information:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datasets and data loaders
    train_dataset = FlexibleFITSDataset('./training_data/')
    val_dataset = FlexibleFITSDataset('./val_data/')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    model = deconv_block(n_frames=7).to(device)
    
    # Print network structure
    print_network_structure(model)
    
    # Set up checkpoint directory
    checkpoint_dir = os.path.abspath('./checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train the model
    num_epochs = 100
    trained_model = train_and_evaluate(model, train_loader, val_loader, num_epochs, checkpoint_dir, device)
    
    print("Training completed.")

if __name__ == "__main__":
    main()