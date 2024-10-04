import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from astropy.io import fits
import glob
import os
import os.path
import numpy as np
from typing import Sequence, Any
from flax.training import checkpoints
import orbax.checkpoint
import GPUtil
import cv2
import gc

def crop(x):
    return(x[200:-200,200:-200])

class FlexibleFITSDataset:
    def __init__(self, data_dir, batch_size=4, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.image_numbers = sorted(list(set([
            os.path.basename(f).split('_')[1] 
            for f in glob.glob(os.path.join(data_dir, 'image_*_1.fits'))
        ])))
        
        with fits.open(os.path.join(data_dir, f'image_{self.image_numbers[0]}_1.fits')) as hdul:
            self.image_shape = crop(hdul[0].data).shape

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_numbers) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_numbers))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_numbers = [self.image_numbers[k] for k in indexes]
        return self.__data_generation(batch_image_numbers)

    def __data_generation(self, batch_image_numbers):
        X = np.empty((self.batch_size, *self.image_shape, 7), dtype=np.float32)
        y = np.empty((self.batch_size, *self.image_shape, 1), dtype=np.float32)

        for i, image_number in enumerate(batch_image_numbers):
            for j in range(7):
                with fits.open(os.path.join(self.data_dir, f'image_{image_number}_{j+1}.fits')) as hdul:
                    X[i, :, :, j] = crop(np.array(hdul[0].data.astype(np.float32)))

            with fits.open(os.path.join(self.data_dir, f'image_{image_number}_sharp.fits')) as hdul:
                y[i, :, :, 0] = crop(np.array(hdul[0].data.astype(np.float32)))

            i = y[i, :, :, 0]
            
            cv2.imshow("image", i / np.max(i))
            cv2.waitKey(1)

            print("load")

        X = X / 1e3
        y = y / 1e3

        return X, y

class ConvBlock(nn.Module):
    filters: int
    kernel_size: int = 3
    strides: int = 1
    upsample: bool = False

    @nn.compact
    def __call__(self, x, training: bool = True):
        if self.upsample:
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
        
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.filters, kernel_size=(self.kernel_size, self.kernel_size), 
                    strides=(self.strides, self.strides), padding='SAME')(x)
        return x

class FlexibleEncoderDecoder(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Initial convolutional layer
        x = ConvBlock(32, kernel_size=3)(x, training=training)
        
        # Encoder path
        e1 = ConvBlock(64, strides=2)(x, training=training)
        e1 = ConvBlock(64)(e1, training=training)
        e1 = ConvBlock(64)(e1, training=training)
        e1 = ConvBlock(64, kernel_size=1)(e1, training=training)
        e1 = e1 + ConvBlock(64, strides=2)(x, training=training)  # Skip connection
        
        e2 = ConvBlock(64)(e1, training=training)
        e2 = ConvBlock(64)(e2, training=training)
        e2 = ConvBlock(64)(e2, training=training)
        e2 = ConvBlock(64, kernel_size=1)(e2, training=training)
        e2 = e2 + e1  # Skip connection
        
        e3 = ConvBlock(128, strides=2)(e2, training=training)
        e3 = ConvBlock(128)(e3, training=training)
        e3 = ConvBlock(128)(e3, training=training)
        e3 = ConvBlock(128, kernel_size=1)(e3, training=training)
        e3 = e3 + ConvBlock(128, strides=2)(e2, training=training)  # Skip connection
        
        e4 = ConvBlock(256, strides=2)(e3, training=training)
        e4 = ConvBlock(256)(e4, training=training)
        e4 = ConvBlock(256)(e4, training=training)
        e4 = ConvBlock(256, kernel_size=1)(e4, training=training)
        e4 = e4 + ConvBlock(256, strides=2)(e3, training=training)  # Skip connection
        
        # Decoder path
        d1 = ConvBlock(128, upsample=True)(e4, training=training)
        d1 = d1 + e3  # Skip connection
        d1 = ConvBlock(128)(d1, training=training)
        d1 = ConvBlock(128)(d1, training=training)
        d1 = ConvBlock(128)(d1, training=training)
        
        d2 = ConvBlock(64, upsample=True)(d1, training=training)
        d2 = d2 + e2  # Skip connection
        d2 = ConvBlock(64)(d2, training=training)
        d2 = ConvBlock(64)(d2, training=training)
        d2 = ConvBlock(64)(d2, training=training)
        
        d3 = ConvBlock(64, upsample=True)(d2, training=training)
        d3 = ConvBlock(64)(d3, training=training)
        d3 = ConvBlock(16)(d3, training=training)
        
        # Final convolution
        outputs = nn.Conv(features=1, kernel_size=(1, 1))(d3)
        outputs = outputs + x[:, :, :, 0:1]  # Skip connection from input
        
        return outputs

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng, input_shape, learning_rate, params=None, batch_stats=None):
    model = FlexibleEncoderDecoder()
    if params is None or batch_stats is None:
        variables = model.init(rng, jnp.ones(input_shape), training=True)
        params = variables['params']
        batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )


#@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        inputs, targets = batch
        variables = {'params': params, 'batch_stats': state.batch_stats}
        predictions, new_batch_stats = state.apply_fn(
            variables, inputs, training=True, mutable=['batch_stats']
        )
        loss = jnp.mean((predictions - targets) ** 2)
        return loss, (predictions, new_batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (predictions, new_batch_stats)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, loss

@jax.jit
def eval_step(state, batch):
    inputs, targets = batch
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    predictions = state.apply_fn(variables, inputs, training=False)
    loss = jnp.mean((predictions - targets) ** 2)
    return loss

def save_checkpoint(state, checkpoint_dir, step):
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target={
            'step': state.step,
            'params': state.params,
            'opt_state': state.opt_state,
            'batch_stats': state.batch_stats
        },
        step=step,
        overwrite=True,
        keep=3
    )

def load_checkpoint(checkpoint_dir):
    return checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)

def print_gpu_utilization():
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(f'GPU {i}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB')

def train_and_evaluate(state, train_ds, val_ds, num_epochs, checkpoint_dir):
    # Check if there's a checkpoint to restore
    last_checkpoint = load_checkpoint(checkpoint_dir)
    if last_checkpoint:
        try:
            # Reinitialize the state with loaded parameters and batch stats, but new optimizer state
            state = create_train_state(
                rng=jax.random.PRNGKey(0),
                input_shape=state.params['ConvBlock_0']['Conv_0']['kernel'].shape,
                learning_rate=1e-3,  # Make sure this matches your initial learning rate
                params=last_checkpoint['params'],
                batch_stats=last_checkpoint['batch_stats']
            )
            state = state.replace(step=last_checkpoint['step'])
            start_epoch = state.step // len(train_ds)
            print(f"Restored checkpoint from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from initial state...")
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        # Training
        train_losses = []
        period = 0
        for batch_idx, batch in enumerate(train_ds):
            state, loss = train_step(state, batch)
            train_losses.append(loss)
            
            # Save checkpoint after every batch
            print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}")
            if (period % 10 == 0):
                save_checkpoint(state, checkpoint_dir, state.step)
                print(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx+1}")
            period = period + 1
            del batch
            gc.collect()
            
            
            # Print GPU utilization
            if batch_idx % 10 == 0:  # Print every 10 batches
                print_gpu_utilization()
        
        # Validation
        val_losses = []
        for batch in val_ds:
            loss = eval_step(state, batch)
            val_losses.append(loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
        print_gpu_utilization()
    
    return state


# Initialize datasets
train_dataset = FlexibleFITSDataset('./training_data/', batch_size=2)
val_dataset = FlexibleFITSDataset('./val_data/', batch_size=2)

# Initialize model and state
rng = jax.random.PRNGKey(0)
input_shape = (1, *train_dataset.image_shape, 7)
state = create_train_state(rng, input_shape, learning_rate=1e-3)

# Set up checkpoint directory with absolute path
checkpoint_dir = os.path.abspath('./checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Print initial GPU utilization
print("Initial GPU Memory Usage:")
print_gpu_utilization()

# Train the model
num_epochs = 100
final_state = train_and_evaluate(state, train_dataset, val_dataset, num_epochs, checkpoint_dir)

# Print final GPU utilization
print("Final GPU Memory Usage:")
print_gpu_utilization()