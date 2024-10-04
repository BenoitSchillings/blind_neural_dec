import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from astropy.io import fits
import glob
import os
import numpy as np
import cv2
import random
from typing import Any

def crop(x):
    return x[200:-200, 200:-200]

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
        variables = model.init(rng, jnp.ones(input_shape), training=False)
        params = variables['params']
        batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )

def load_checkpoint(checkpoint_dir):
    return checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=None)

def load_and_process_images(data_dir, num_images=7):
    image_files = glob.glob(os.path.join(data_dir, 'image_*_1.fits'))
    selected_files = random.sample(image_files, num_images)
    
    X = []
    for file in selected_files:
        image_number = os.path.basename(file).split('_')[1]
        image_set = []
        for j in range(7):
            with fits.open(os.path.join(data_dir, f'image_{image_number}_{j+1}.fits')) as hdul:
                image_set.append(crop(np.array(hdul[0].data.astype(np.float32))))
        X.append(np.stack(image_set, axis=-1))
    
    X = np.stack(X, axis=0)
    X = X / 1e3  # Normalize as in the original code
    return X

def main():
    # Set up paths
    data_dir = './training_data/'  # Adjust this path as needed
    checkpoint_dir = './checkpoints/'
    checkpoint_dir = os.path.abspath('./checkpoints')

    # Load the latest checkpoint
    last_checkpoint = load_checkpoint(checkpoint_dir)
    if last_checkpoint is None:
        print("No checkpoint found. Please train the model first.")
        return

    # Initialize the model with the loaded checkpoint
    rng = jax.random.PRNGKey(0)
    input_shape = (1, 600, 600, 7)  # Adjust these dimensions based on your actual image size
    state = create_train_state(
        rng=rng,
        input_shape=input_shape,
        learning_rate=1e-3,
        params=last_checkpoint['params'],
        batch_stats=last_checkpoint['batch_stats']
    )

    # Load and process a random set of 7 images
    X = load_and_process_images(data_dir)

    # Run the network on the loaded images
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    predictions = state.apply_fn(variables, X, training=False)

    # Display the result image (assuming single-channel output)
    result_image = predictions[0, :, :, 0]  # Take the first image from the batch
    cv2.imshow("Processed Image", result_image / np.max(result_image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()