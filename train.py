import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from astropy.io import fits
from tensorflow.keras.utils import Sequence
import glob
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from astropy.io import fits
import glob
import os

class FlexibleFITSDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=4, shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get all unique image numbers
        self.image_numbers = sorted(list(set([
            os.path.basename(f).split('_')[1] 
            for f in glob.glob(os.path.join(data_dir, 'image_*_1.fits'))
        ])))
        
        # Determine image shape from the first image
        with fits.open(os.path.join(data_dir, f'image_{self.image_numbers[0]}_1.fits')) as hdul:
            self.image_shape = hdul[0].data.shape

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_numbers) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_numbers = [self.image_numbers[k] for k in indexes]
        X, y = self.__data_generation(batch_image_numbers)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_numbers))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_image_numbers):
        X = np.empty((self.batch_size, *self.image_shape, 7), dtype=np.float32)
        y = np.empty((self.batch_size, *self.image_shape, 1), dtype=np.float32)

        for i, image_number in enumerate(batch_image_numbers):
            # Load 7 blurred images
            for j in range(7):
                with fits.open(os.path.join(self.data_dir, f'image_{image_number}_{j+1}.fits')) as hdul:
                    X[i, :, :, j] = np.array(hdul[0].data.astype(np.float32))

            # Load sharp image
            with fits.open(os.path.join(self.data_dir, f'image_{image_number}_sharp.fits')) as hdul:
                y[i, :, :, 0] = np.array(hdul[0].data.astype(np.float32))

        # Normalize the data
        X = X / 1e3
        y = y / 1e3
        X = np.array(X)
        # Convert to TensorFlow tensors
        print(type(X))
        #y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        #X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        #print(X_tensor)

        return X, y

def ConvBlock(filters, kernel_size=3, strides=1, upsample=False):
    def block(x):
        if upsample:
            x = layers.UpSampling2D(size=(2, 2))(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.ZeroPadding2D(padding=(kernel_size-1)//2)(x)
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='valid')(x)
        return x
    return block

def flexible_encoder_decoder_network(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = ConvBlock(32, kernel_size=3)(inputs)
    
    # Encoder path
    e1 = ConvBlock(64, strides=2)(x)
    e1 = ConvBlock(64)(e1)
    e1 = ConvBlock(64)(e1)
    e1 = ConvBlock(64, kernel_size=1)(e1)
    e1 = layers.Add()([e1, ConvBlock(64, strides=2)(x)])  # Skip connection
    
    e2 = ConvBlock(64)(e1)
    e2 = ConvBlock(64)(e2)
    e2 = ConvBlock(64)(e2)
    e2 = ConvBlock(64, kernel_size=1)(e2)
    e2 = layers.Add()([e2, e1])  # Skip connection
    
    e3 = ConvBlock(128, strides=2)(e2)
    e3 = ConvBlock(128)(e3)
    e3 = ConvBlock(128)(e3)
    e3 = ConvBlock(128, kernel_size=1)(e3)
    e3 = layers.Add()([e3, ConvBlock(128, strides=2)(e2)])  # Skip connection
    
    e4 = ConvBlock(256, strides=2)(e3)
    e4 = ConvBlock(256)(e4)
    e4 = ConvBlock(256)(e4)
    e4 = ConvBlock(256, kernel_size=1)(e4)
    e4 = layers.Add()([e4, ConvBlock(256, strides=2)(e3)])  # Skip connection
    
    # Decoder path
    d1 = ConvBlock(128, upsample=True)(e4)
    d1 = layers.Add()([d1, e3])  # Skip connection
    d1 = ConvBlock(128)(d1)
    d1 = ConvBlock(128)(d1)
    d1 = ConvBlock(128)(d1)
    
    d2 = ConvBlock(64, upsample=True)(d1)
    d2 = layers.Add()([d2, e2])  # Skip connection
    d2 = ConvBlock(64)(d2)
    d2 = ConvBlock(64)(d2)
    d2 = ConvBlock(64)(d2)
    
    d3 = ConvBlock(64, upsample=True)(d2)
    d3 = ConvBlock(64)(d3)
    d3 = ConvBlock(16)(d3)
    
    # Final convolution
    outputs = layers.Conv2D(1, kernel_size=1, strides=1)(d3)
    outputs = layers.Add()([outputs, inputs[:, :, :, 0:1]])  # Skip connection from input
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create data generators
train_generator = FlexibleFITSDataGenerator('./training_data/', batch_size=4)
val_generator = FlexibleFITSDataGenerator('./training_data/', batch_size=4)

# Get the image shape from the generator
input_shape = (*train_generator.image_shape, 7)

# Create the model with the correct input shape
model = flexible_encoder_decoder_network(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # You might want to use a custom loss function

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(122)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')
plt.legend()
plt.title('Training and Validation Loss (Log Scale)')

plt.tight_layout()
plt.show()