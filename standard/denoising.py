from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import save_model
import matplotlib.pyplot as plt
import numpy as np

# Denoising autoencoder
x_train = np.load('/Users/rikbw/Dev/autoencoders/data/mnist_noisy_35.npy')
x_test = np.load('/Users/rikbw/Dev/autoencoders/data/mnist_noisy_test_35.npy')


(x_train_denoised, _), (x_test_denoised, _) = mnist.load_data()

# Flatten images
x_train_denoised = x_train_denoised.reshape((len(x_train), -1))
x_train_denoised = x_train_denoised.astype('float32') / 255.  # Convert to floating point
x_test_denoised = x_test_denoised.reshape((len(x_test), -1))
x_test_denoised = x_test_denoised.astype('float32') / 255.

latent_size = 64
output_size = x_train.shape[1]

input_img = Input(shape=(784,)) # Input placeholder
# Tensor of encoded input
encoded = Dense(latent_size, activation='relu')(input_img)
# Tensor of decoded input
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

# TODO test metrics
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train

autoencoder.fit(x_train, x_train_denoised, epochs=64, shuffle=True, batch_size=256)

# Extra models

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(latent_size,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Save encoder and decoder model

# save_model(autoencoder, './checkpoints/denoising.hdf5')
# save_model(encoder, './checkpoints/denoising-encoder.hdf5')
# save_model(decoder, './checkpoints/denoising-decoder.hdf5')

# Visualize

decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()