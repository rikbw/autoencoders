from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import save_model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Standard autoencoder

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Flatten images
x_train = x_train.reshape((len(x_train), -1))
x_train = x_train.astype('float32') / 255.  # Convert to floating point
x_test = x_test.reshape((len(x_test), -1))
x_test = x_test.astype('float32') / 255.

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

# Checkpoint

checkpoint = ModelCheckpoint("./checkpoints/standard-{epoch:02d}.hdf5")

# Train

autoencoder.fit(x_train, x_train, epochs=128, shuffle=True, batch_size=256, callbacks=[checkpoint])

# Extra models

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(latent_size,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Save encoder and decoder model

save_model(encoder, './checkpoints/standard-encoder.hdf5')
save_model(decoder, './checkpoints/standard-decoder.hdf5')
