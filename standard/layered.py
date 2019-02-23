from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import save_model
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import save_model

# Standard autoencoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images
x_train = x_train.reshape((len(x_train), -1))
x_train = x_train.astype('float32') / 255.  # Convert to floating point
x_test = x_test.reshape((len(x_test), -1))
x_test = x_test.astype('float32') / 255.

latent_size = 2
output_size = x_train.shape[1]

input_img = Input(shape=(784,)) # Input placeholder
# Tensor of encoded input
encoded1 = Dense(256, activation='relu')(input_img)
encoded2 = Dense(128, activation='relu')(encoded1)
encoded3 = Dense(32, activation='relu')(encoded2)
h = Dense(latent_size, activation='relu')(encoded3)
# Tensor of decoded input
decoded1 = Dense(32, activation='sigmoid')(h)
decoded2 = Dense(128, activation='sigmoid')(decoded1)
decoded3 = Dense(256, activation='sigmoid')(decoded2)
r = Dense(784, activation='sigmoid')(decoded3)

autoencoder = Model(input_img, r)

autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

# Checkpoint

# checkpoint = ModelCheckpoint("./checkpoints/layered-{epoch:02d}.hdf5")

# Train

autoencoder.fit(x_train, x_train, epochs=1, shuffle=True, batch_size=512)


# Extra models

encoder = Model(input_img, h)

encoded_input = Input(shape=(latent_size,))

# Visualizations

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

latent_representation = encoder.predict(x_train)
colors = y_train

plt.figure(figsize=(20,4))
plt.scatter(latent_representation[:, 0], latent_representation[:, 1], c=colors, cmap='tab10')

plt.show()

save_model(autoencoder, 'checkpoints/layered.hdf5')
