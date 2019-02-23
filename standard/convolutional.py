from keras.layers import Dense, Input
from keras.models import Model
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import save_model
import matplotlib.pyplot as plt

# Standard autoencoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_train = x_train.astype('float32') / 255.  # Convert to floating point
x_test = x_test.reshape((len(x_test), 28, 28, 1))
x_test = x_test.astype('float32') / 255.

latent_size = 2
output_size = x_train.shape[1]

inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))


def encoder(input_img):
  # encoder
  # input = 28 x 28 x 1 (wide and thin)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
  conv1 = BatchNormalization()(conv1)
  conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
  conv1 = BatchNormalization()(conv1)
  pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
  conv2 = BatchNormalization()(conv2)
  conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
  conv2 = BatchNormalization()(conv2)
  pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
  conv3 = BatchNormalization()(conv3)
  conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
  conv3 = BatchNormalization()(conv3)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
  conv4 = BatchNormalization()(conv4)
  conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
  conv4 = BatchNormalization()(conv4)
  return conv4


def decoder(conv4):
  # decoder
  conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
  conv5 = BatchNormalization()(conv5)
  conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
  conv5 = BatchNormalization()(conv5)
  conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
  conv6 = BatchNormalization()(conv6)
  conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
  conv6 = BatchNormalization()(conv6)
  up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
  conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
  conv7 = BatchNormalization()(conv7)
  conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
  conv7 = BatchNormalization()(conv7)
  up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
  decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
  return decoded

autoencoder = Model(input_img, decoder(encoder(input_img)))

# TODO test metrics
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

# Checkpoint

checkpoint = ModelCheckpoint("./checkpoints/convolutional-{epoch:02d}.hdf5")

# Train

autoencoder.fit(x_train, x_train, epochs=1, shuffle=True, batch_size=64, callbacks=[checkpoint])

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

plt.show()