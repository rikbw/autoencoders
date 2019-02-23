import keras
from keras.datasets import fashion_mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import math
from sklearn.manifold import TSNE
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Flatten images
x_train = x_train.reshape((len(x_train), -1))
x_train = x_train.astype('float32') / 255.  # Convert to floating point
x_test = x_test.reshape((len(x_test), -1))
x_test = x_test.astype('float32') / 255.

model = load_model('./checkpoints/standard-128.hdf5')
decoder = load_model('./checkpoints/standard-decoder.hdf5')
encoder = load_model('./checkpoints/standard-encoder.hdf5')

decoded_imgs = model.predict(x_test)

encoded_imgs = encoder.predict(x_test)

img1 = encoded_imgs[3]
img2 = encoded_imgs[9]

n_ipols = 10

def interpolations(vec1, vec2, n):
    step = 1/(n+2)
    return np.asarray(
        [
            np.asarray([x + (i * (y - x)) for (x, y) in zip(vec1, vec2)])
            for i in np.linspace(step, 1, n)
        ])

ipol_encoded = interpolations(img1, img2, n_ipols)

ipol_decoded = decoder.predict(ipol_encoded)
plt.figure(figsize=(20, 4))

cols = math.floor(math.sqrt(n_ipols))
rows = cols if cols ** 2 == n_ipols else cols + 1
for i in range(0, len(ipol_decoded)):
    ax = plt.subplot(rows, cols, 1 + i)
    plt.gray()
    plt.imshow(ipol_decoded[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

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

# # t-SNE of the latent space in 2 dimensions
#
# # Step 1: generate all data in a latent representation, and a sequence of colors for plotting.
#
# latent_representation = encoder.predict(x_train)
# colors = y_train
#
# # Step 2: t-SNE
#
# print('Calculating t-SNE')
#
# tsne = TSNE(n_components=2)
# tsne.fit(latent_representation)
# latent_representation = tsne.transform(latent_representation)
#
# print('t-SNE complete')
#
# # Step 3: scatter plot
# plt.figure(figsize=(20,4))
# plt.scatter(latent_representation[:, 0], latent_representation[:, 1], c=colors)
#
plt.show()
