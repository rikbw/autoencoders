from keras.datasets import mnist
import numpy as np
from tempfile import TemporaryFile
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images
x_train = x_train.reshape((len(x_train), -1))
x_train = x_train.astype('float32') / 255.  # Convert to floating point
x_test = x_test.reshape((len(x_test), -1))
x_test = x_test.astype('float32') / 255.

noisy_example = np.asarray([max(min(pix + np.random.normal(0, 0.35), 1), 0) for pix in x_train[1]])
plt.figure()
plt.imshow(noisy_example.reshape(28, 28))
plt.gray()

plt.show()

train = 'mnist_noisy_35.npy'
test = 'mnist_noisy_test_35.npy'

# Add noise
print("Adding noise to training data")
noisy_train = np.asarray([[max(min(pix + np.random.normal(0, 0.35), 1), 0) for pix in x] for x in x_train])
print("Saving noisy training data")
np.save(train, noisy_train)
print("Adding noise to test data")
noisy_test = np.asarray([[max(min(pix + np.random.normal(0, 0.35), 1), 0) for pix in x] for x in x_test])
print("Saving noisy test data")
np.save(test, noisy_test)