import numpy as np
import random
import string
from keras.layers import Dense, Input
from keras.models import Model
import matplotlib.pyplot as plt


def encode_embedding(str):
    str_length = len(str)
    alphabet = list(string.ascii_lowercase)
    embedding = [0] *  str_length * (len(alphabet))
    #embedding = np.zeros(1 , str_length * (len(alphabet)))

    for i in range(0, str_length):
        letter = str[i]
        embedding[alphabet.index(letter) + i*len(alphabet)] = 1

    return embedding

def decode_embedding(emb):

    alphabet = list(string.ascii_lowercase)
    alphabet_size = len(alphabet)
    str_length = int(len(emb)/alphabet_size)
    word = ""

    for str_index in range(0,str_length):
        for alpha_index in range(0,alphabet_size):
            if emb[str_index*alphabet_size + alpha_index] == 1:
                word += alphabet[alpha_index]
    return word

def load_words(filename):
    with open(filename , 'r') as myfile:
        data= myfile.read()
        data = data.replace("\n","").split(", ")
        print( "Found " + str(len(data)) + " words. ")
    return data

def add_noise(emb,str_list,nb_letters):
    original_emb = emb

    alphabet = list(string.ascii_lowercase)
    alphabet_size = len(alphabet)
    str_length = int(len(emb)/alphabet_size)

    for i in range(0,nb_letters):
        random_ind = random.randint(0,str_length-1)
        emb[random_ind*alphabet_size :(random_ind+1)*alphabet_size] = [0] * alphabet_size
        random_letter = random.randint(0,alphabet_size-1)
        emb[random_ind*alphabet_size + random_letter] = 1
    #If the altered word is another word in the dataset, Try again
    if decode_embedding(emb) in str_list:
        return add_noise(original_emb , str_list , nb_letters)
    else:
        return emb

def create_dataset(file,nb_repeats):
    # Load all the words of the dataset
    str_list = load_words(file)
    x_train_denoised = [encode_embedding(word) for word in str_list for i in range(0, nb_repeats)]

    return x_train_denoised
def create_noisy_dataset(file,nb_repeats,nb_letters):
    # Load all the words of the dataset
    str_list = load_words(file)
    x_train_denoised = [encode_embedding(word) for word in str_list for i in range(0,nb_repeats)]

    x_translated = [decode_embedding(word) for word in x_train_denoised]
    print("Size training set: " + str(len(x_train_denoised)))

    # Adjust one letter in each word to create a noisy dataset
    x_train_noisy = [add_noise(sample,str_list,nb_letters) for sample in x_train_denoised]
    return x_train_noisy

def transform_output(emb):
    emb = emb.tolist()
    alphabet = list(string.ascii_lowercase)
    alphabet_size = len(alphabet)

    str_length = int(len(emb) / alphabet_size)

    for str_index in range(0,str_length):
        max_index = emb.index(max(emb[str_index*alphabet_size :(str_index+1)*alphabet_size]))
        emb[str_index * alphabet_size:(str_index + 1) * alphabet_size] = [0] * alphabet_size
        emb[max_index] = 1
    return emb

def calculate_acc(X_normal,X_predicted):
    correct = 0
    items = len(X_normal)
    for index in range(0,len(X_normal)):
        if (X_normal[index] == X_predicted[index]):
            correct += 1
    return correct/items



file = '1000_14letterwords.txt'
X_normal = create_dataset(file, 100)
X_noisy  = create_noisy_dataset(file, 100, 4)

epochs = 100
latent_size = 64
size_words = 14
embedding_size = size_words*26

input_word = Input(shape=(embedding_size,)) # Input placeholder
# Tensor of encoded input
encoded = Dense(latent_size, activation='relu')(input_word)
# Tensor of decoded input
decoded = Dense(embedding_size, activation='relu')(encoded)

autoencoder = Model(input_word, decoded)

# TODO test metrics
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Train

autoencoder.fit(np.array(X_noisy), np.array(X_normal), epochs=epochs, shuffle=True, batch_size=128)


decoded_words = autoencoder.predict(np.array(X_noisy))

letters = [1,2,4,8]

x_test_normal = create_dataset(file, 1)
for nb_letter in letters:
    X_test_noisy = create_noisy_dataset(file, 1, nb_letter)
    X_denoised = autoencoder.predict(np.array(X_test_noisy))
    X_translated = [transform_output(emb) for emb in X_denoised]
    print([decode_embedding(sample) for sample in X_translated])
    print("Accuracy with changing " + str(nb_letter) + " letter: " + str(calculate_acc(x_test_normal, X_translated)))
