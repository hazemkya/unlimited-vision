from models.predict import *
from models.train_utils import *
from models.utilities import *
from models.subclasses import *
import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import configparser

print("Starting...")

config = configparser.ConfigParser()
config.read("config.ini")

# importing local module

# Train sample size (-1 for max)
# can't exceed 118286 sample
sample = int(config['config']['train_sample'])

# train split percentage 80-20
percentage = float(config['config']['percentage'])

# Max word count for a caption.
max_length = int(config['config']['max_length'])
# Use the top words for a vocabulary.
vocabulary_size = int(config['config']['vocabulary_size'])
use_glove = bool(config['config']['use_glove'])
glove_dim = int(config['config']['glove_dim'])

# create data lists
# import data and save it to a dict, also save it's keys in a list
train_image_paths, image_path_to_caption = import_files(
    shuffle=False, method="train")

train_captions = []
img_name_vector = []
for image_path in train_image_paths:
    caption_list = image_path_to_caption[image_path]
    if len(caption_list) != 5:
        caption_list = caption_list[:5]
    train_captions.extend(caption_list)
    img_name_vector.extend([image_path] * len(caption_list))

# print(train_captions[0])
# Image.open(img_name_vector[0])

# create and freeze feature extractor model
image_features_extract_model = get_feature_extractor()

word_to_index, index_to_word, tokenizer, cap_vector = tokenization(
    train_captions, max_length, vocabulary_size)

embeddings_index = {}

if use_glove:
    new_glove_path = f"./dataset/glove.6B/new_glove.6B.{glove_dim}d.pkl"
    tuned_glove = pickle.load(open(new_glove_path, "rb"))
    len(tuned_glove)

    glove_path = f"./dataset/glove.6B/glove.6B.{glove_dim}d.txt"

    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    embeddings_index.update(tuned_glove)

    print("Found %s word vectors." % len(embeddings_index))

    vocabulary = tokenizer.get_vocabulary()
    word_index = dict(zip(vocabulary, range(len(vocabulary))))

    num_tokens = len(vocabulary)
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

img_name_train, cap_train = split_data(
    img_name_vector, cap_vector, image_features_extract_model,  percentage)

save_dataset(img_name_train, cap_train,
             tokenizer.get_vocabulary(), train_captions)

print("Done..")
