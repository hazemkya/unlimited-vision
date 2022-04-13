import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import collections
import random

import configparser

config = configparser.ConfigParser()
config.read("config.ini")

sample = int(config["config"]["sample"])


def import_files(shuffle):

    annotation_file = 'dataset\coco\\tarin\captions_train2017.json'
    annotation_folder = '\dataset\coco\\annotations_trainval2017\\annotations\\'

    image_folder = '\dataset\coco\\tarin\images\\'
    PATH = os.path.abspath('.') + image_folder

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + '%012d.jpg' % (val['image_id'])
        # image_path.replace("\\","/") #windows
        image_path_to_caption[image_path].append(caption)

    # image_path_to_caption = [item.replace("\\","/") for item in image_path_to_caption] #windows
    image_paths = list(image_path_to_caption.keys())

    if shuffle:
        random.shuffle(image_paths)

    # Select the first sample image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    train_image_paths = image_paths[:sample]

    # train_image_paths = [item.replace("\\","/") for item in train_image_paths] #windows

    print(len(train_image_paths))
    # Download caption annotation files

    return train_image_paths, image_path_to_caption


def load_image(image_path, format='jpeg'):
    # load and pre-process an image
    img = tf.io.read_file(image_path)

    if format == "jpeg":
        img = tf.io.decode_jpeg(img, channels=3)
    elif format == "png":
        img = tf.io.decode_png(img, channels=3)

    img = tf.keras.layers.Resizing(224, 224)(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    return img, image_path


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def tokenization(train_captions, max_length, vocabulary_size):
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    # We will override the default standardization of TextVectorization to preserve
    # "<>" characters, so we preserve the tokens for the <start> and <end>.
    def standardize(inputs):
        inputs = tf.strings.lower(inputs)
        return tf.strings.regex_replace(inputs,
                                        r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length)
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(caption_dataset)

    # Create the tokenized vectors
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))

    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)
    return word_to_index, index_to_word, tokenizer, cap_vector


def split_data(img_name_vector, cap_vector, percentage=0.8):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*percentage)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    return img_name_train, cap_train, img_name_val, cap_val
