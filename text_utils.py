import os
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization


def load_captions_data(filename, SEQ_LENGTH, IMAGES_PATH):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.split("\t")

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            img_name = os.path.join(
                IMAGES_PATH, img_name.strip()).split('\\')[1]

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # We will add a start and an end token to each caption
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data


def custom_standardization(input_string):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


def text_vectorization(VOCAB_SIZE, SEQ_LENGTH, text_data):
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    # Data augmentation for image data
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )
    return vectorization, image_augmentation


def decode_and_resize(IMAGES_PATH, IMAGE_SIZE):
    img = tf.io.read_file(IMAGES_PATH)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(IMAGES_PATH, captions, vectorization, IMAGE_SIZE):
    return decode_and_resize(IMAGES_PATH, IMAGE_SIZE), vectorization(captions)


def make_dataset(images, captions, AUTOTUNE, BATCH_SIZE, vectorization):
    if split == "train":
        img_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            read_train_image, num_parallel_calls=AUTOTUNE
        )
    else:
        img_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            read_valid_image, num_parallel_calls=AUTOTUNE
        )

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(
        vectorization, num_parallel_calls=AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(AUTOTUNE)
    return dataset


def preprocess_text(VOCAB_SIZE, SEQ_LENGTH, text_data, AUTOTUNE, train_data, valid_data, BATCH_SIZE):

    vectorization, image_augmentation = text_vectorization(
        VOCAB_SIZE, SEQ_LENGTH, text_data)

    # Pass the list of images and the list of corresponding captions
    train_dataset = make_dataset(list(train_data.keys()), list(
        train_data.values()), AUTOTUNE, BATCH_SIZE, vectorization)

    valid_dataset = make_dataset(list(valid_data.keys()), list(
        valid_data.values()), AUTOTUNE, BATCH_SIZE, vectorization)

    print("Done preprocessing text.")