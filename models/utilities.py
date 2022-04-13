import os
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import collections
import random
from tqdm import tqdm
import pickle


import configparser

config = configparser.ConfigParser()
config.read("config.ini")

sample = int(config["config"]["sample"])
save_path = config["config"]["save_path"]

BATCH_SIZE = int(config['config']['BATCH_SIZE'])
BUFFER_SIZE = int(config['config']['BUFFER_SIZE'])


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


def get_feature_extractor():
    image_model = tf.keras.applications.resnet50.ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    # freeze layers

    for layer in image_features_extract_model.layers[:]:
        layer.trainable = False

    return image_features_extract_model


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


def index_vocab(vocabulary):
    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary)
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary,
        invert=True)

    return word_to_index, index_to_word


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

    word_to_index, index_to_word = index_vocab(tokenizer.get_vocabulary())

    tokens_shape = word_to_index('<start>')

    return word_to_index, index_to_word, tokenizer, cap_vector


def split_data(img_name_vector, cap_vector, image_features_extract_model, percentage=0.8, ):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    if shuffle:
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

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

    return img_name_train, cap_train, img_name_val, cap_val


def make_dataset(img_name_train, cap_train):
    # Load the numpy files
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8')+'.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int64]),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    if shuffle:
        dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def save_dataset(img_name_train, cap_train, img_name_val, cap_val, vocabulary):
    pickle.dump(img_name_train, open(
        f"{save_path}dataset/img_name_train", "wb"))
    pickle.dump(cap_train, open(f"{save_path}dataset/cap_train", "wb"))
    pickle.dump(img_name_val, open(f"{save_path}dataset/img_name_val", "wb"))
    pickle.dump(cap_val, open(f"{save_path}dataset/cap_val", "wb"))
    pickle.dump(vocabulary, open(f"{save_path}dataset/vocabulary", "wb"))


def load_dataset():
    img_name_train = pickle.load(
        open(f'{save_path}dataset/img_name_train', 'rb'))
    cap_train = pickle.load(open(f"{save_path}dataset/cap_train", "rb"))
    img_name_val = pickle.load(open(f"{save_path}dataset/img_name_val", "rb"))
    cap_val = pickle.load(open(f"{save_path}dataset/cap_val", "rb"))
    vocabulary = pickle.load(open(f"{save_path}dataset/vocabulary", "rb"))
    return img_name_train, cap_train, img_name_val, cap_val, vocabulary


def save_models(encoder, decoder, image_features_extract_model):
    encoder.save(f"{save_path}models/encoder")
    decoder.save(f"{save_path}models/decoder")
    image_features_extract_model.save(f"{save_path}models/feature_extractor")


def load_models():
    encoder = tf.keras.models.load_model(f"{save_path}models/encoder")
    decoder = tf.keras.models.load_model(f"{save_path}models/decoder")
    image_features_extract_model = tf.keras.models.load_model(
        f"{save_path}models/feature_extractor")

    return encoder, decoder, image_features_extract_model
