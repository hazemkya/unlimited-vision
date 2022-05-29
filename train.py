from models.utilities import *
from models.subclasses import *
from models.train_utils import *
from models.predict import *

import configparser

print("Started training...")

config = configparser.ConfigParser()
config.read("config.ini")

max_length = int(config['config']['max_length'])
vocabulary_size = int(config['config']['vocabulary_size'])
use_glove = bool(config['config']['use_glove'])
glove_dim = int(config['config']['glove_dim'])

img_name_train, cap_train, vocabulary, train_captions = load_dataset()
dataset = make_dataset(img_name_train, cap_train)
word_to_index_train, index_to_word_train, tokenizer, cap_vector = tokenization(
    train_captions, max_length, vocabulary_size)

val_image_paths, image_path_to_caption_val = import_files(
    shuffle=False, method="val")

val_captions = []
img_name_vector_val = []
for image_path in val_image_paths:
    caption_list = image_path_to_caption_val[image_path]
    if len(caption_list) != 5:
        caption_list = caption_list[:5]
    val_captions.extend(caption_list)
    img_name_vector_val.extend([image_path] * len(caption_list))

word_to_index_val, index_to_word_val, tokenizer_val, cap_vector_val = tokenization(
    val_captions, max_length, vocabulary_size)

image_features_extract_model = get_feature_extractor()

img_name_val, cap_val = split_data(img_name_vector_val, cap_vector_val,
                                   image_features_extract_model, 1)

units = int(config['config']['units'])
embedding_dim = int(config['config']['embedding_dim'])

# Training parameters
epochs = int(config['config']['epochs'])
num_steps = len(img_name_train) // BATCH_SIZE
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
    embedding_dim = glove_dim
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


encoder = CNN_Encoder(embedding_dim)
if use_glove:
    decoder = RNN_Decoder(embedding_dim, units, num_tokens,
                          embedding_matrix)
    print(f"Vocabulary size : {num_tokens}")

else:
    decoder = RNN_Decoder(embedding_dim, units,
                          tokenizer.vocabulary_size(), None)
    print(f"Vocabulary size : {len(tokenizer.get_vocabulary())}")


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

if start_epoch == 0:
    loss_plot = []
else:
    loss_plot = load_loss()


train(epochs, start_epoch, ckpt_manager,
      num_steps, dataset, decoder,
      encoder, loss_plot, word_to_index_train,
      index_to_word_train, img_name_vector_val,
      image_features_extract_model)


print("Done training...")
