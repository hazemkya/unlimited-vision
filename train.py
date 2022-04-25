from models.utilities import *
from models.subclasses import *
from models.train_utils import *
from models.predict import *

import configparser

print("Starting...")

config = configparser.ConfigParser()
config.read("config.ini")

max_length = int(config['config']['max_length'])
vocabulary_size = int(config['config']['vocabulary_size'])
use_glove = bool(config['config']['use_glove'])

img_name_train, cap_train, vocabulary, train_captions = load_dataset()
dataset = make_dataset(img_name_train, cap_train)
word_to_index, index_to_word, tokenizer, cap_vector = tokenization(
    train_captions, max_length, vocabulary_size)

units = int(config['config']['units'])
embedding_dim = int(config['config']['embedding_dim'])

# Training parameters
epochs = int(config['config']['epochs'])
num_steps = len(img_name_train) // BATCH_SIZE
embeddings_index = {}

if use_glove:
    glove_path = "./dataset/glove.6B/glove.6B.100d.txt"

    with open(glove_path, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    vocabulary = tokenizer.get_vocabulary()
    word_index = dict(zip(vocabulary, range(len(vocabulary))))

    num_tokens = len(vocabulary) + 2
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


encoder = CNN_Encoder(embedding_dim)
if use_glove:
    decoder = RNN_Decoder(embedding_dim, units, num_tokens, embedding_matrix)
else:
    decoder = RNN_Decoder(embedding_dim, units,
                          tokenizer.vocabulary_size(), None)

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
      encoder, word_to_index, loss_plot)

print("Done...")
