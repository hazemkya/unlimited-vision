import tensorflow as tf
from models.utilities import *
import configparser

config = configparser.ConfigParser()
config.read("config.ini")
max_length = int(config['config']['max_length'])
attention_features_shape = int(config['config']['attention_features_shape'])


def predict(image, encoder, decoder,
            image_features_extract_model,
            word_to_index, index_to_word):

    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features,
                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def predict_image(image, encoder, decoder,
                  image_features_extract_model,
                  word_to_index, index_to_word):

    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims((image), 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []
    for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features,
                                         hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            result.pop
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def evaluate(image, encoder, decoder, image_features_extract_model,
             word_to_index, index_to_word):

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)

    img_tensor_val = image_features_extract_model(temp_input)

    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot
