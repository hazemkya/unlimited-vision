from numpy import float32
import tensorflow as tf
import time
import pickle
import configparser
from models.evaluation_utils import *
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

config = configparser.ConfigParser()
config.read("config.ini")

save_path = config["config"]["save_path"]
early_stop = int(config["config"]["early_stop"])


@tf.function
def train_step(img_tensor, target, decoder, encoder, word_to_index):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train(epochs, start_epoch, ckpt_manager,
          num_steps, dataset, decoder,
          encoder, loss_plot, word_to_index_train,
          index_to_word_train, img_name_vector_val,
          image_features_extract_model):

    EPOCHS = epochs
    total_time = 0
    no_change_since = 0

    # if loss_plot:
    #     last_save = loss_plot[-1]
    # else:
    #     last_save = float('inf')

    best_score = 0

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0
        current_average_batch_loss = 0
        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, decoder,
                                            encoder, word_to_index_train)
            total_loss += t_loss

            if batch % 100 == 0:
                current_average_batch_loss = batch_loss.numpy() / \
                    int(target.shape[1])
                print(
                    f'Epoch {epoch+1} Batch {batch} Loss {current_average_batch_loss}')

        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        print(f'Epoch {epoch+1} Loss {float(total_loss/num_steps)}',
              "last save: ", float(best_score))

        bleu_curr = evaluate_epoch(img_name_vector_val, encoder, decoder,
                                   image_features_extract_model, word_to_index_train,
                                   index_to_word_train, best_score)

        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
        total_time += time.time()-start

        # save a checkpoint if the loss is better than the last saved loss
        if (bleu_curr < best_score):
            no_change_since = 0
            ckpt_manager.save()
            save_loss(loss_plot)
            print("Chekpoint autosave current: ",
                  float(bleu_curr), "last save: ", float(best_score))
            best_score = bleu_curr
        else:
            no_change_since += 1
            print(f"No loss gained since {no_change_since}")

        if no_change_since >= early_stop:
            print(
                f"Ending training because no loss has been gained for {no_change_since} epochs.")
            break

    print(f'Total time taken: {(total_time/60):.2f} min\n')

    return loss_plot


def evaluate_epoch(img_name_vector_val, encoder, decoder,
                   image_features_extract_model, word_to_index_train,
                   index_to_word_train, best_score):

    annotation_file = 'dataset\coco\\val\captions_val2017.json'
    results_file = 'dataset\coco\\result\\temp_result.json'

    result_a, ImgIDs = makeResultFile(img_name_vector_val, encoder, decoder,
                                      image_features_extract_model, word_to_index_train,
                                      index_to_word_train, percentage=0.1)
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result, ImgIDs)

    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        if metric == 'Bleu_1':
            if score > best_score:
                bleu_curr = score

    print(f'{metric}: {score:.3f}')

    return bleu_curr


def save_loss(loss_plot):
    pickle.dump(loss_plot, open(f"{save_path}dataset/loss_plot", "wb"))


def load_loss():
    loss_plot = pickle.load(open(f'{save_path}dataset/loss_plot', 'rb'))
    return loss_plot
