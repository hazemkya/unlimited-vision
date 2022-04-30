from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from models.predict import *


def evaluate_all(num_of_samples, list_of_references, list_of_hypotheses):
    bleu_score(num_of_samples, list_of_references, list_of_hypotheses)
    meteor_eval(num_of_samples, list_of_references, list_of_hypotheses)
    rouge_score(num_of_samples, list_of_references, list_of_hypotheses)


def bleu_score(num_of_samples, list_of_references, list_of_hypotheses):
    avg_BLEU1 = 0
    avg_BLEU2 = 0
    avg_BLEU3 = 0
    avg_BLEU4 = 0
    for i in range(int(num_of_samples/5)):
        test_references = []
        for j in range(5):
            test_references.append(list_of_references[i][j].split())
        # for j in range(5):
        avg_BLEU1 += corpus_bleu([test_references],
                                 [list_of_hypotheses[i].split()], weights=(1.0, 0, 0, 0))
        avg_BLEU2 += corpus_bleu([test_references],
                                 [list_of_hypotheses[i].split()], weights=(0.5, 0.5, 0, 0))
        avg_BLEU3 += corpus_bleu([test_references],
                                 [list_of_hypotheses[i].split()], weights=(0.3, 0.3, 0.3, 0))
        avg_BLEU4 += corpus_bleu([test_references],
                                 [list_of_hypotheses[i].split()], weights=(0.25, 0.25, 0.25, 0.25))

    avg_BLEU1 /= num_of_samples/5
    avg_BLEU2 /= num_of_samples/5
    avg_BLEU3 /= num_of_samples/5
    avg_BLEU4 /= num_of_samples/5

    print('Average BLEU-1: %f' % avg_BLEU1)
    print('Average BLEU-2: %f' % avg_BLEU2)
    print('Average BLEU-3: %f' % avg_BLEU3)
    print('Average BLEU-4: %f' % avg_BLEU4)
    print("")


def meteor_eval(num_of_samples, list_of_references, list_of_hypotheses):
    avg_meteor = 0.0
    for i in range(int(num_of_samples/5)):
        for j in range(5):
            avg_meteor += meteor_score([list_of_references[i][j].split()],
                                       list_of_hypotheses[i].split())
    avg_meteor /= num_of_samples
    print("Average Meteor score: ", avg_meteor)
    print("")


def rouge_score(num_of_samples, list_of_references, list_of_hypotheses):
    rouge = Rouge()
    avg_rouge1_r = 0
    avg_rouge1_p = 0
    avg_rouge1_f = 0
    avg_rouge2_r = 0
    avg_rouge2_p = 0
    avg_rouge2_f = 0
    avg_rougel_r = 0
    avg_rougel_p = 0
    avg_rougel_f = 0
    for i in range(int(num_of_samples/5)):
        for j in range(5):
            result = rouge.get_scores(
                refs=list_of_references[i][j], hyps=list_of_hypotheses[i], avg=True)
            avg_rouge1_r += result['rouge-1']['r']
            avg_rouge1_p += result['rouge-1']['p']
            avg_rouge1_f += result['rouge-1']['f']
            avg_rouge2_r += result['rouge-2']['r']
            avg_rouge2_p += result['rouge-2']['p']
            avg_rouge2_f += result['rouge-2']['f']
            avg_rougel_r += result['rouge-l']['r']
            avg_rougel_p += result['rouge-l']['p']
            avg_rougel_f += result['rouge-l']['f']

    avg_rouge1_r /= num_of_samples
    avg_rouge1_p /= num_of_samples
    avg_rouge1_f /= num_of_samples
    avg_rouge2_r /= num_of_samples
    avg_rouge2_p /= num_of_samples
    avg_rouge2_f /= num_of_samples
    avg_rougel_r /= num_of_samples
    avg_rougel_p /= num_of_samples
    avg_rougel_f /= num_of_samples

    print(
        f'Average Rouge-1 recall: {avg_rouge1_r}\nAverage Rouge-1 precision: {avg_rouge1_p}\nAverage Rouge-1 f1_score: {avg_rouge1_f}\n')
    print(
        f'Average Rouge-2 recall: {avg_rouge2_r}\nAverage Rouge-2 precision: {avg_rouge2_p}\nAverage Rouge-2 f1_score: {avg_rouge2_f}\n')
    print(
        f'Average Rouge-l recall: {avg_rougel_r}\nAverage Rouge-l precision: {avg_rougel_p}\nAverage Rouge-l f1_score: {avg_rougel_f}\n')


def getHypotheses(img_name_val, encoder, decoder,
                  image_features_extract_model, word_to_index_train,
                  index_to_word_train):

    list_of_hypotheses = []
    for i in range(0, len(img_name_val), 5):
        result = predict(img_name_val[i], encoder, decoder, image_features_extract_model,
                         word_to_index_train, index_to_word_train)
        if result[-1] == "<end>":
            result.remove("<end>")

        list_of_hypotheses.append(' '.join(result))
    return list_of_hypotheses
