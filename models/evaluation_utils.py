from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from models.predict import *


def makeResultFile(img_name_vector_val, encoder, decoder,
                   image_features_extract_model, word_to_index_train,
                   index_to_word_train, percentage=1):

    result_list = []
    ImgIDs = []
    samples = (int)(len(img_name_vector_val) * percentage)

    for i in range(0, samples, 5):
        id = img_name_vector_val[i].split("val2017\\")[1].split(".")[0]
        cap = predict(img_name_vector_val[i], encoder, decoder, image_features_extract_model,
                      word_to_index_train, index_to_word_train)

        if cap[-1] == "<end>":
            cap.remove("<end>")

        cap = ' '.join(cap)
        imid = int(id.lstrip('0'))
        temp = {"image_id": imid, "caption": cap}

        result_list.append(temp)
        ImgIDs.append(imid)

    with open('dataset\coco\\result\\result.json', 'w') as outfile:
        json.dump(result_list, outfile, sort_keys=True)

    return result_list, ImgIDs
