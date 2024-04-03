import argparse
import json
import hierarchy_helper as hierarchy_helper
import numpy as np

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def topic_verbalizer_text(category):
    hypothesis = f"This text is about {category.lower()}."
    return hypothesis

def topic_verbalizer_word(category):
    hypothesis = f"This word is about {category.lower()}."
    return hypothesis


def classify_entailment(model, tokenizer, categories, verbalizer, text):
    hypotheses = [(text, verbalizer(category)) for category in categories]
    tokenized = tokenizer(
        hypotheses,
        truncation="only_first",
        padding="longest",
        return_tensors="pt",
    )
    logits = model(**tokenized.to(model.device))[0]
    logits_label_is_true = logits[:, 2]
    max_index = torch.argmax(logits_label_is_true)
    category = categories[max_index]
    probs = torch.softmax(logits_label_is_true, dim=0)
    return max_index, probs.tolist()


def truncate_text_to_word_limit(text, word_limit):
    words = text.split()
    if len(words) > word_limit:
        truncated_text = ' '.join(words[:word_limit])
        return truncated_text
    else:
        return text


if __name__ == "__main__":

    # Example:
    #
    # python scripts/emnlp22/create_pseudo_labels.py \
    #     -d data/20News/data.json \
    #     -o data/20News/preds_nsp.json \
    #     -v topic \
    #     -m nsp

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-file", help="Data file")
    parser.add_argument("-o", "--out-file", help="Pseudo label output")
    parser.add_argument("-c", "--check-file", help="Check point output")
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("-s", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-dv", "--device", help="Cuda device number", default=0)
    parser.add_argument("-sl", "--start_level", help="Start from which level", default=0)
    parser.add_argument("-si", "--start_index", help="Start from which index (of level)", default=0)
    args = parser.parse_args()

    #define entailment model
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    classify = classify_entailment

    if (int(args.device) == 0):
        model.to("cpu")
    else: 
        print(torch.cuda.device_count())
        num = int(args.device) - 1
        model.to(f"cuda:{num}")

    with open(args.data_file, "r") as json_file:
        data = json.load(json_file)
    print(f"{len(data['classes'])} Classes")

    #TODO: give a rough prediction?
    #correct = total = 0
    #samples = []

    #select sample core class for first level
    G = hierarchy_helper.create_label_graph("./TaxoClass-dataset/Amazon-531/label_hierarchy.txt", 20)
    first_level = hierarchy_helper.get_first_level(G)
    labels = hierarchy_helper.get_labels("./TaxoClass-dataset/Amazon-531/train/labels.txt")
    print(first_level)
    first_classes = [labels[i] for i in first_level]
    print(first_classes)

    if int(args.start_level) <= 0: 
        print("Level 0 Candidate Core Class Selection")
        for i, doc_dict in enumerate(tqdm(data['data'])):
            if i < int(args.start_index) and int(args.start_level) == 0:
                continue

            for j, sen_dict in enumerate(doc_dict['sentences']):
                #classify sentence
                sen_text = truncate_text_to_word_limit(sen_dict['text'], 100)
                sen_prediction, sen_probs = classify(model, tokenizer, first_classes, topic_verbalizer_text, sen_text)
                sen_dict['first_labels'] = [first_level[i] for i in np.argsort(sen_probs)[-2:]]
                sen_dict['first_childs'] = hierarchy_helper.get_childs(G, sen_dict['first_labels'], level = 0)
                sen_dict['first_probs'] = [sen_probs[i] for i in np.argsort(sen_probs)[-2:]]

            if i % 1000 == 0:
                print(f"saved to {i} to check point")
                with open(args.check_file, "w") as wf:
                    json.dump(data, wf, indent=4, ensure_ascii=False)

        with open(args.out_file, "w") as wf:
            json.dump(data, wf, indent=4, ensure_ascii=False)
        
    
    if int(args.start_level) <= 1:
        print("Level 1 Candidate Core Class Selection")
        for i, doc_dict in enumerate(tqdm(data['data'])):
            if i < int(args.start_index) and int(args.start_level) == 1:
                continue

            for j, sen_dict in enumerate(doc_dict['sentences']):
                #classify sentence
                sen_text = truncate_text_to_word_limit(sen_dict['text'], 100)
                second_classes = [labels[index] for index in sen_dict["first_childs"]]
                sen_prediction, sen_probs = classify(model, tokenizer, second_classes, topic_verbalizer_text, sen_text)
                sen_dict['second_labels'] = [sen_dict['first_childs'][t] for t in np.argsort(sen_probs)[-3:]]
                sen_dict['second_childs'] = hierarchy_helper.get_childs(G, sen_dict['second_labels'], level = 1)
                sen_dict['second_probs'] = [sen_probs[i] for i in np.argsort(sen_probs)[-3:]]
            
            if i % 1000 == 0:
                print(f"saved to {i} to check point")
                with open(args.check_file, "w") as wf:
                    json.dump(data, wf, indent=4, ensure_ascii=False)

            for j, sen_dict in enumerate(doc_dict['sentences']): 
                sen_dict.pop('first_childs')

        with open(args.out_file, "w") as wf:
            json.dump(data, wf, indent=4, ensure_ascii=False)

    if int(args.start_level) <= 2:
        print("Level 2 Candidate Core Class Selection")
        for i, doc_dict in enumerate(tqdm(data['data'])):
            if i < int(args.start_index) and int(args.start_level) == 2:
                continue

            for j, sen_dict in enumerate(doc_dict['sentences']):
                #classify sentence
                sen_text = truncate_text_to_word_limit(sen_dict['text'], 100)
                third_classes = [labels[index] for index in sen_dict["second_childs"]]
                sen_prediction, sen_probs = classify(model, tokenizer, third_classes, topic_verbalizer_text, sen_text)
                sen_dict['third_labels'] = [sen_dict['second_childs'][t] for t in np.argsort(sen_probs)[-4:]]
                sen_dict['third_probs'] = [sen_probs[i] for i in np.argsort(sen_probs)[-4:]]

            if i % 1000 == 0:
                print(f"saved to {i} to check point")
                with open(args.check_file, "w") as wf:
                    json.dump(data, wf, indent=4, ensure_ascii=False)

            for j, sen_dict in enumerate(doc_dict['sentences']):
                sen_dict.pop('second_childs')
                
        #print(f"Stats for {args.model_type}, {args.out_file}: {correct / total}")

        with open(args.out_file, "w") as wf:
            json.dump(data, wf, indent=4, ensure_ascii=False)
