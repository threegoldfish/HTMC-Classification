import argparse
import json
import hierarchy_helper as hierarchy_helper
import numpy as np

from tqdm import tqdm

def get_keywords(keyword_file):
    keywords_dict = {}
    with open(keyword_file) as file:
        for i, line in enumerate(file):
            keywords = line.split()
            for word in keywords:
                 keywords_dict[word.strip()] = i

    return keywords_dict

def get_doc_sent(data, keywords_dict):
    res = []
    keywords_set = set(keywords_dict.keys())
    for a, doc_dict in enumerate(tqdm(data['data'])):
        doc_res = [doc_dict['text'], []]
        for b, sen_dict in enumerate(doc_dict['sentences']):
            #add keywords
            cur_keywords = []
            words = sen_dict['text'].split()
            for word in words:
                if (len(word.strip()) == 0):
                     continue
                if word.strip() in keywords_set:
                    cur_keywords.append(word.strip())
            doc_res[1].append([sen_dict['text'], cur_keywords])
        res.append(doc_res)
    
    out_path = "TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/doc_sent_30k.json"
    print(f"Doc-sent list saved to {out_path}")
    with open(out_path, "w") as json_file:
        json.dump(res, json_file, indent=4, ensure_ascii=False)

    return res


def get_small(data):
    data['data'] = data['data'][0:10]
    with open("TaxoClass-dataset/Amazon-531/train/conf_small.json", 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def get_label_names(label_file):
    label_names = []
    with open(label_file) as file:
        for line in file:
            label_names.append(line.split('\t')[1].strip())
    return label_names

def get_id_map(doc_sent, keywords_dict):
    id_to_text = {}
    text_to_id = {}

    i = 0
    doc_id = -1
    #label all docs
    for doc in doc_sent:
        doc_id = doc_id + 1
        if not doc[0] in text_to_id:
            text_to_id[doc[0]] = i
        else:
            print((f"\nold id: {text_to_id[doc[0]]}, doc id: {doc_id}\n"))
            print(doc[0])
            continue
        id_to_text[i] = doc[0]
        i = i + 1

    print(i)
    #label all sentences
    for doc in doc_sent:
        for sent in doc[1]:
            if not sent[0] in text_to_id:
                text_to_id[sent[0]] = i
            else:
                continue
            id_to_text[i] = sent[0]
            i = i + 1

    print(i)
    keywords_set = set(keywords_dict.keys())
    for j, keyword in enumerate(keywords_set):
        if not keyword in text_to_id:
                text_to_id[keyword] = i
        else:
            continue
        id_to_text[i] = keyword
        i = i + 1
    print(i)

    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/text_to_id_30k.json", 'w') as file:
        json.dump(text_to_id, file, indent=4, ensure_ascii=False)
    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/id_to_text_30k.json", 'w') as file:
        json.dump(id_to_text, file, indent=4, ensure_ascii=False)

    return text_to_id, id_to_text

def flatten(conf_label):
    res = []
    for i in range(len(conf_label[0])):
        res.append(conf_label[0][i])
    for i in range(len(conf_label[1])):
        res.append(conf_label[1][i])
    for i in range(len(conf_label[2])):
        res.append(conf_label[2][i])
    return res

def format_label(data, text_to_id, label_names, keywords_dict):
    label_dict = {}
    for a, doc_dict in enumerate(data['data']):
        id = text_to_id[doc_dict['text']]

        label_dict[id] = []

        for b, sen_dict in enumerate(doc_dict['sentences']):
            id = text_to_id[sen_dict['text']]
            label_ids = flatten(sen_dict['conf_label'])
            label_dict[id] = [label_names[i] for i in label_ids]
    
    print("adding keywords")
    keywords_set = set(keywords_dict.keys())
    for i, keyword in enumerate(keywords_set):
        id = text_to_id[keyword]
        #print(id)
        label_dict[id] = [label_names[keywords_dict[keyword]]]
    
    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/label_dict_30k.json", 'w') as file:
        json.dump(label_dict, file, indent=4, ensure_ascii=False)

    return label_dict

def check_cont(text_to_id, id_to_text):
    print(f"size of two map same: {len(text_to_id)}, {len(id_to_text)}")

    prev_id = -1
    for i, key in enumerate(text_to_id):
        id = text_to_id[key]
        if id - prev_id != 1:
            print("Not cont")
            print(id, prev_id)
            return
        else:
            prev_id = id

    print("Cont")

def map_to_txt(id_to_text):
    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/id_to_text_30k.txt", 'w') as file:
        for i, key in enumerate(id_to_text):
            line = f"{key}:{id_to_text[key]}\n"
            file.write(line)
    print("File written to TaxoClass-dataset/Amazon-531/train/id_to_text.txt")

def label_to_txt(label_dict):
    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/label_dict_30k.txt", 'w') as file:
        for i, key in enumerate(label_dict):
            line = f"{key}:{' '.join(label_dict[key])}\n"
            file.write(line)
    print("File written to TaxoClass-dataset/Amazon-531/train/label_dict_30k.txt")

def construct_label_dict():
    label_dict = {}
    with open("TaxoClass-dataset/Amazon-531/train/label_dict_30k.txt") as file:
        for line in file:
            key = line.split(":")[0].strip()
            val = line.split(":")[1].strip().split(' ')
            label_dict[key] = val
    with open("TaxoClass-dataset/Amazon-531/train/format_30k_bylevel_prob/label_dict_check.json", 'w') as file:
        json.dump(label_dict, file, indent=4, ensure_ascii=False)


"""if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-file", help="Data file")
    parser.add_argument("-k", "--keyword-file", help="Keyword file")
    parser.add_argument("-l", "--label-file", help="Label file")
    args = parser.parse_args()

    with open(args.data_file, "r") as json_file:
        data = json.load(json_file)
    #get_small(data)
    keywords_dict = get_keywords(args.keyword_file)
    labels = get_label_names(args.label_file)

    doc_sent = get_doc_sent(data, keywords_dict)
    text_to_id, id_to_text = get_id_map(doc_sent, keywords_dict)
    label_dict = format_label(data, text_to_id, labels, keywords_dict)

    label_to_txt(label_dict)
    map_to_txt(id_to_text)"""

    

