
import argparse
import json
import hierarchy_helper as hierarchy_helper
import numpy as np
import format as format
from tqdm import tqdm

def get_parent_index(label, G, cani_pars):
    pars = list(G.predecessors(label))
    for par in pars:
        if par in cani_pars:
            par_index= cani_pars.index(par)
            return par_index
    
    return -1

def calculate_conf(data, G):
    all_conf = [[], [], []]
    #enumerate through all docs and calculate confidence
    for a, doc_dict in enumerate(tqdm(data['data'])):

        for b, sen_dict in enumerate(doc_dict['sentences']):
            #initialize all conf level as 0
            sen_dict["conf_val"] = [[], [], []]
            sen_dict["conf_label"] = [[], [], []]
            for i, label in enumerate(sen_dict["first_labels"]):
                sib_probs = []
                for j, other_label in enumerate(sen_dict["first_labels"]):
                    if other_label != label:
                        sib_probs.append(sen_dict["first_probs"][j])
                cur_conf = sen_dict["first_probs"][i] - max(sib_probs)
                sen_dict["conf_label"][0].append(label)
                sen_dict["conf_val"][0].append(cur_conf)
                all_conf[0].append(cur_conf)

            for i, label in enumerate(sen_dict["second_labels"]):
                #find sibling
                sib_probs = [0]
                for j, other_label in enumerate(sen_dict["second_labels"]):
                    if other_label != label and list(G.predecessors(label))[0] == list(G.predecessors(other_label))[0] and j < 3:
                        sib_probs.append(sen_dict["second_probs"][j])
                #find parent
                par_prob = 0
                par_index = get_parent_index(label, G, sen_dict["first_labels"])
                if par_index != -1:
                    par_prob = sen_dict['first_probs'][par_index]
                cur_conf = sen_dict["second_probs"][i] - max(max(sib_probs), par_prob)
                sen_dict["conf_label"][1].append(label)
                sen_dict["conf_val"][1].append(cur_conf)
                all_conf[1].append(cur_conf)

            for i, label in enumerate(sen_dict["third_labels"]):
                #find sibling
                sib_probs = [0]
                for j, other_label in enumerate(sen_dict["third_labels"]):
                    if other_label != label and list(G.predecessors(label))[0] == list(G.predecessors(other_label))[0]:
                        sib_probs.append(sen_dict["third_probs"][j])
                #find parent
                par_prob = 0
                par_index = get_parent_index(label, G, sen_dict["second_labels"])
                if par_index != -1:
                    par_prob = sen_dict['second_probs'][par_index]
                cur_conf = sen_dict["third_probs"][i] - max(max(sib_probs), par_prob)
                sen_dict["conf_label"][2].append(label)
                sen_dict["conf_val"][2].append(cur_conf)
                all_conf[2].append(cur_conf)

    
    return data, all_conf
        
def get_core_class_by_level(data, all_conf):
    first_cut = np.percentile(all_conf[0], 90)
    second_cut = np.percentile(all_conf[1], 90)
    third_cut = np.percentile(all_conf[2], 90)

    print(f"First-level cutoff confidence is {first_cut}")
    print(f"Second-level cutoff confidence is {second_cut}")
    print(f"Third-level cutoff confidence is {third_cut}")

    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            cur_conf_val = sen_dict["conf_val"][0]
            conf_index = [i for i in range(len(cur_conf_val)) if cur_conf_val[i] >= first_cut]
            sen_dict["conf_val"][0] = [sen_dict["conf_val"][0][i] for i in conf_index]
            sen_dict["conf_label"][0] = [sen_dict["conf_label"][0][i] for i in conf_index]

            cur_conf_val = sen_dict["conf_val"][1]
            conf_index = [i for i in range(len(cur_conf_val)) if cur_conf_val[i] >= second_cut]
            sen_dict["conf_val"][1] = [sen_dict["conf_val"][1][i] for i in conf_index]
            sen_dict["conf_label"][1] = [sen_dict["conf_label"][1][i] for i in conf_index]

            cur_conf_val = sen_dict["conf_val"][2]
            conf_index = [i for i in range(len(cur_conf_val)) if cur_conf_val[i] >= third_cut]
            sen_dict["conf_val"][2] = [sen_dict["conf_val"][2][i] for i in conf_index]
            sen_dict["conf_label"][2] = [sen_dict["conf_label"][2][i] for i in conf_index]

    return data

def get_core_class_by_class(data):
    #add all labels and conf values to a dict
    conf_dict = {}
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            for i in range(3):
                labels = sen_dict['conf_label'][i]
                vals = sen_dict['conf_val'][i]
                for j in range(len(labels)):
                    if labels[j] not in conf_dict:
                        conf_dict[labels[j]] = [vals[j]]
                    else:
                        conf_dict[labels[j]].append(vals[j])

    #calcualte 10 percentile for each class
    for i, key in enumerate(conf_dict):
        conf_dict[key] = np.percentile(conf_dict[key], 90)

    #filter and add
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            for i in range(3):
                final_label = []
                final_val = []

                cur_label = sen_dict['conf_label'][i]
                cur_val = sen_dict['conf_val'] [i]
                for j in range(len(cur_label)):
                    if cur_val[j] >= conf_dict[cur_label[j]]:
                        final_label.append(cur_label[j])
                        final_val.append(cur_val[j])
                sen_dict['conf_label'][i] = final_label
                cur_val = sen_dict['conf_val'] [i] = final_val

    return data

def get_core_class_by_class_prob(data):
    #add all labels and conf values to a dict
    prob_dict = {}
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            label_keys = ['first_labels', 'second_labels', 'third_labels']
            label_vals = ['first_probs', 'second_probs', 'third_probs']
            for i in range(3):
                labels = sen_dict[label_keys[i]]
                vals = sen_dict[label_vals[i]]
                for j in range(len(labels)):
                    if labels[j] not in prob_dict:
                        prob_dict[labels[j]] = [vals[j]]
                    else:
                        prob_dict[labels[j]].append(vals[j])

    #calcualte 10 percentile for each class
    for i, key in enumerate(prob_dict):
        prob_dict[key] = np.percentile(prob_dict[key], 90)

    #filter and add
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            label_keys = ['first_labels', 'second_labels', 'third_labels']
            label_vals = ['first_probs', 'second_probs', 'third_probs']
            for i in range(3):
                final_label = []
                final_val = []

                cur_label = sen_dict[label_keys[i]]
                cur_val = sen_dict[label_vals[i]]
                for j in range(len(cur_label)):
                    if cur_val[j] >= prob_dict[cur_label[j]]:
                        final_label.append(cur_label[j])
                        final_val.append(cur_val[j])
                sen_dict['conf_label'][i] = final_label
                cur_val = sen_dict['conf_val'] [i] = final_val

    return data

def get_core_class_by_level_prob(data):
    #add all labels and conf values to a dict
    level_prob = [[], [], []]
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            label_keys = ['first_labels', 'second_labels', 'third_labels']
            label_vals = ['first_probs', 'second_probs', 'third_probs']
            for i in range(3):
                labels = sen_dict[label_keys[i]]
                vals = sen_dict[label_vals[i]]
                for j in range(len(labels)):
                    level_prob[i].extend(vals)

    #calcualte 10 percentile for each class
    for i in range(3):
        level_prob[i] = np.percentile(level_prob[i], 90)
        print(f"Cutoff at level {i} is {level_prob[i]}")

    #filter and add
    for a, doc_dict in enumerate(tqdm(data['data'])):
        for b, sen_dict in enumerate(doc_dict['sentences']):
            label_keys = ['first_labels', 'second_labels', 'third_labels']
            label_vals = ['first_probs', 'second_probs', 'third_probs']
            for i in range(3):
                final_label = []
                final_val = []

                cur_label = sen_dict[label_keys[i]]
                cur_val = sen_dict[label_vals[i]]
                for j in range(len(cur_label)):
                    if cur_val[j] >= level_prob[i]:
                        final_label.append(cur_label[j])
                        final_val.append(cur_val[j])
                sen_dict['conf_label'][i] = final_label
                cur_val = sen_dict['conf_val'] [i] = final_val

    return data

def merge():
    with open("./TaxoClass-dataset/Amazon-531/train/probs_10000_1.json", "r") as json_file:
            data1 = json.load(json_file)
    with open("./TaxoClass-dataset/Amazon-531/train/probs_10000_2.json", "r") as json_file:
            data2 = json.load(json_file)
    with open("./TaxoClass-dataset/Amazon-531/train/probs_10000_3.json", "r") as json_file:
            data3 = json.load(json_file)

    data1['data'].extend(data2['data'])
    data1['data'].extend(data3['data'])

    return data1

#develop a visually easy format
def get_check(conf_data, labels):
    for a, doc_dict in enumerate(tqdm(conf_data['data'])):
        doc_label = doc_dict['label']
        doc_name = [labels[i] for i in doc_label]
        for b, sen_dict in enumerate(doc_dict['sentences']):
            sen_dict['doc_label'] = doc_label
            sen_dict['doc_name'] = doc_name

            conf_label = sen_dict['conf_label']
            conf_name = [[], [], []]
            for i in range(3):
                conf_name[i].extend([labels[num] for num in conf_label[i]])
            sen_dict['conf_name'] = conf_name

    return conf_data
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data-file", help="Data file")
    parser.add_argument("-o", "--out-file", help="Pseudo label output")
    parser.add_argument("-m", "--merge", help="Data file", default=0)
    parser.add_argument("-k", "--keyword-file", help="Keyword file")
    parser.add_argument("-l", "--label-file", help="Label file")
    args = parser.parse_args()

    if int(args.merge) == 0:
        with open(args.data_file, "r") as json_file:
            data = json.load(json_file)
    else:
        data = merge()

    #read in hierarchy info
    G = hierarchy_helper.create_label_graph("./TaxoClass-dataset/Amazon-531/label_hierarchy.txt", 531)
    labels = hierarchy_helper.get_labels("./TaxoClass-dataset/Amazon-531/train/labels.txt")

    #calculate confidence
    conf_data, all_conf = calculate_conf(data, G)
    final_data = get_core_class_by_level(conf_data, all_conf)

    """with open(args.out_file, "w") as wf:
        json.dump(final_data, wf, indent=4, ensure_ascii=False)
    print(f"Core class data written to {args.out_file}")"""

    #this is format content
    keywords_dict = format.get_keywords(args.keyword_file)
    labels = format.get_label_names(args.label_file)

    doc_sent = format.get_doc_sent(final_data, keywords_dict)
    text_to_id, id_to_text = format.get_id_map(doc_sent, keywords_dict)
    label_dict = format.format_label(final_data, text_to_id, labels, keywords_dict)
    
    format.map_to_txt(id_to_text)
    format.label_to_txt(label_dict)
    
    """check_dict = get_check(final_data, labels)
    
    with open("./TaxoClass-dataset/Amazon-531/train/check/check_bylevel_conf.json", "w") as wf:
        json.dump(check_dict, wf, indent=4, ensure_ascii=False)"""