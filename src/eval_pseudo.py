import argparse
import json
import hierarchy_helper as hierarchy_helper
import numpy as np
import statistics

from tqdm import tqdm


def example_f1(data):
    f1s = []
    max_f1s = []
    for a, doc_dict in enumerate(tqdm(data['data'])):
        cur_label = doc_dict['label']
        cur_f1s = []
        for b, sen_dict in enumerate(doc_dict['sentences']):
            cur_pred = sen_dict['conf_label']
            cur_f1 = len([i for i in cur_pred if (i in cur_label)])/(len(cur_label) + len(cur_pred))
            cur_f1s.append(cur_f1)
        if len(cur_f1s) == 0:
            continue
        f1s.append(np.mean(cur_f1s))
        max_f1s.append(max(cur_f1s))
    
    print(f"Average of max example-f1 among sentences: {np.mean(max_f1s)}")
    print(f"Average of mean example-f1 among sentences: {np.mean(f1s)}\n")


def level_info_preconf(data):
    first_prob = []
    second_prob = []
    third_prob = []
    for a, doc_dict in enumerate(tqdm(data['data'])):
        cur_label = doc_dict['label']
        sent_num = len(doc_dict['sentences'])

        first_correct = 0
        second_correct = 0
        third_correct = 0
        for b, sen_dict in enumerate(doc_dict['sentences']): 
            
            for i, label in enumerate(sen_dict["first_labels"]):
                if label in cur_label:
                    first_correct  = first_correct + 1
                    break   

            for i, label in enumerate(sen_dict["second_labels"]):
                if label in cur_label:
                    second_correct = second_correct + 1
                    break   

            for i, label in enumerate(sen_dict["third_labels"]):
                if label in cur_label:
                    third_correct = third_correct + 1
                    break  
        if sent_num == 0:
            continue            
        first_prob.append(first_correct/sent_num)
        second_prob.append(second_correct/sent_num)
        third_prob.append(third_correct/sent_num)
    print(f"Probability of first level pseudo-labels containing correct label is {np.mean(first_prob)}.")
    print(f"Probability of second level pseudo-labels containing correct label is {np.mean(second_prob)}.")
    print(f"Probability of third level pseudo-labels containing correct label is {np.mean(third_prob)}.\n")


def level_info_postconf(data):
    first_prob = []
    second_prob = []
    third_prob = []
    for a, doc_dict in enumerate(tqdm(data['data'])):
        cur_label = doc_dict['label']
        sent_num = len(doc_dict['sentences'])

        first_correct = 0
        second_correct = 0
        third_correct = 0
        for b, sen_dict in enumerate(doc_dict['sentences']): 
            
            if cur_label[0] in sen_dict['conf_label']:
                first_correct = first_correct + 1

            if len(cur_label) >= 2 and cur_label[1] in sen_dict['conf_label']:
                second_correct = second_correct + 1

            if len(cur_label) >= 3 and cur_label[2] in sen_dict['conf_label']:
                second_correct = second_correct + 1        
        if sent_num == 0:
            continue
        first_prob.append(first_correct/sent_num)
        second_prob.append(second_correct/sent_num)
        third_prob.append(third_correct/sent_num)
    print(f"Probability of confident core class contaning correct first-level label is {np.mean(first_prob)}.")
    print(f"Probability of confident core class contaning correct second-level label is {np.mean(second_prob)}.")
    print(f"Probability of confident core class contaning correct third-level label is {np.mean(third_prob)}.\n")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-file", help="Data file")
    args = parser.parse_args()

    with open(args.data_file, "r") as json_file:
            data = json.load(json_file)

    example_f1(data)
    level_info_preconf(data)
    level_info_postconf(data)