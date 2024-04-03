import json
import argparse
import re
import hierarchy_helper as hierarchy_helper
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def span_corpus(corpus_path, class_path, label_path, out_path, lower_size, upper_size):
    #read auto phrase result and determine keyword
    keywords = set()
    """with open(auto_path) as file:
        for line in file:
            line = line.replace('\n', '')
            val = float(line.split('\t')[0])
            if val > threshold:
                keywords.add(line.split('\t')[1])"""

    docs = []
    with open(corpus_path) as file:
        for i, line in enumerate(file):
            if i < lower_size:
                continue
            if i >= upper_size:
                break
            #print(i)
            line = line.replace('\n', '')
            text = line
            #text = line.split('\t')[1].strip()
            #add to doc
            doc_dict = {}
            doc_dict['id'] = i
            doc_dict['text'] = text
            doc_dict['sentences'] = []
            docs.append(doc_dict)

            #add all sentences
            #sents = text.split('.')
            sents = re.split(r'[!?.]', text)
            sents = [sentence.strip() for sentence in sents if sentence.strip()]
            for j, sen in enumerate(sents):
                if sen == "":
                    break
                sen_dict = {}
                sen_dict['id'] = j
                sen_dict['text'] = sen
                
                doc_dict['sentences'].append(sen_dict)

                #add all words
                """words = sen.strip().split(" ")
                sen_dict['words'] = []
                for k, word in enumerate(words):
                    #travers and determine if word is key word
                    if word.strip() in keywords:
                        word_dict = {}
                        word_dict['id'] = k
                        word_dict['text'] = word.strip()
                        sen_dict['words'].append(word_dict)"""
    classes = []
    with open(class_path) as file:
        for line in file:
            line = line.replace('\n', '')
            label_name = line.split('\t')[1].strip()
            classes.append(label_name)

    print(len(docs))
    #this part probably shouldn't parpiticate traning process, only used for validation
    with open(label_path) as file:
        for i, line in enumerate(file):
            if i < lower_size:
                continue
            if i >= upper_size:
                break
            line = line.replace('\n', '')
            labels = line.split('\t')[1].strip().split(',')
            docs[i-lower_size]['label'] = [int(label) for label in labels]

    #add every thing to a big dict
    data = {}
    data['classes'] = classes
    data['data'] = docs

    #save as json format
    with open(out_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print("Data written to {}".format(out_path))   

def transform_label(label_path, out_label_path):
    data = []
    with open(label_path) as file:
        for line in file:
            data.append(line.replace("\t", ":"))

    with open(out_label_path, "w") as file:
        for line in data:
            file.write(line)

    print("Labels written to {}".format(out_label_path))  


def transform_label_by_level():
    #get labels in the same level
    G = hierarchy_helper.create_label_graph("./TaxoClass-dataset/Amazon-531/label_hierarchy.txt", 20)
    first_level = hierarchy_helper.get_first_level(G)
    labels = hierarchy_helper.get_labels("./TaxoClass-dataset/Amazon-531/train/labels.txt")
    first_labels = [labels[i] for i in first_level]
    print(len(first_level))
    
    second_level =  hierarchy_helper.get_childs(G, first_level, 1)
    second_set = set(second_level)
    second_u = list(second_set)
    second_u.sort()
    second_labels = [labels[i] for i in second_u]
    print(len(second_u))

    third_level =  hierarchy_helper.get_childs(G, second_level, 2)
    third_set = set(third_level)
    third_u = list(third_set)
    third_u.sort()
    third_labels = [labels[i] for i in third_u]
    print(len(third_u))

    with open("./SeeTopic/Amazon-531/keywords_first_0.txt", "w") as file:
        for i, label in enumerate(first_labels):
            file.write(f"{i}:{label}\n")

    with open("./SeeTopic/Amazon-531/keywords_second_0.txt", "w") as file:
        for i, label in enumerate(second_labels):
            file.write(f"{i}:{label}\n")

    with open("./SeeTopic/Amazon-531/keywords_third_0.txt", "w") as file:
        for i, label in enumerate(third_labels):
            file.write(f"{i}:{label}\n")
    

def lem():
    sents = []
    with open("./TaxoClass-dataset/Amazon-531/train/text.txt", "r") as file:
        for line in file:
            words = word_tokenize(line.strip())
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            sents.append(' '.join(lemmatized_words))
    
    with open("./TaxoClass-dataset/Amazon-531/train/text_lem.txt", "w") as file:
        for line in sents:
            file.write(f"{line}\n")
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, help="Output path")
    parser.add_argument("-l", "--lower_size", type=int, help="Document lower size", default = 0)
    parser.add_argument("-u", "--upper_size", type=int, help="Document upper size", default = 10)

    args = parser.parse_args()
    
    nltk.download('punkt')
    nltk.download('wordnet')
    
    span_corpus("./TaxoClass-dataset/Amazon-531/train/text.txt", 
                "./TaxoClass-dataset/Amazon-531/train/labels.txt", 
                "./TaxoClass-dataset/Amazon-531/train/doc2labels.txt",
                args.output_path,
                int(args.lower_size), int(args.upper_size))
    
    """transform_label("./TaxoClass-dataset/Amazon-531/train/labels.txt",
                    "./SeeTopic/Amazon-531/keywords_0.txt")"""
    #transform_label_by_level()

    #lem()
    