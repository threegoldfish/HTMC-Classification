import argparse
import os
import csv
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import re

def make_blank(str):
	str = re.sub(r',', ' , ', str)
	str = re.sub(r'\.', ' . ', str)
	str = re.sub(r':', ' : ', str)
	str = re.sub(r';', ' ; ', str)
	str = re.sub(r'#', ' # ', str)
	str = re.sub(r'=', ' = ', str)
	str = re.sub(r'\(', ' ( ', str)
	str = re.sub(r'\)', ' ) ', str)
	str = re.sub(r'\?', ' ? ', str)
	str = re.sub(r'\!', ' ! ', str)
	str = re.sub(r'\'', ' \' ', str)
	str = re.sub(r'\"', ' \" ', str)
	str = re.sub(r'--', ' -- ', str)
	str = re.sub(r'(\d+)', r' \1 ', str)

	
	str = re.sub(r'\s+', ' ', str)

	return str
	
	"""string = re.sub(r"[^A-Za-z0-9(),.!?_\"\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\"", " \" ", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'m", " \'m", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"\.", " . ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\$", " $ ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	
	return string.strip().lower()"""


def phrase_process():
	f = open('./AutoPhrase/models/Amazon-531Model/segmentation.txt')#CHANGE
	g = open(args.out_file, 'w')

	q_pattern = r'<phrase_Q=.....>'

	for line in tqdm(f):
		line = line.split('\t', 1)[1].strip()
		doc = ''
		temp = re.split(q_pattern, line)
		#temp = line.split('<phrase>')
		for seg in temp:
			temp2 = seg.split('</phrase>')
			if len(temp2) > 1:
				doc += ('_').join(temp2[0].split(' ')) + make_blank(temp2[1])
			else:
				doc += make_blank(temp2[0])
		g.write(doc.strip().lower()+'\n')
	print("Phrase segmented corpus written to {}".format(args.out_file))
	return 


def preprocess():
	f = open(os.path.join(args.dataset, args.in_file))
	docs = f.readlines()
	f_out = open(args.out_file, 'w')
	for doc in tqdm(docs):
		f_out.write(' '.join([w.lower() for w in word_tokenize(doc.strip())]) + '\n')
	return 


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mode', type=int)
	parser.add_argument('--dataset', default='NEW')
	parser.add_argument('-o', '--out_file', default='./TaxoClass-dataset/Amazon-531/train/text.txt')#CHANGE
	args = parser.parse_args()

	if args.mode == 0:
		preprocess()
	else:
		phrase_process()
	
	"""str = "String without,blank.bla#bla; bla:bla 1234bla"
	print(make_blank(str))"""
	
				