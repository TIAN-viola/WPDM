#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
从不同词典找动词基本语义的例句：
- Longman
- Oxford
- 牛津九阶高阶词典
@Author      :Yuan Tian
@version      :1.0
'''

import pandas as pd
import numpy as np
import random
from nltk.corpus import wordnet
import re
from readmdict import MDX, MDD  # pip install readmdict
from pyquery import PyQuery as pq    # pip install pyquery
import os
import csv
import sys
import argparse
from Collins_dictionary import collins_html_parser
from oxford_dictionary import oxford_html_parser
from longman_dictionary import longman_html_parser

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()



POS_change = {
    'v': 'VERB',
    'n': 'NOUN',
    'adj': 'ADJ',
    'adv': 'ADV'
}

pos_list = {
    "JJ": 'adjective',
    "JJR": 'adjective',
    "JJS": 'adjective',
    "NN": 'noun',
    "NNS": 'noun',
    "NNP": 'noun',
    "NNPS": 'noun',
    "PRP": 'noun',
    'RB': 'adverb',
    'RBR': 'adverb',
    'RBS': 'adverb',
    'VB': 'verb',
    'VBG': 'verb',
    'VBN': 'verb',
    'VBP': 'verb',
    'VBZ': 'verb',
    'VBD': 'verb',
    'VERB':'verb',
    'verb':'verb',
    'NOUN': 'noun',
    'ADJ': 'adjective',
    'ADV': 'adverb'
}

pos_to_wordnet_pos = {
    'verb': 'v',
    'noun': 'n',
    'adjective': 'a'
}

class Data_Processor:
    def __init__(self, args):
        self.data_root_path = os.path.join(args.data_dir, args.dataset)
        self.output_root_path = os.path.join(args.output_path, args.dictionary, args.dataset)
    
    def get_dataset(self, file):
        file_dir = os.path.join(self.data_root_path, file)
        examples = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = csv.reader(f)
            next(lines)  # skip the headline
            for i, line in enumerate(lines):
                # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                # example sentence may be empty, caution for processing
                
                examples.append(line)
        return examples

    def write_dataset(self, file, head, examples):
        if not os.path.exists(self.output_root_path):
            os.makedirs(self.output_root_path)
        with open(os.path.join(self.output_root_path, file), 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            for line in examples:
                writer.writerow(line)

    def get_dataset_output(self, file):
        if not os.path.exists(os.path.join(self.output_root_path, file)):
            return []
        else:
            with open(os.path.join(self.output_root_path, file), 'r', encoding='utf-8') as f:
                lines = csv.reader(f)
                next(lines)  # skip the headline
                examples = []
                for i, line in enumerate(lines):
                    # sentence,label,target_position,target_word,pos_tag,gloss,eg_sent
                    # example sentence may be empty, caution for processing
                    examples.append(line)

            return examples

dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
    'VUA_Verb':['test.csv','train.csv','val.csv'],
    'LCC_Verb': ['LCC_Verb.csv'],
}


def parse_option():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='datasets/LCC_Verb/', type=str)
    parser.add_argument('--dataset', default='LCC_Verb', type=str, help='dataset name: MOH-X, TroFi, VUA_Verb,LCC_Verb')
    parser.add_argument('--output_path', default='datasets/data-basicmeaning-examples/', type=str)
    parser.add_argument('--dictionary', default='oxford_advanced', type=str, help='collins, longman, oxford_advanced')
    args, unparsed = parser.parse_known_args()
    return args

dictionary_paths_dict = {
    'collins': 'your collins mdx file',
    'oxford_advanced': 'your oxford advanced mdx file',
    'longman': 'your longman mdx file',
}

dictionary_parser_dict = {
    'collins': collins_html_parser,
    'oxford_advanced': oxford_html_parser,
    'longman': longman_html_parser,
}

other_list = ['ADP','DET', 'PROPN', 'CCONJ', 'PRON', 'NUM', 'PART', 'INTJ', 'SCONJ', 'AUX', 'PUNCT', 'SYM', 'X', 'SCONJ']

# SCONJ: subordinating conjunction The 10 most frequent SCONJ lemmas: that, if, as, because, for, of, since, before, like, after

if __name__ == "__main__":
    args = parse_option()
    dataloader = Data_Processor(args)

    dictionary_path = dictionary_paths_dict[args.dictionary]

    # load dictionary
    headwords = [*MDX(dictionary_path)]       # 单词名列表
    items_html = [*MDX(dictionary_path).items()]   # 释义html源码列表
    if len(headwords)==len(items_html):
        print(f'加载成功：共{len(headwords)}条')
    else:
        print(f'【ERROR】加载失败{len(headwords)}，{len(items_html)}')
    dictionary_parser = dictionary_parser_dict[args.dictionary]
    
    head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent', 'examples']
    print(args.dictionary)
    print(args.dataset)

    for file in dataset_to_fileList[args.dataset]:

        samples = dataloader.get_dataset(file)
        new_samples = dataloader.get_dataset_output(file)
        for sample_id in range(len(new_samples), len(samples)):
            sample  = samples[sample_id]
            sentence,label,target_position,target_word_origional,pos_tag,gloss,eg_sent = sample
            if args.dataset == 'VUA_Verb':  
                pos_tag = 'VERB'
            if pos_tag in other_list:
                continue
            if pos_list[pos_tag] not in ['adverb', 'adjective']:
                target_word = wnl.lemmatize(target_word_origional, pos_to_wordnet_pos[pos_list[pos_tag]])
            else:
                target_word = target_word_origional
            first_definition, examples_list = dictionary_parser(target_word.lower(), pos_list[pos_tag], headwords, items_html)

            new_samples.append([sentence,label,target_position,target_word,pos_tag,gloss,eg_sent,first_definition, examples_list])
            dataloader.write_dataset(file, head, new_samples)

