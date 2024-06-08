#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :将不同词典中的基本意思整合起来
@Author      :Yuan Tian
@version      :1.0
'''

import os
import csv
import argparse
import numpy as np

dataset_to_fileList = {
    'MOH-X':['MOH-X.csv'],
    'TroFi':['TroFi.csv'],
    'VUA_Verb':['test.csv','train.csv','val.csv'],
    'LCC_Verb': ['LCC_Verb.csv'],
}
def get_dataset(file_dir):

    examples = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        lines = csv.reader(f)
        next(lines)  # skip the headline
        for i, line in enumerate(lines):
            
            examples.append(line)
    return examples
def write_dataset(file_dir, head, examples):

    with open(file_dir, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for line in examples:
            writer.writerow(line)
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
                    examples.append(line)

            return examples

def remove_none(sentences):
    new_sentences = []
    for sentence in sentences:
        if sentence != None:
            new_sentences.append(sentence)
    return new_sentences
def parse_option():
    parser = argparse.ArgumentParser(description='Train on MOH-X dataset, do cross validation')
    parser.add_argument('--data_dir', default='datasets/data-basicmeaning-examples/', type=str)
    parser.add_argument('--dataset', default='LCC_Verb', type=str, help='dataset name: MOH-X, TroFi, VUA_Verb')
    parser.add_argument('--output_path', default='datasets/', type=str)

    args, unparsed = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_option()
    dict_list = os.listdir(args.data_dir)
    for file in dataset_to_fileList[args.dataset]:
        samples_dict = []
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent']
        for dict_name in dict_list:
            file_dir = os.path.join(args.data_dir, dict_name, args.dataset, file)
            samples = get_dataset(file_dir)
            samples_dict.append(samples)
            head.extend(['bis_def_' + dict_name, 'exps_' + dict_name])
        new_dataset = []
        for i, dict_name in enumerate(dict_list):
            if i == 0:
                for sample in samples_dict[i]:
                    new_dataset.append(sample[:-1] + [eval(sample[-1])])
            else:
                for j, sample in enumerate(samples_dict[i]):
                    sentences = eval(samples_dict[i][j][-1])
                    sentences_clear = remove_none(sentences)

                    new_dataset[j].append(samples_dict[i][j][-2])
                    new_dataset[j].append(sentences_clear)

        count_sentences_1 = []
        count_sentences_2 = []
        count_sentences_3 = []
        count_1 = 0
        count_2 = 0
        count_3 = 0
        sentence_wrong_1 = []
        sentence_wrong_2 = []
        sentence_wrong_3 = []
        for sample in new_dataset:
            count_sentences_1.append(len(sample[-5]))
            count_sentences_2.append(len(sample[-3]))
            count_sentences_3.append(len(sample[-1]))

            for sentence in sample[-5]:
                if len(sentence) < 10:
                    # print(sentence)
                    sentence_wrong_1.append(sentence)
                    count_1 += 1
                if sentence[-1] == ' ' and sentence[-2] != '.':
                    # print(sentence)
                    sentence_wrong_1.append(sentence)
                    count_1 += 1

            for sentence in sample[-3]:
                if len(sentence) < 10:
                    # print(sentence)
                    sentence_wrong_2.append(sentence)
                    count_2 += 1
                if sentence[-1] == ' ' and sentence[-2] != '.':
                    # print(sentence)
                    sentence_wrong_2.append(sentence)
                    count_2 += 1

            for sentence in sample[-1]:
                if len(sentence) < 10:
                    # print(sentence)
                    sentence_wrong_3.append(sentence)
                    count_3 += 1
                if sentence[-1] == ' ' and sentence[-2] != '.':
                    # print(sentence)   
                    sentence_wrong_3.append(sentence)        
                    count_3 += 1                                          


        if not os.path.exists(os.path.join(args.output_path, args.dataset)):
            os.makedirs(os.path.join(args.output_path, args.dataset))
        write_dataset(os.path.join(args.output_path, args.dataset, file), head, new_dataset)

