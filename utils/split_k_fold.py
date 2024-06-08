#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : 将数据集 k fold split
@Date     :2024/01/21 20:27:16
@Author      :Yuan Tian
@version      :1.0
'''
import numpy as np
from sklearn.model_selection import KFold
import os
import csv
import random
random.seed(42)

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        
        lines = csv.reader(f)
        next(lines)
        new_lines = []
        for i, line in enumerate(lines):


            new_lines.append(line)
        return new_lines

def write_dataset(file, head, examples):

    with open(file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(head)
        for line in examples:
            writer.writerow(line)

if __name__ == "__main__":
    head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser', 'domain_role_dict']

    kf = KFold(n_splits=10)

    dataset_name = 'MOH-X'
    input_file = 'datasets_with_WPDom/verbparse_deparser_domain_scores/'+ dataset_name + '/' + dataset_name +'.csv'
    output_path = 'datasets_with_WPDom/verbparse_deparser_domain_scores_k_fold/'+ dataset_name + '/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    datasets = read_tsv(input_file)
    random.shuffle(datasets)



    for i, (train, test) in enumerate(kf.split(datasets)):
        test_dataset = []
        train_dataset = []
        for test_index in test:
            test_dataset.append(datasets[test_index])
        for train_index in train:
            train_dataset.append(datasets[train_index])
        write_dataset(os.path.join(output_path, 'train' + str(i) + '.csv'), head, train_dataset)
        write_dataset(os.path.join(output_path, 'test' + str(i) + '.csv'), head, test_dataset)

