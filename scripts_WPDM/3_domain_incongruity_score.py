#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
使用wordnet中path similarity 计算domain相似度
@Author      :Yuan Tian
@version      :1.0
'''
from nltk.corpus import wordnet 
import os
import csv
import numpy as np
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
def path_similarity(list1, list2):
    scores = []
    for word1 in list1:
        for word2 in list2:
            scores.append(wordnet.synset(word1).path_similarity(wordnet.synset(word2)))
    return scores

def lch_similarity(list1, list2):
    scores = []
    for word1 in list1:
        for word2 in list2:
            if wordnet.synset(word1)._pos == wordnet.synset(word2)._pos:
                scores.append(wordnet.synset(word1).lch_similarity(wordnet.synset(word2)))
    return scores

def wup_similarity(list1, list2):
    scores = []
    for word1 in list1:
        for word2 in list2:
            scores.append(wordnet.synset(word1).wup_similarity(wordnet.synset(word2)))
    return scores

def res_similarity(list1, list2, ic):
    scores = []
    for word1 in list1:
        for word2 in list2:
            scores.append(wordnet.synset(word1).res_similarity(wordnet.synset(word2)), ic)
    return scores

def jcn_similarity(list1, list2):
    scores = []
    for word1 in list1:
        for word2 in list2:
            scores.append(wordnet.synset(word1).jcn_similarity(wordnet.synset(word2)))
    return scores

def lin_similarity(list1, list2):
    scores = []
    for word1 in list1:
        for word2 in list2:
            scores.append(wordnet.synset(word1).lin_similarity(wordnet.synset(word2)))
    return scores

similarity_dict = {
    'path_similarity': path_similarity,
    'lch_similarity': lch_similarity,
    'wup_similarity': wup_similarity,
    'res_similarity': res_similarity,
    'jcn_similarity': jcn_similarity,
    'lin_similarity': lin_similarity
}
similarity_list = ['path_similarity', 'lch_similarity', 'wup_similarity']
dataset_to_file = {
    'MOH-X': ['MOH-X.csv'],
    'TroFi': ['TroFi.csv'],
    'VUA_Verb': ['test.csv', 'train.csv', 'val.csv'],
    'LCC_Verb': ['LCC_Verb.csv'],

}
if __name__ == "__main__":

    dataset = 'LCC_Verb' # LCC_Verb/MOH-X/TroFi/VUA_Verb
    input_root_path = 'datasets_with_WPDom/verbparse_deparser_domain/'
    output_root_path = 'datasets_with_WPDom/verbparse_deparser_domain_scores/'
    sim_method = 'path_similarity'
    domain_to_definition = {}
    

    for file_name in dataset_to_file[dataset]:
        count_domain_dict_none = 0
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser', 'domain_role_dict']
        print(dataset)
        print(file_name)
        input_file_path = os.path.join(input_root_path, dataset, file_name)
        output_file_path = os.path.join(output_root_path, dataset, file_name)
        
        if not os.path.exists(os.path.join(output_root_path, dataset)):
            os.makedirs(os.path.join(output_root_path, dataset))
            output_dataset = []
        else:
            if not os.path.exists(output_file_path):
                output_dataset = []
            else:
                output_dataset = read_tsv(output_file_path)
        # output_dataset = []
        
        input_dataset = read_tsv(input_file_path)

        for sample_i in range(len(output_dataset), len(input_dataset)):
            sentence = input_dataset[sample_i][0]
            examples = eval(input_dataset[sample_i][8]) + eval(input_dataset[sample_i][10]) + eval(input_dataset[sample_i][12])
            domain_role_dict = eval(input_dataset[sample_i][-1])
            if domain_role_dict == {}:
                output_dataset.append(input_dataset[sample_i][:-1] + [domain_role_dict])
                continue

            if 'A' in list(domain_role_dict.keys()) and len(list(domain_role_dict.keys())) > 1:
                domain_role_dict.pop('A') 
            if 'A0' in list(domain_role_dict.keys()):
                for key_role in list(domain_role_dict.keys()):
                    if 'A0 <->' in key_role:
                        domain_role_dict.pop('A0') 
            if 'A1' in list(domain_role_dict.keys()):
                for key_role in list(domain_role_dict.keys()):
                    if 'A1 <->' in key_role:
                        domain_role_dict.pop('A1') 
            if 'A2' in list(domain_role_dict.keys()):
                for key_role in list(domain_role_dict.keys()):
                    if 'A2 <->' in key_role:
                        domain_role_dict.pop('A2') 
            print(sentence)
            for key_role in list(domain_role_dict.keys()):
                print(key_role)
                target_source_scores = []
                source_domain_selected_list = []
                source_domain_selected_definition_list = []
                for domain_index, target_domain in enumerate(domain_role_dict[key_role]['target_domain']):
                    score_list = []
                    
                    for domain_index2, source_domain in enumerate(domain_role_dict[key_role]['source_domain']):
                        sim_score = similarity_dict[sim_method]([target_domain], [source_domain])[0]
                        score_list.append(sim_score)
                    max_sim_index = list(np.where(np.array(sim_score) == np.max(np.array(sim_score)))[0])[0]
                    score_pair = score_list[max_sim_index]
                    source_domain_selected = domain_role_dict[key_role]['source_domain'][max_sim_index]
                    source_domain_definition_selected = domain_role_dict[key_role]['source_domain_definition'][max_sim_index]
                    source_domain_selected_list.append(source_domain_selected)
                    source_domain_selected_definition_list.append(source_domain_definition_selected)
                    target_source_scores.append(score_pair)
                
                    # print('-----------------------') 
                domain_role_dict[key_role]['target_source_scores'] = target_source_scores 
                domain_role_dict[key_role]['source_domain_selected'] = source_domain_selected_list
                domain_role_dict[key_role]['source_domain_selected_definition'] = source_domain_selected_definition_list
            output_dataset.append(input_dataset[sample_i][:-1] + [domain_role_dict])
            write_dataset(output_file_path, head, output_dataset)
        write_dataset(output_file_path, head, output_dataset)
        print(dataset)
        print(file_name)


 

