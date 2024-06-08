#!/usr/bin/env pythondomain_list
# -*- encoding: utf-8 -*-
'''
@Description:       :
 mining domain for MOH-X
@Author      :Yuan Tian
@version      :1.0
'''
import requests
import json
import csv
import os
import copy
import numpy as np
from nltk.corpus import wordnet 
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from itertools import combinations
from corenlp import StanfordCoreNLP
from itertools import combinations, permutations
import random
random.seed(42)
Pronoun_list = ['I', 'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'she', 'his', 'her', 'hers', 'himself', 'herself', 'we', 'us', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourselves', 'they', 'them', 'theirs', 'themselves']

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
dataset_to_file = {
    'MOH-X': ['MOH-X.csv'],
    'TroFi': ['TroFi.csv'],
    'LCC_Verb': ['LCC_Verb.csv'],
    'VUA_Verb': ['test.csv', 'train.csv', 'val.csv']
}



def find_hypernyms_wordnet(word, pos=None):
    if pos:
        synset_list = wordnet.synsets(word, pos)
    else:
        synset_list = wordnet.synsets(word)
    if len(synset_list) == 0:
        return [], []
    
    synset = synset_list[0]
    hypernyms_temp = synset.hypernyms()
    hypernym_string = [synset._name]
    hypernym_level = [0]
    count = 1
    while hypernyms_temp != []:
        next_hypernys_list = []
        for hypernym in hypernyms_temp:
            if hypernym._name not in ['physical_entity.n.01', 'abstraction.n.06', 'entity.n.01']:
                hypernym_string.append(hypernym._name)
                hypernym_level.append(count)
                if hypernym.hypernyms() != []:
                    next_hypernys_list.extend(hypernym.hypernyms())
            else:
                if hypernym_string == []: # only have 'physical_entity.n.01', 'abstraction.n.06', 'entity.n.01' as hypernyms
                    hypernym_string.append(hypernym._name)
                    hypernym_level.append(count)
                    return hypernym_string, hypernym_level
                else:
                    return hypernym_string, hypernym_level
        hypernyms_temp = next_hypernys_list
        count += 1

    return hypernym_string, hypernym_level


def split_key(sentence_dict): 
    # 不要的关系'AM-MOD' 是情绪动词 might等等 'R-A0 <-> Agent' 是 who 'R-A1 是 'that'
    # 'C-' 都是介词
    new_sentnece_dict = {}
    key_project = {}
    key_ori_list = list(sentence_dict.keys())
    for key_ori in key_ori_list:
        if 'R-' in key_ori or 'AM-MOD' in key_ori or 'AM-ADV' in key_ori or 'C-' in key_ori or 'AM'  in key_ori:
            continue
        if '<->' not in key_ori: # A or A0 or A1
            key_project[key_ori] = key_ori
        else:
            tokens = key_ori.split(' <-> ')
            if tokens[1][0].isupper():
                
                key_project[key_ori] = ' <-> '.join(tokens[:2])
            else:
                key_project[key_ori] = tokens[0]
    
    for key_ori in key_ori_list:
        if 'R-' in key_ori or 'AM-MOD' in key_ori or 'AM-ADV' in key_ori or 'C-' in key_ori or 'AM'  in key_ori:
            continue
        if key_ori == 'V <-> Verb':
            continue
        else:
            if key_project[key_ori] not in list(new_sentnece_dict.keys()):
                new_sentnece_dict[key_project[key_ori]] = sentence_dict[key_ori]
            else:
                new_sentnece_dict[key_project[key_ori]] = {
                    'span': [new_sentnece_dict[key_project[key_ori]]['span'], sentence_dict[key_ori]['span']],
                    'tokens': new_sentnece_dict[key_project[key_ori]]['tokens'] + sentence_dict[key_ori]['tokens']
                }
    # 补充
    for key_ori in key_ori_list:
        if 'R-' in key_ori or 'AM-MOD' in key_ori or 'AM-ADV' in key_ori or 'C-' in key_ori or 'AM'  in key_ori:
            continue

        if key_ori == 'V <-> Verb':
            continue
        if '<->' in key_ori:
            tokens = key_ori.split(' <-> ')
            if tokens[0] not in list(new_sentnece_dict.keys()):
                new_sentnece_dict[tokens[0]] = sentence_dict[key_ori]
    # 再补充
    for key_ori in key_ori_list:
        if 'R-' in key_ori or 'AM-MOD' in key_ori or 'AM-ADV' in key_ori or 'C-' in key_ori or 'AM'  in key_ori:
            continue

        if key_ori == 'V <-> Verb':
            continue
        if 'A' == key_ori[0]:
            if 'A' not in list(new_sentnece_dict.keys()):
                new_sentnece_dict['A'] = sentence_dict[key_ori]        
    
    return new_sentnece_dict
last_one_level_entity = ['entity.n.01']
last_second_level_entity = ['physical_entity.n.01', 'abstraction.n.06']

def task_assignments(tasks, groups):
    # # Example usage:
    # tasks_4 = ['task_1', 'task_2', 'task_3', 'task_4', 'task_5']
    # groups = 3
    # [{'task_1'}, {'task_2', 'task_3'}]
    # [{'task_1', 'task_2'}, {'task_3'}]
    # [{'task_1', 'task_3'}, {'task_2'}]

    n = len(tasks)
    m = groups

    if n < m:
        raise ValueError("Number of tasks should be greater than or equal to the number of groups.")
    if m == 1:
        group_assignments = []
        for task in tasks:
            group_assignments.append(set([task]))
        return [group_assignments]
    if m == n:
        
        return [[set(tasks)]]
    
    all_permutations = list(permutations(tasks, n))

    board_list = []
    for i in range(1, n):
        board_list.append(i)
    
    all_boards = list(permutations(board_list, m-1))
    all_boards_ascending = []
    for board in all_boards:
        if list(np.sort(np.array(board)))==list(board):
            all_boards_ascending.append(board)
    
    group_assignments = []
    for permutation_list in all_permutations:
        
        for board in all_boards_ascending:
            result = []
            for index, board_i in enumerate(board):
                if index == 0:
                    result.append(set(permutation_list[:board_i]))
                else:
                    result.append(set(permutation_list[board[index-1]:board_i]))
            result.append(set(permutation_list[board[-1]:]))
            result_permutations = list(permutations(result, len(result)))
            flag = False
            for result_permutation in result_permutations:
                if list(result_permutation) in group_assignments:
                    flag = True
            if not flag:
                group_assignments.append(result)

    return group_assignments

def find_hyper_in_domain_list(hypernym_string, hypernym_level, domain_list):
    hyper = ''
    hyper_index = -1
    for hyper_string_i, hyper_string_text in enumerate(hypernym_string):
        if hyper_string_text.split('.')[0] in domain_list:
            hyper = hyper_string_text
            hyper_index = hypernym_level[hyper_string_i]
            break
    if hyper_index != -1:
        return hyper, hyper_index
    else:
        if hypernym_level[0] == 0 and len(hypernym_string) > 1:
            return hypernym_string[1], hypernym_level[1]
        else:
            return hypernym_string[0], hypernym_level[0]


NER_dict = {
    'PERSON': 'person.n.01',
    'LOCATION': 'location.n.01',
    'ORGANIZATION': 'organization.n.01',
    'MONEY': 'money.n.03',
    'NUMBER': 'number.n.02',
    'ORDINAL': 'ordinal.a.02',
    'PERCENT': 'percentage.n.01',
    'DATE': 'date.n.01',
    'TIME': 'time.n.01',
    'DURATION': 'duration.n.01',
}
FULL_MODEL = ' your stanford-corenlp-full-2018-10-05 file path'
props = {'timeout': '5000000','annotators': 'pos, parse, depparse, lemma, ner', 'tokenize.whitespace': 'true' ,
        'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
if __name__ == '__main__':
    dataset = 'MOH-X' # MOH-X
    input_root_path = 'datasets_with_WPDom/verbparse_deparser/'
    output_root_path = 'datasets_with_WPDom/verbparse_deparser_domain/'
    nlp = StanfordCoreNLP(FULL_MODEL, lang='en')

    domain_list = []
    with open('datasets_with_WPDom/domain_list/Domains_in_Master_Metaphor_List_and_Cyc.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            domain_list.append(line.strip())


    for file_name in dataset_to_file[dataset]:

        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser', 'domain_role_dict']

        input_file_path = os.path.join(input_root_path, dataset, file_name)
        output_file_path = os.path.join(output_root_path, dataset, file_name)
        input_dataset = read_tsv(input_file_path)
        if not os.path.exists(os.path.join(output_root_path, dataset)):
            os.makedirs(os.path.join(output_root_path, dataset))
            output_dataset = []
        else:
            if not os.path.exists(output_file_path):
                output_dataset = []
            else:
                output_dataset = read_tsv(output_file_path)

        key_look = []
        for sample_i in range(len(output_dataset), len(input_dataset)):
            # sentence pharse
            sentence = input_dataset[sample_i][0]
            target_position = int(input_dataset[sample_i][2])
            verb_parser_dict = eval(input_dataset[sample_i][-2])
            sentences_parser_dict = eval(input_dataset[sample_i][-1])

            key_look.extend(list(verb_parser_dict.keys()))

            verb_parser_dict_clean = split_key(verb_parser_dict)
            verb_parser_dict_clean_keys = list(verb_parser_dict_clean.keys())
            flag_have_pair = False
            pair_roles_raw = {}
            for sentence_parser_dict in sentences_parser_dict:
                example_key_list = list(set(list(sentence_parser_dict.keys())))
                example_key_clean = []

                ovelap_key = list(set(list(verb_parser_dict.keys())).intersection(set(list(sentence_parser_dict.keys()))))
                key_look.extend(list(sentence_parser_dict.keys()))

                sentence_parser_dict_clean = split_key(sentence_parser_dict)
                sentence_parser_dict_clean_keys = list(sentence_parser_dict_clean.keys())
                intersection_keys = list(set(verb_parser_dict_clean_keys) & set(sentence_parser_dict_clean_keys))
                if intersection_keys != []:
                    flag_have_pair = True
                    for intersection_key in intersection_keys:
                        if intersection_key not in list(pair_roles_raw.keys()):
                            pair_roles_raw[intersection_key] = {
                                'target': verb_parser_dict_clean[intersection_key]['tokens'].copy(),
                                'source': sentence_parser_dict_clean[intersection_key]['tokens'].copy(),
                            }
                        else:
                            # no need for target tokens
                            pair_roles_raw[intersection_key]['source'].extend(sentence_parser_dict_clean[intersection_key]['tokens'])
            
            pair_roles = {} # deduplication & lower
            for key_role in list(pair_roles_raw.keys()):
                source_list = []
                target_list = []
                for source_word in pair_roles_raw[key_role]['source']:
                    if source_word.lower() == 'it':
                        continue
                    source_list.append(source_word.lower())
                for target_word in pair_roles_raw[key_role]['target']:
                    if target_word.lower() == 'it':
                        continue
                    target_list.append(target_word.lower())

                pair_roles[key_role] = {
                    'source': list(set(source_list)),
                    'target': list(set(target_list))
                                   }

            domain_role_dict = {}
            if not flag_have_pair:
                if eval(input_dataset[sample_i][-1]) != []:
                    print('can not find paired role', input_dataset[sample_i])

            else:
                pair_roles_hyper_list_dict = {}
                pair_roles_hyper_domain_dict = {}
                for key_role in list(pair_roles.keys()):
                    source_tokens_ori = pair_roles[key_role]['source']
                    target_tokens = pair_roles[key_role]['target']
                    source_tokens_lemma = []
                    target_tokens_lemma = []
                    for source_token in source_tokens_ori:
                        source_tokens_lemma.append(wnl.lemmatize(source_token.lower(), 'n'))
                    for target_token in target_tokens:
                        target_tokens_lemma.append(wnl.lemmatize(target_token.lower(), 'n'))
                    if source_tokens_lemma == [] or target_tokens_lemma == []:
                        continue
                    source_tokens_lemma_set = list(set(source_tokens_lemma))
                    target_tokens_lemma_set = list(set(target_tokens_lemma))
                    
                    source_hypernym_string_ori_list = []
                    source_hypernym_level_ori_list = []
                    source_tokens_pre = []
                    for source_token in source_tokens_lemma_set:
                        
                        source_hypernym_string, source_hypernym_level = find_hypernyms_wordnet(source_token, 'n')
                        if source_token in Pronoun_list:
                            source_hypernym_string_ori_list.append(['person.n.01'])
                            source_hypernym_level_ori_list.append([1])
                            source_tokens_pre.append(source_token)
                        elif source_hypernym_string == []:
                            # named (PERSON, LOCATION, ORGANIZATION, MISC), numerical (MONEY, NUMBER, ORDINAL, PERCENT), and temporal (DATE, TIME, DURATION, SET) entities (12 classes)
                            results_span_1=nlp.annotate(source_token, properties=props)
                            NER_labels_1 = results_span_1["sentences"][0]['entitymentions']
                            results_span_2=nlp.annotate(source_token[0].upper() + source_token[1:], properties=props)
                            NER_labels_2 = results_span_2["sentences"][0]['entitymentions']
                            if NER_labels_1 != []:
                                NER_label = NER_labels_1[0]['ner']
                                if NER_label in list(NER_dict.keys()):
                                    source_hypernym_string_ori_list.append([NER_dict[NER_label]])
                                    source_hypernym_level_ori_list.append([1])
                                    source_tokens_pre.append(source_token)
                            elif NER_labels_2 != []:
                                NER_label = NER_labels_2[0]['ner']
                                if NER_label in list(NER_dict.keys()):
                                    source_hypernym_string_ori_list.append([NER_dict[NER_label]])
                                    source_hypernym_level_ori_list.append([1])
                                    source_tokens_pre.append(source_token)
                        elif source_hypernym_string != []:
                            source_hypernym_string_ori_list.append(source_hypernym_string)
                            source_hypernym_level_ori_list.append(source_hypernym_level)
                            source_tokens_pre.append(source_token)
                    if source_tokens_pre == []:
                        continue
                    source_tokens_ori_indexes = list(range(len(source_tokens_pre)))

                    source_tokens = []
                    source_hypernym_string_list = []
                    source_hypernym_level_list = []
                    if len(source_tokens_pre) > 5: # no more than 5 elements
                        source_tokens_sample_index = random.sample(source_tokens_ori_indexes, 5)
                        for index_source in source_tokens_sample_index:
                            source_tokens.append(source_tokens_pre[index_source])
                            source_hypernym_string_list.append(source_hypernym_string_ori_list[index_source])
                            source_hypernym_level_list.append(source_hypernym_level_ori_list[index_source])
                    else:
                        source_tokens = source_tokens_pre
                        source_hypernym_string_list = source_hypernym_string_ori_list
                        source_hypernym_level_list = source_hypernym_level_ori_list
                    # 算不同source 分组的 共同上位词指标
                    source_tokens_indexes = list(range(len(source_tokens)))

                    source_combination_have_shared_domains = []
                    source_combination_have_shared_domains_scores = [] # 计算评价指标
                    source_combination_have_shared_domains_domains = []
                    source_combination_have_shared_domains_levels = []
                    for source_group_i in range(1, len(source_tokens)+1):
                        source_combinations = task_assignments(source_tokens_indexes, source_group_i)
                        
                        for source_combination in source_combinations:
                            # [{'task_1'}, {'task_2', 'task_3'}]
                            # find whether have shared domain
                            flag_have_shared_domain = True
                            source_combination_shared_domains = [] # 计算评价指标
                            source_combination_shared_levels = []
                            source_combination_scores = []
                            for source_combination_sub_ in source_combination:
                                source_combination_sub = list(source_combination_sub_)
                                if len(source_combination_sub) > 1:
                                    source_shared_domain = set(source_hypernym_string_list[source_combination_sub[0]]) 
                                    for source_combination_sub_index_i in range(1, len(source_combination_sub)):
                                        source_shared_domain = set(source_shared_domain) & set(source_hypernym_string_list[source_combination_sub[source_combination_sub_index_i]])
                                        if len(list(source_shared_domain)) == 0:
                                            flag_have_shared_domain = False
                                            break
                                    if not flag_have_shared_domain:
                                        break
                                # else: onky have one list in source_cobination_sub don't need to find shared domain
                            if flag_have_shared_domain:
                                for source_combination_sub_ in source_combination:
                                    source_combination_sub = list(source_combination_sub_)
                                    if len(source_combination_sub) == 1: # only have one argument
                                        source_hyper, source_hyper_level = find_hyper_in_domain_list(source_hypernym_string_list[source_combination_sub[0]], source_hypernym_level_list[source_combination_sub[0]], domain_list)
                                        source_combination_shared_domains.append(source_hyper)
                                        source_combination_shared_levels.append([source_hyper_level])
                                        source_combination_scores.append(1/(source_hyper_level+1))
                                    else:
                                        source_shared_domain = set(source_hypernym_string_list[source_combination_sub[0]]) 
                                        for source_combination_sub_index_i in range(1, len(source_combination_sub)):
                                            source_shared_domain = set(source_shared_domain) & set(source_hypernym_string_list[source_combination_sub[source_combination_sub_index_i]])

                                        # if no domain in Master & CYC domain, calculate the metric
                                        # 寻找level最低的source domain
                                        min_level = 1000000
                                        min_level_first_index = -1
                                        domains_score = 0 # abstraction level
                                        for source_domain_shared in list(source_shared_domain): 
                                            current_level = source_hypernym_level_list[source_combination_sub[0]][source_hypernym_string_list[source_combination_sub[0]].index(source_domain_shared)]
                                            if min_level > current_level:
                                                min_level = current_level
                                                min_level_first_index = source_hypernym_string_list[source_combination_sub[0]].index(source_domain_shared)
                                        min_level_source_domain = source_hypernym_string_list[source_combination_sub[0]][min_level_first_index]
                                        min_level_set = [min_level]
                                        domains_score += (min_level+1)
                                        
                                        for source_combination_sub_index in source_combination_sub[1:]:
                                            min_level_current = source_hypernym_level_list[source_combination_sub_index][source_hypernym_string_list[source_combination_sub_index].index(min_level_source_domain)]
                                            min_level_set.append(min_level_current)
                                            domains_score += min_level_current+1
                                        source_combination_shared_domains.append(min_level_source_domain)
                                        source_combination_shared_levels.append(min_level_set)
                                        source_combination_scores.append(len(source_combination_sub)/(domains_score/len(source_combination_sub)))
                                    # 计算score    
                                source_combination_have_shared_domains.append(source_combination)
                                source_combination_have_shared_domains_domains.append(source_combination_shared_domains)
                                source_combination_have_shared_domains_levels.append(source_combination_shared_levels)
                                source_combination_have_shared_domains_scores.append(np.mean(np.array(source_combination_scores)))
                            # else: don't have shared domain discard this combination of arguments
                    # 找评价指标source 最大的
                    max_score_index = np.argmax(np.array(source_combination_have_shared_domains_scores))
                    source_combination_have_shared_domain_selected = source_combination_have_shared_domains[max_score_index]
                    source_combination_have_shared_domains_score_selected = source_combination_have_shared_domains_scores[max_score_index] # 计算评价指标
                    source_combination_have_shared_domains_domain_selected = source_combination_have_shared_domains_domains[max_score_index]
                    source_combination_have_shared_domains_level_selected = source_combination_have_shared_domains_levels[max_score_index]


                    # target方面 只有一个词 不用计算 domain granularity metric
                    target_hypernym_string_list = []
                    target_hypernym_level_list = []
                    target_tokens = []
                    for target_token in target_tokens_lemma_set:
                        target_hypernym_string, target_hypernym_level = find_hypernyms_wordnet(target_token, 'n')
                        if target_token in Pronoun_list:
                            target_hypernym_string_list.append(['person.n.01'])
                            target_hypernym_level_list.append([1])
                            target_tokens.append(target_token)
                        elif target_hypernym_string == []:
                            # named (PERSON, LOCATION, ORGANIZATION, MISC), numerical (MONEY, NUMBER, ORDINAL, PERCENT), and temporal (DATE, TIME, DURATION, SET) entities (12 classes)
                            results_span_1=nlp.annotate(target_token, properties=props)
                            NER_labels_1 = results_span_1["sentences"][0]['entitymentions']
                            results_span_2=nlp.annotate(target_token[0].upper() + target_token[1:], properties=props)
                            NER_labels_2 = results_span_2["sentences"][0]['entitymentions']
                            if NER_labels_1 != []:
                                NER_label = NER_labels_1[0]['ner']
                                if NER_label in list(NER_dict.keys()):
                                    target_hypernym_string_list.append([NER_dict[NER_label]])
                                    target_hypernym_level_list.append([1])
                                    target_tokens.append(target_token)
                            elif NER_labels_2 != []:
                                NER_label = NER_labels_2[0]['ner']
                                if NER_label in list(NER_dict.keys()):
                                    target_hypernym_string_list.append([NER_dict[NER_label]])
                                    target_hypernym_level_list.append([1])
                                    target_tokens.append(target_token)
                        elif target_hypernym_string != []:
                            target_hyper, target_hyper_level = find_hyper_in_domain_list(target_hypernym_string, target_hypernym_level, domain_list)
            
                            target_hypernym_string_list.append([target_hyper])
                            target_hypernym_level_list.append([target_hyper_level])
                            target_tokens.append(target_token)
                        
                    if target_hypernym_string_list == []:
                        print("no target_hypernym_string_list")
                    target_domains_list_check = []  
                    for target_i, target_token in enumerate(target_tokens):
                        target_domains_list_check.append(target_hypernym_string_list[target_i][0]) 
                    # source domain target domain的definition:
                    source_domains_list_definition = []
                    target_domains_list_definition = []
                    for source_domain in source_combination_have_shared_domains_domain_selected:
                        source_domains_list_definition.append(wordnet.synset(source_domain).definition())
                    for target_domain in target_domains_list_check:
                        target_domains_list_definition.append(wordnet.synset(target_domain).definition())

                    if source_hypernym_string_list != [] and  target_hypernym_string_list!=[]:
                        domain_role_dict[key_role] = {
                            'source_string': source_hypernym_string_list.copy(),
                            'source_level': source_hypernym_level_list.copy(),
                            'source_combination': source_combination_have_shared_domain_selected.copy(),
                            'source_combination_levels': source_combination_have_shared_domains_level_selected.copy(),
                            'source_score': source_combination_have_shared_domains_score_selected.copy(),
                            'target_string': target_hypernym_string_list.copy(),
                            'target_level': target_hypernym_level_list.copy(),
                            'source_tokens': source_tokens.copy(),
                            'source_domain': source_combination_have_shared_domains_domain_selected.copy(),
                            'source_domain_definition': source_domains_list_definition.copy(),
                            'target_tokens': target_tokens.copy(),
                            'target_domain': target_domains_list_check.copy(),
                            'target_domain_definition': target_domains_list_definition.copy(),
                        }
                


            print(domain_role_dict)
            output_dataset.append(input_dataset[sample_i] + [domain_role_dict])
            write_dataset(output_file_path, head, output_dataset)
                
                



        
