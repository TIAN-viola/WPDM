#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import requests
import json
import csv
import os
import numpy as np
from corenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from corenlp import StanfordCoreNLP
import copy
puncts = [',', ';', ':', '.', '!', '?', "'", '"', ']', ')', '(', '[', '-', '—']


def split_punct(sentence):
    tokens_ori = sentence.split(' ')
    tokens = []
    for token in tokens_ori:
        span = ''
        span_list = []
        for symbol in token:
            if symbol in puncts:
                if span != '':
                    span_list.append(span)
                    span = ''
                span_list.append(symbol)
            else:
                span += symbol
        if span !='':
            span_list.append(span)
        tokens.extend(span_list)
    return tokens 

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

def get_verb_parser(url, params):
# Making a GET request to the API
    response = requests.get(url, params=params)

    # Checking if the request was successful
    if response.status_code == 200:

        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def sentence_dependency_parser(tokens, target_position):
    sentence = ' '.join(tokens)
    target_word = tokens[target_position]

    agent_list = ['nsubj', 'nmod:agent', 'obl:agent', 'nmod:agent', 'agent']
    theme_list = ['obj', 'dobj', 'nsubjpass', 'nsubj:xsubj', 'nsubj:pass', 'nsubj:passdir']
    results_span=nlp.annotate(sentence, properties=props)
    head_dependent_span = results_span["sentences"][0]['enhancedDependencies']
    # find the index of focus verb
    target_index = -1
    # example {'A0 <-> Agent <-> abuser, agent': {'span': 'This boss', 'tokens': ['boss']}, 'V <-> Verb': {'span': 'abuses', 'tokens': ['abuses']}, 'A1 <-> Theme <-> entity abused': {'span': 'his workers', 'tokens': ['workers']}}
    frame_verb = {}
    for head_dependent_sample in head_dependent_span:
        if head_dependent_sample['governorGloss'] == target_word:
            target_index = head_dependent_sample['governor']
            break
    if target_index == -1:
        print('can find target word in this sentence deparser!', sentence)    
    
    agent_token_index = []
    theme_token_index = []
    for head_dependent_sample in head_dependent_span:
        if head_dependent_sample['governor'] == target_index:
            if head_dependent_sample['dep'] in agent_list:
                if 'A0 <-> Agent' not in list(frame_verb.keys()):
                    frame_verb['A0 <-> Agent'] = {
                        'span': [head_dependent_sample['dependentGloss']], 
                        'tokens': [head_dependent_sample['dependentGloss']]
                        }
                else:
                    frame_verb['A0 <-> Agent']['span'].append(head_dependent_sample['dependentGloss'])
                    frame_verb['A0 <-> Agent']['tokens'].append(head_dependent_sample['dependentGloss'])
                agent_token_index.append(head_dependent_sample['dependent'])
            elif head_dependent_sample['dep'] in theme_list:

                if 'A1 <-> Theme' not in list(frame_verb.keys()):
                    frame_verb['A1 <-> Theme'] = {
                        'span': [head_dependent_sample['dependentGloss']], 
                        'tokens': [head_dependent_sample['dependentGloss']]
                        }
                else:
                    frame_verb['A1 <-> Theme']['span'].append(head_dependent_sample['dependentGloss'])
                    frame_verb['A1 <-> Theme']['tokens'].append(head_dependent_sample['dependentGloss'])
                theme_token_index.append(head_dependent_sample['dependent'])
    if frame_verb == {}:
        return None
    else:
        return frame_verb

def sentence_dependency_parser_other(tokens, target_position):
    sentence = ' '.join(tokens)
    target_word = tokens[target_position]

    count_agent = 0
    results_span=nlp.annotate(sentence, properties=props)
    head_dependent_span = results_span["sentences"][0]['enhancedDependencies']
    # find the index of focus verb
    target_index = -1
    # example {'A0 <-> Agent <-> abuser, agent': {'span': 'This boss', 'tokens': ['boss']}, 'V <-> Verb': {'span': 'abuses', 'tokens': ['abuses']}, 'A1 <-> Theme <-> entity abused': {'span': 'his workers', 'tokens': ['workers']}}
    frame_verb = {}
    for head_dependent_sample in head_dependent_span:
        if head_dependent_sample['governorGloss'] == target_word:
            target_index = head_dependent_sample['governor']
            break
        elif head_dependent_sample['dependentGloss'] == target_word:
            target_index = head_dependent_sample['dependent']
            break
    if target_index == -1:
        print('can find target word in this sentence deparser!', sentence)    
    
    agent_token_index = []
    theme_token_index = []
    for head_dependent_sample in head_dependent_span:
        if head_dependent_sample['governor'] == target_index:
            frame_verb['A' + str(count_agent)] = {
                'span': [head_dependent_sample['dependentGloss']], 
                'tokens': [head_dependent_sample['dependentGloss']]
                }
            count_agent+1
        elif head_dependent_sample['dependent'] == target_index:
            frame_verb['A' + str(count_agent)] = {
                'span': [head_dependent_sample['governorGloss']], 
                'tokens': [head_dependent_sample['governorGloss']]
                }
            count_agent+1

    if frame_verb == {}:
        return None
    else:
        return frame_verb


def sentence_verb_parser(tokens, target_position):
        sentence = ' '.join(tokens)

        # 一些verbnet pharser处理不了的情况
        if ' - ' in sentence:
            sentence_new = sentence.replace(' - ', '-')
        else:
            sentence_new = copy.deepcopy(sentence)
        if 'drop-kicked' in sentence:
            sentence_new = sentence.replace('drop-kicked', 'dropkicked')
        else:
            sentence_new = copy.deepcopy(sentence)
        params = {'utterance': sentence_new}
        sentence_parser_json = get_verb_parser(url, params)
        word = sentence.split(' ')[int(target_position)]
        frame_verb = {}
        frame_index = -1
        if sentence_parser_json:
            for frame_i in range(len(sentence_parser_json['props'])):
                for span_i in range(len(sentence_parser_json['props'][frame_i]['spans'])):
                    if sentence_parser_json['props'][frame_i]['spans'][span_i]['text'] == word and sentence_parser_json['props'][frame_i]['spans'][span_i]['predicate'] == True:
                        frame_index = frame_i; break # find the frame_i of focus verb
            if frame_index != -1:
                for span_dict in sentence_parser_json['props'][frame_index]['spans']:
                    frame_verb[span_dict['label']] = {
                        'span':span_dict['text'],
                        'tokens': []
                        }
            else:
                print('can not find target word parser in sentence: ', sentence)
                return None
            if frame_verb != {}:
                for key in frame_verb.keys():
                    if len(frame_verb[key]['span'].split(' ')) > 1:
                        # 多个词组成的短语用句法
                        results_span=nlp.annotate(frame_verb[key]['span'], properties=props)
                        head_dependent_span = results_span["sentences"][0]['basicDependencies']
                        important_tokens = []
                        root_index = -1
                        for head_dependent_sample in head_dependent_span:
                            if head_dependent_sample['dep'] == 'ROOT':
                                root_index = head_dependent_sample['dependent'];break
                        for head_dependent_sample in head_dependent_span:
                            if head_dependent_sample['dep'] == 'ROOT':
                                important_tokens.append(head_dependent_sample['dependentGloss'])
                            elif head_dependent_sample['dep'] == 'conj' and head_dependent_sample['governor'] == root_index:
                                important_tokens.append(head_dependent_sample['dependentGloss'])
                            elif head_dependent_sample['dep'] == 'compound' and head_dependent_sample['governor'] == root_index:
                                if head_dependent_sample['governor'] > head_dependent_sample['dependent']:
                                    important_tokens.append(head_dependent_sample['dependentGloss'] + '_' + head_dependent_sample['governorGloss'])
                                else:
                                    important_tokens.append(head_dependent_sample['governorGloss'] + '_' + head_dependent_sample['dependentGloss'])

                        frame_verb[key]['tokens'] = important_tokens
                    else:
                        frame_verb[key]['tokens'] = [frame_verb[key]['span']]
                return frame_verb
            else:
                print("can not parse:", sentence)
                return None
        else:
            return None
            

def find_string_all_index(string, span):
    span_index_list = []
    for i in range(len(string) - len(span) + 1):
        if string[i: i + len(span)] == span:
            span_index_list.append(i)
    return span_index_list
dataset_to_file = {
    'MOH-X': ['MOH-X.csv'],
    'TroFi': ['TroFi.csv'],
    'VUA_Verb': ['test.csv', 'train.csv', 'val.csv'],
    'LCC_Verb': ['LCC_Verb.csv'],
}
FULL_MODEL = 'xxxx/stanford-corenlp-full-2018-10-05' # path of your stanford-corenlp-full-2018-10-05 folder 
props = {'timeout': '5000000','annotators': 'pos, parse, depparse, lemma', 'tokenize.whitespace': 'true' ,
        'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
if __name__ == '__main__':
    # URL for the API endpoint
    # refer to https://github.com/jgung/verbnet-parser for the usage of the verbnet parser
    url = 'http://localhost:8080/predict/semantics'
    input_root_path = 'datasets/'
    output_root_path = 'datasets_with_WPDom/verbparse_deparser/'
    dataset = 'MOH-X' 
    nlp = StanfordCoreNLP(FULL_MODEL, lang='en') 
    
    for file_name in dataset_to_file[dataset]:
        examples_sentences_count = []
        count_can_not_phrase_sentence = 0
        count_can_not_be_root_sentence = 0
        count_can_not_be_dependent_parser_sentence = 0
        count_have_no_examples = 0
        count_have_no_examples_deparser = 0
        count_have_no_examples_deparser_other = 0
        input_file_path = os.path.join(input_root_path, dataset, file_name)
        output_file_path = os.path.join(output_root_path, dataset, file_name)
        head = ['sentence','label','target_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser']
        input_dataset = read_tsv(input_file_path)
        if not os.path.exists(os.path.join(output_root_path, dataset)):
            os.makedirs(os.path.join(output_root_path, dataset))
            output_dataset = []
        else:
            if not os.path.exists(output_file_path):
                output_dataset = []
            else:
                output_dataset = read_tsv(output_file_path)
        count_wrong = 0
        examples_dict_parsers = {}

        for sample_i in range(len(output_dataset), len(input_dataset)):
            # sentence pharse
            sentence = input_dataset[sample_i][0]
            target_position = int(input_dataset[sample_i][2])
            if sentence.split(' ')[-1] in puncts:
                tokens = sentence.split(' ')[:-1]
            else:
                tokens = sentence.split(' ')
            frame_verb = sentence_verb_parser(tokens, target_position)
            pos_tag = input_dataset[sample_i][4]
            target_word_origional = sentence.split(' ')[target_position]
            
            # examples parse
            examples_list_ori = eval(input_dataset[sample_i][8]) + eval(input_dataset[sample_i][10]) + eval(input_dataset[sample_i][12])
            examples_list = []
            for example_sentence in examples_list_ori:
                if '(' in example_sentence:
                    start_dec = find_string_all_index(example_sentence, '(')
                    end_dec = find_string_all_index(example_sentence, ')')
                    assert len(start_dec) == len(end_dec)
                    example_sentence_text_3 = ''
                    if len(start_dec) > 0:
                        example_sentence_text_3 += example_sentence[:start_dec[0]]
                        for dec_i in range(len(start_dec)-1):
                            example_sentence_text_3 += example_sentence[end_dec[dec_i]+1:start_dec[dec_i+1]]
                        example_sentence_text_3 += example_sentence[end_dec[-1]+1:]
                    else:
                        example_sentence_text_3 = example_sentence
                    examples_list.append(example_sentence_text_3.strip())
                else:
                    examples_list.append(example_sentence.strip())
            if not frame_verb:
                # 使用dependency parser

                print("the sentence can not be parsed with verbnet parser! ", sentence)
                count_can_not_phrase_sentence += 1
                frame_verb = sentence_dependency_parser(tokens, target_position)
                if not frame_verb:
                    print("the sentence can not be parsed with dependency parser! ", sentence)
                    count_can_not_be_root_sentence += 1
                    frame_verb = sentence_dependency_parser_other(tokens, target_position)
                    if not frame_verb:
                        count_can_not_be_dependent_parser_sentence += 1
            if input_dataset[sample_i][3] in list(examples_dict_parsers.keys()):
                output_dataset.append(input_dataset[sample_i] + [frame_verb] + [examples_dict_parsers[input_dataset[sample_i][3]]])
                write_dataset(output_file_path, head, output_dataset)
                continue
            frame_examples = []
            for example_sentence in examples_list:
                if pos_list[pos_tag] not in ['adverb', 'adjective']:
                    target_word = wnl.lemmatize(target_word_origional, pos_to_wordnet_pos[pos_list[pos_tag]])
                    if (target_word_origional == 'riding' or target_word_origional == 'rides') and pos_list[pos_tag] == 'verb':
                        target_word = 'ride'
                else:
                    target_word = target_word_origional
                example_word_index = -1
                if '...' == example_sentence[:3]:
                    example_sentence = example_sentence[3:]
                example_sent_tokens = split_punct(example_sentence)
                for word_i, word in enumerate(example_sent_tokens):
                    word_lemma = wnl.lemmatize(word.lower(), pos_to_wordnet_pos[pos_list[pos_tag]])
                    if word.lower() == 'kitted': 
                        word_lemma = 'kit'
                    if word.lower() == 'lay' and target_word == 'lie':
                        word_lemma = 'lie'
                    if word.lower() == 'riding' and target_word == 'ride': # wnl.lemmatize('riding', 'v') = 'rid' 
                        word_lemma = 'ride'
                    
                    if word_lemma.lower() == target_word.lower():
                        example_word_index = word_i
                        break
                    elif target_word == 'vaporize' and word_lemma == 'vaporise':
                        example_word_index = word_i
                        break
                    
                if example_word_index != -1:
                    word_parser = sentence_verb_parser(example_sent_tokens, example_word_index)
                    if word_parser:
                        frame_examples.append(word_parser)

                else:
                    print('can not find this word: ', input_dataset[sample_i], example_sentence, target_word)
                    count_wrong += 1

            if frame_examples == []:
                count_have_no_examples +=1
                # 用dependency parser
                for example_sentence in examples_list:
                    if pos_list[pos_tag] not in ['adverb', 'adjective']:
                        target_word = wnl.lemmatize(target_word_origional, pos_to_wordnet_pos[pos_list[pos_tag]])
                        if (target_word_origional == 'riding' or target_word_origional == 'rides') and pos_list[pos_tag] == 'verb':
                            target_word = 'ride'
                    else:
                        target_word = target_word_origional
                    example_word_index = -1
                    if '...' == example_sentence[:3]:
                        example_sentence = example_sentence[3:]
                    example_sent_tokens = split_punct(example_sentence)
                    for word_i, word in enumerate(example_sent_tokens):
                        word_lemma = wnl.lemmatize(word.lower(), pos_to_wordnet_pos[pos_list[pos_tag]])
                        if word.lower() == 'kitted': # wnl.lemmatize 没办法还原 kit这个动词
                            word_lemma = 'kit'
                        if word.lower() == 'lay' and target_word == 'lie':
                            word_lemma = 'lie'
                        if word.lower() == 'riding' and target_word == 'ride': # wnl.lemmatize('riding', 'v') = 'rid' 这个不对 rid 的分词是ridding
                            word_lemma = 'ride'
                        
                        if word_lemma.lower() == target_word.lower():
                            example_word_index = word_i
                            break
                        elif target_word == 'vaporize' and word_lemma == 'vaporise':
                            example_word_index = word_i
                            break
                        
                    if example_word_index != -1:
                        word_parser = sentence_dependency_parser(example_sent_tokens, example_word_index)
                        if word_parser:
                            frame_examples.append(word_parser)

                    else:
                        print('can not find this word: ', input_dataset[sample_i], example_sentence, target_word)

                if frame_examples == []:
                    count_have_no_examples_deparser +=1
                    for example_sentence in examples_list:
                        if pos_list[pos_tag] not in ['adverb', 'adjective']:
                            target_word = wnl.lemmatize(target_word_origional, pos_to_wordnet_pos[pos_list[pos_tag]])
                            if (target_word_origional == 'riding' or target_word_origional == 'rides') and pos_list[pos_tag] == 'verb':
                                target_word = 'ride'
                        else:
                            target_word = target_word_origional
                        example_word_index = -1
                        if '...' == example_sentence[:3]:
                            example_sentence = example_sentence[3:]
                        example_sent_tokens = split_punct(example_sentence)
                        for word_i, word in enumerate(example_sent_tokens):
                            word_lemma = wnl.lemmatize(word.lower(), pos_to_wordnet_pos[pos_list[pos_tag]])
                            if word.lower() == 'kitted': # wnl.lemmatize 没办法还原 kit这个动词
                                word_lemma = 'kit'
                            if word.lower() == 'lay' and target_word == 'lie':
                                word_lemma = 'lie'
                            if word.lower() == 'riding' and target_word == 'ride': # wnl.lemmatize('riding', 'v') = 'rid' 这个不对 rid 的分词是ridding
                                word_lemma = 'ride'
                            
                            if word_lemma.lower() == target_word.lower():
                                example_word_index = word_i
                                break
                            elif target_word == 'vaporize' and word_lemma == 'vaporise':
                                example_word_index = word_i
                                break
                            
                        if example_word_index != -1:
                            word_parser = sentence_dependency_parser_other(example_sent_tokens, example_word_index)
                            if word_parser:
                                frame_examples.append(word_parser)
                        else:
                            print('can not find this word: ', input_dataset[sample_i], example_sentence, target_word)
                        if frame_examples == []:
                            count_have_no_examples_deparser_other += 1
            output_dataset.append(input_dataset[sample_i] + [frame_verb] + [frame_examples])
            examples_dict_parsers[input_dataset[sample_i][3]] = frame_examples

            examples_sentences_count.append(len(frame_examples))

            write_dataset(output_file_path, head, output_dataset)
        print(count_wrong)
        print(dataset)
        print(file_name)
