#!/usr/bin/env python
# -*- encoding: utf-8 -*-

NOUN_LIST = ['NN', 'NNS', 'NNP', 'NNPS']
NOUN_LIST_OTHER = ['NP', 'NPS']
ADJ_LIST = ['JJ', 'JJR', 'JJS']
VERB_LIST = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
VERB_LIST_OTHER = ['VV', 'VVD', 'VVG', 'VVN', 'VVP', 'VVZ', 'VH', 'VHD', 'VHG', 'VHN', 'VHP', 'VHZ']
ADV_LIST = ['RB', 'RBR', 'RBS']


FULL_MODEL = 'xxx/stanford-corenlp-full-2018-10-05' # download "stanford-corenlp-full-2018-10-05" folder from CoreNLP
punctuation = ['.', ',', ':', '?', '!', '(', ')', '"', '[', ']', ';', '\'','‘','’']
metaphor_POS_labels = ['ADJ', 'ADV', 'NOUN', 'PRON', 'PROPN', 'VERB']

import csv
import sys
from corenlp import StanfordCoreNLP
# from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
from tqdm import tqdm
from xml.dom.minidom import parse
import xml.dom.minidom
from collections import Counter
from tqdm import tqdm
import re
from readmdict import MDX, MDD  # pip install readmdict
from pyquery import PyQuery as pq    # pip install pyquery


POS_change = {
    'v': 'VERB',
    'n': 'NOUN',
    'adj': 'ADJ',
    'adv': 'ADV'
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

def split_punc(s):
    word_list = ''.join([" "+x+" " if x in punctuation else x for x in s]).split()
    return [w for w in word_list if len(w) > 0]

if __name__ == '__main__':

    filepath = 'en_small.xml' # Please contact the author of the original paper (Mohler et al., 2016) for this file
    save_file = 'datasets/LCC_Verb/LCC_all.csv'
    DOMTree = xml.dom.minidom.parse(filepath)
    document = DOMTree.documentElement
    document_lang = document.getAttribute("lang")
    statistic_dict = {
        'noun':{
            'nsubj':{
                'metaphorial':0,
                'literal':0
            },
            'compound':{
            'metaphorial':0,
            'literal':0
            }
        },
        'verb':{
            'dobj':{
                'metaphorial':0,
                'literal':0
            },
            'nsubj':{
            'metaphorial':0,
            'literal':0
            },
            'nsubjpass':{
            'metaphorial':0,
            'literal':0
            },
            'xsubj':{
            'metaphorial':0,
            'literal':0
            },
            'agent':{
            'metaphorial':0,
            'literal':0
            }
        },
        'adj':{
            'amod':{
                'metaphorial':0,
                'literal':0
            },
            'nsubj':{
            'metaphorial':0,
            'literal':0
            }
        },
        'adv':{
            'advmod':{
                'metaphorial':0,
                'literal':0
            }
        },       
    }
    final_datasets = []
    text_all = []
    text_metaphor_score_0 = []
    text_metaphor_score_1 = []
    text_metaphor_score_2 = []
    text_metaphor_score_3 = []

    targetconcept_metaphor_score_0 = []
    targetconcept_metaphor_score_1 = []
    targetconcept_metaphor_score_2 = []
    targetconcept_metaphor_score_3 = []

    sourceconcept_metaphor_score_0 = []
    sourceconcept_metaphor_score_1 = []
    sourceconcept_metaphor_score_2 = []
    sourceconcept_metaphor_score_3 = []

    label_list = []
    metaphor_token_count = 0
    all_token_count = 0
    source_token_count = 0
    target_token_count = 0
    have_source_concept_count = 0
    sentence_length_list = []    
    metaphor_sentence_length_list = []
    source_lemma_list = [] # 词的原型
    source_POS_list = [] #  词性
    source_dep_list = [] # syntactic dependency
    source_i_list = []
    source_head_i_list = []
    target_lemma_list = [] # 词的原型
    target_POS_list = [] #  词性
    target_dep_list = [] # syntactic dependency
    target_i_list = []
    target_head_i_list = []
    source_list = []
    target_list = []

    # 导入spacy 

    nlp = StanfordCoreNLP(FULL_MODEL, lang='en') 

    # 保存至文件内需要的list
    id_list = []
    word_pair = []
    word_type = []
    word_type_multi = [] # 更细致的分类
    dep_list = []
    metaphor_word_index = []
    target_index = []
    metaphor_word_pos = []
    target_pos = []
    metaphor_score = []
    metaphor_word_definition_list = []
    target_word_definition_list = []
    index_all = 0

    count_metaphor_word_miss = 0
    count_target_word_miss = 0
    count_metaphor_target_word_miss = 0
    count_metaphor_word_miss_list = []
    count_target_word_miss_list = []
    count_metaphor_target_word_miss_list = []
    LM_instances = document.getElementsByTagName("LmInstance")
    for LM_instance in tqdm(LM_instances):
    # 在集合中获取段落的句子
        LM_instance_id = LM_instance.getAttribute("id")
        LM_instance_docid = LM_instance.getAttribute("docid")
        LM_instance_targetConcept = LM_instance.getAttribute("targetConcept")
        LM_instance_type = LM_instance.getAttribute("type")
        LM_instance_chain = LM_instance.getAttribute("chain")
        TextContent = LM_instance.getElementsByTagName("TextContent")[0]


        text_list = []
        text_list_ = [] # [source] 【target】
        source_in_sentence = []
        target_in_sentence = []
        for text_span in TextContent.getElementsByTagName("Current")[0].childNodes:
            if text_span.nodeName == '#text':
                text_list.append(text_span.data)
                text_list_.append(text_span.data)
            elif text_span.nodeName == 'LmSource':
                text_list.append(text_span.firstChild.data)
                text_list_.extend(['[', text_span.firstChild.data, ']'])
                metaphor_token_count += len(text_span.firstChild.data.split(' '))

                LmSource = text_span.firstChild.data
                source_token_count += 1
                source_list.append(LmSource)
                source_in_sentence.append(LmSource)
            elif text_span.nodeName == 'LmTarget':
                text_list.append(text_span.firstChild.data)
                text_list_.extend(['【', text_span.firstChild.data, '】'])
                LmTarget = text_span.firstChild.data
                target_token_count += 1
                target_list.append(LmTarget)
                target_in_sentence.append(LmTarget)
        text_all.append(' '.join(text_list_))
        all_token_count += len(' '.join(text_list).split(' '))


        Annotations = LM_instance.getElementsByTagName("Annotations")[0]
        MetaphoricityAnnotations = Annotations.getElementsByTagName("MetaphoricityAnnotations")[0].getElementsByTagName("MetaphoricityAnnotation")[0]
        MetaphoricityAnnotations_id = MetaphoricityAnnotations.getAttribute("id")
        MetaphoricityAnnotations_score = MetaphoricityAnnotations.getAttribute("score")
        label_list.append(int(float(MetaphoricityAnnotations_score)))

        sourceConcept_score_list = []
        if Annotations.getElementsByTagName("CMSourceAnnotations") != []:
            CMSourceAnnotations_list = Annotations.getElementsByTagName("CMSourceAnnotations")[0].getElementsByTagName("CMSourceAnnotation")
            if len(CMSourceAnnotations_list) > 1:
                for CMSourceAnnotations in CMSourceAnnotations_list:
                    CMSourceAnnotations_id = CMSourceAnnotations.getAttribute("id")
                    CMSourceAnnotations_score = CMSourceAnnotations.getAttribute("score")
                    CMSourceAnnotations_sourceConcept = CMSourceAnnotations.getAttribute("sourceConcept")
                    sourceConcept_score_list.append((CMSourceAnnotations_sourceConcept, CMSourceAnnotations_score))


        sentence = ''.join(text_list)
        # 检查source和target是否在http网址中
        urls_1 = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ''.join(text_list))
        urls_2 = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ''.join(text_list_))
        flag_e_in_url = 0
        if len(urls_1) != len(urls_2):
            flag_e_in_url = 1
        elif len(urls_1) !=0:
            for i, url_1 in enumerate(urls_1): # 如果不相等说明e1_text 或者e2_text在url内
                if url_1 != urls_2[i]:
                    flag_e_in_url = 1

        if not flag_e_in_url:
            # 只要无url的样本
            if MetaphoricityAnnotations_score == '0.0':
                text_metaphor_score_0.append(' '.join(text_list_))
                targetconcept_metaphor_score_0.append(LM_instance_targetConcept)
                sourceconcept_metaphor_score_0.append(sourceConcept_score_list)
            elif MetaphoricityAnnotations_score == '1.0':
                text_metaphor_score_1.append(' '.join(text_list_))
                targetconcept_metaphor_score_1.append(LM_instance_targetConcept)
                sourceconcept_metaphor_score_1.append(sourceConcept_score_list)
            elif MetaphoricityAnnotations_score == '2.0':
                text_metaphor_score_2.append(' '.join(text_list_))
                targetconcept_metaphor_score_2.append(LM_instance_targetConcept)
                sourceconcept_metaphor_score_2.append(sourceConcept_score_list)
            elif MetaphoricityAnnotations_score == '3.0':
                text_metaphor_score_3.append(' '.join(text_list_))
                targetconcept_metaphor_score_3.append(LM_instance_targetConcept)
                sourceconcept_metaphor_score_3.append(sourceConcept_score_list)
            final_datasets.append([' '.join(text_list_), MetaphoricityAnnotations_score, LM_instance_targetConcept, sourceConcept_score_list])


        else:
            sentence_ = re.sub(r'[\s|\"]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[\s|\"]*', ' ', sentence)
            sentence__ = re.sub(r'[\s|\"]*www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[\s|\"]*', ' ', sentence_)




    head = ['sentence', 'label', 'targetconcept', 'source_concept']
    write_dataset(save_file, head, final_datasets)
        
