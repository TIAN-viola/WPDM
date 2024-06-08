#!/usr/bin/env python
# -*- encoding: utf-8 -*-


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

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def find_string_all_index(string, span):
    span_index_list = []
    for i in range(len(string) - len(span) + 1):
        if string[i: i + len(span)] == span:
            span_index_list.append(i)
    return span_index_list
def oxford_html_parser(word, target_pos, headwords_oxford, items_html_oxford):
    # first_n 前几个同义词
    try:
        wordIndex = headwords_oxford.index(word.encode())
    except:
        print('no such word in dic', word, target_pos)
        return '', []
    word,html = items_html_oxford[wordIndex]
    word,html = word.decode(), html.decode()

    doc_all = pq(html)
    word_parse = []
    poses = doc_all('span[class="pos"]')
    pos_list = {} # {词性：索引}
    # 需要排除的同义词框框内的东西
    synonyms = str(doc_all('span[otitle="Synonyms"]'))
    # 排除黄色框里面注释的东西
    notes_1 = str(doc_all('span[class="un"]'))
    # 额外的注释 verb forms extra examples等等
    notes_2 = str(doc_all('span[class="collapse"]'))
    pos_indexes = [] # pos html index
    pos_list = []
    
    for pos in poses:
        pos_index = html.index(str(doc_all('span[id="'+pos.attrib['id']+'"]')))
        pos_indexes.append(pos_index)
        pos_list.append(pos.text)
    
    pos_indexes.append(len(html))

    pos_index_start = -1
    pos_index_end = -1
    for pos_i, pos in enumerate(pos_list):
        if pos == target_pos:
            pos_index_start = pos_indexes[pos_i] 
            pos_index_end = pos_indexes[pos_i + 1] 
            break
    
    if pos_index_start == -1:
        print("no such pos", word, target_pos)
        return '', []
    else:
        definitions_text = str(doc_all('span[class="def"]'))
        definitions_starts = find_string_all_index(definitions_text, '<span class="def"')
        definitions_starts.append(len(definitions_text))
        definition_start_indexes = []
        for def_start_index_i, def_start_index in enumerate(definitions_starts[:-1]):
            definition_start_indexes.append(html.index(definitions_text[def_start_index:definitions_starts[def_start_index_i+1]]))
        definition_start_indexes.append(len(html))

        # 找到当前pos下的第一个definition索引
        definition_index_start = -1 # html的索引
        definition_index_end = -1
        for def_start_index_i, def_start_index in enumerate(definition_start_indexes[:-1]):
            if def_start_index > pos_index_start and def_start_index < pos_index_end:
                definition_index_start = def_start_index
                definition_index_end = definition_start_indexes[def_start_index_i+1]
                first_definition_ = definitions_text[definitions_starts[def_start_index_i]
                :definitions_starts[def_start_index_i+1]]
                start_def = find_string_all_index(first_definition_, '<')
                end_def = find_string_all_index(first_definition_, '>')
                assert len(start_def) == len(end_def)
                first_definition = ''
                if len(start_def) > 0:
                    first_definition += first_definition_[:start_def[0]]
                    for dec_i in range(len(start_def)-1):
                        first_definition += first_definition_[end_def[dec_i]+1:start_def[dec_i+1]]
                    first_definition += first_definition_[end_def[-1]+1:]
                else:
                    first_definition = first_definition_
                
                break
        if definition_index_start == -1:
            print("no definition in the target pos", word, target_pos)
            return '', []
        else:

            # definition text
            examples_list = []
            # example list 找到符合当前definition下的例句
            example_sentences = doc_all('span[class="x"]')
            example_sentences_text = str(doc_all('span[class="x"]'))
            example_sentences_start_indexes = find_string_all_index(example_sentences_text, '<span class="x"')
            example_sentences_start_indexes.append(len(example_sentences_text))

            for example_i, exp_start_index in enumerate(example_sentences_start_indexes[:-1]):
                example_text_ = example_sentences_text[example_sentences_start_indexes[example_i]
                :example_sentences_start_indexes[example_i+1]]
                if example_text_ in synonyms or example_text_ in notes_1 or example_text_ in notes_2:
                    continue
                exp_start_index_html = html.index(example_text_)
                if exp_start_index_html > definition_index_start and exp_start_index_html < definition_index_end:
                    start_dec = find_string_all_index(example_text_, '<')
                    end_dec = find_string_all_index(example_text_, '>')
                    assert len(start_dec) == len(end_dec)
                    next_element_text = ''
                    if len(start_dec) > 0:
                        next_element_text += example_text_[:start_dec[0]]
                        for dec_i in range(len(start_dec)-1):
                            next_element_text += example_text_[end_dec[dec_i]+1:start_dec[dec_i+1]]
                        next_element_text += example_text_[end_dec[-1]+1:]
                    else:
                        next_element_text = example_text_
                    examples_list.append(next_element_text)
            return first_definition, examples_list
