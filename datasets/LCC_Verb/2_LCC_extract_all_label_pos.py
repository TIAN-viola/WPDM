#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import csv
from corenlp import StanfordCoreNLP
punctuation = ['.', ',', ':', '?', '!', '(', ')', '"', ';', '\'', '/','‘','’', '-', '“', '”', '$']

def split_punc(s):
    if s == '''This presentation explores how credit/ 【 debt 】   [ circulates ]  globally.''':
        print(s)
    s_ = ''
    for i, x in enumerate(s):
        if x in punctuation:
            if i-1 >= 0 and i+1<len(s):
                if s[i-1] != ' ' and s[i+1] != ' ':
                    s_ += " "+x+" "
                elif s[i-1] != ' ' and s[i+1] == ' ':
                    s_ += " "+x
                elif s[i-1] == ' ' and s[i+1] == ' ':
                    s_ += x
                elif s[i-1] == ' ' and s[i+1] != ' ':
                    s_ += x + " "
            elif i-1 >= 0 and i+1>=len(s):
                if s[i-1] != ' ':
                    s_ += " "+x
                elif s[i-1] == ' ':
                    s_ += x
            elif i-1 <= 0 and i+1<len(s):
                if s[i+1] == ' ':
                    s_ += x
                elif s[i+1] != ' ':
                    s_ += x + " "
            elif i-1 <= 0 and i+1>=len(s):
                s_ += x
        else:
            s_ += x
    for punct in punctuation:
        if ' ] '+punct in s_:
            s_ = s_.replace(' ] '+punct, ' ]  '+punct)
        if punct+' [ ' in s_:
            s_ = s_.replace(punct+' [ ', punct+'  [ ')
        if ' 】 '+punct in s_:
            s_ = s_.replace(' 】 '+punct, ' 】  '+punct)
        if punct+' 【 ' in s_:
            s_ = s_.replace(punct+' 【 ', punct+'  【 ')
    return s_


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

def find_index_source(sentence):
    sentence_remove_target = sentence.replace(' 】 ', '').replace(' 【 ', '').replace(' 】', '').replace('【 ', '')
    if ' [ ' not in sentence_remove_target:
        index_start =sentence_remove_target.index('[ ')
        span = 2
    else:
        index_start = sentence_remove_target.index(' [ ')
        span = 3
    if ' ] ' not in sentence_remove_target:
        index_end = sentence_remove_target.index(' ]')
        span_end = 2
    else:
        index_end = sentence_remove_target.index(' ] ')
        span_end = 3
    previous_text = sentence_remove_target[:index_start].strip()
    current_text = sentence_remove_target[index_start+span:index_end].strip()
    end_text = sentence_remove_target[index_end+span_end:].strip()

    if len(current_text.split(' ')) > 1:
        return -1, sentence # source target 是多个单词的不要
    if len(previous_text) == 0:
        index = 0
        if len(end_text) == 0:
            return index, current_text
        else:
            sentence_ = current_text + ' ' + end_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_
    else:
        index = len(previous_text.split(' '))
        if len(end_text) == 0:
            sentence_ = previous_text + ' ' + current_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_
        else:
            sentence_ = previous_text + ' ' + current_text + ' ' + end_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_

def find_index_target(sentence):

    sentence_remove_target = sentence.replace(' ] ', '').replace(' [ ', '')
    if ' ] ' not in sentence_remove_target:
        sentence_remove_target = sentence_remove_target.replace(' ]', '')
    if ' [ ' not in sentence_remove_target:
        sentence_remove_target = sentence_remove_target.replace('[ ', '')
    if ' 【 ' not in sentence_remove_target:
        index_start =sentence_remove_target.index('【 ')
        span = 2
    else:
        index_start = sentence_remove_target.index(' 【 ')
        span = 3
    if ' 】 ' not in sentence_remove_target:
        index_end = sentence_remove_target.index(' 】')
        span_end = 2
    else:
        index_end = sentence_remove_target.index(' 】 ')
        span_end = 3
    previous_text = sentence_remove_target[:index_start].strip()
    current_text = sentence_remove_target[index_start+span:index_end].strip()
    end_text = sentence_remove_target[index_end+span_end:].strip()
    if len(current_text.split(' ')) > 1:
        return -1, sentence # source target 是多个单词的不要
    if len(previous_text) == 0:
        index = 0
        if len(end_text) == 0:
            return index, current_text
        else:
            sentence_ = current_text + ' ' + end_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_
    else:
        index = len(previous_text.split(' '))
        if len(end_text) == 0:
            sentence_ = previous_text + ' ' + current_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_
        else:
            sentence_ = previous_text + ' ' + current_text + ' ' + end_text
            assert sentence_.split(' ')[index] == current_text
            return index, sentence_

FULL_MODEL = 'xxxx/stanford-corenlp-full-2018-10-05'    # your stanford-corenlp-full-2018-10-05 folder path        
nlp = StanfordCoreNLP(FULL_MODEL, lang='en') 
props = {'timeout': '5000000','annotators': 'pos, parse, depparse, lemma', 'tokenize.whitespace': 'true' ,
        'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
def get_pos_tag(sentence, target_word):
    results=nlp.annotate(sentence, properties=props)

    tokens = results['sentences'][0]['tokens']
    pos_tag = ''
    lemma = ''
    for token_dict in tokens:
        if token_dict['originalText'] == target_word:
            pos_tag = token_dict['pos']
            lemma = token_dict['lemma']
            break
    return pos_tag, lemma
label_project = {
    '0.0':0,
    '2.0':1,
    '3.0':1,
    '1.0':0
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
    'ADV': 'adverb',
    'IN': 'prep',
}


if __name__ == '__main__':
    input_file = 'datasets/LCC_Verb/LCC_all.csv'
    input_dataset =  read_tsv(input_file)
    save_file = 'datasets/data-basicmeaning-examples/LCC_Verb/LCC_Verb.csv'
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_no_pos = 0
    output_dataset = []
    for sample_i, sample in enumerate(input_dataset):
        # print(sample_i)
        sentence_ori,label,targetconcept,source_concept = sample
        sentence = split_punc(sentence_ori)
        
        # remove the noise samples
        if '*' in sentence or 'http' in sentence or '\xa0' in sentence or '\x92s' in sentence or '\x93' in sentence or '\x94' in sentence:
            continue
        if  'Maxwell in 1865 shows that the wave speed derived by differentially' in sentence_ori or '[...] Fortunately, by using email marketing' in sentence_ori or 'Accordingly, all these circulars/clarifications/' in sentence_ori or 'Accordingly, all these circulars/clarifications/ instructions' in sentence_ori or 'She fitted out as Wolf, with 4 by 15 [ cm ]' in sentence_ori or 'Moewe, the Armed Merchant Raider had captured her on the 11th.' in sentence_ori or 'The good guy bruce wayne is a  【 rich 】  1 % [ er ]  and the bad guys look like occupiers.' in sentence_ori or 'To some people, $35 [ k ]  might be  【 rich 】 .' in sentence_ori or 'Too contrived to be effectively self-referential, this scene is slowed down' in sentence_ori or 'wpp2003 Guts In The  [ Edge ]  Of  【 Wealth 】  by Deni Khanafiah [Downloadable!]' in sentence_ori or 'It also now  [ lists ]  them in the [[Category: 【 Socialism 】 ]] page.' in sentence_ori or '[ Floating ]   【 Ideas 】 ™' in sentence_ori or "Does God answer prayers, or do we—by ardently pursuing our  [ heart ] 's  【 desires 】 —answer our own prayers, gift and bless ourselves?" in sentence_ori or "For each possible environmental state the `` 【 belief 】   [ vector ] '' provides the agent's estimate of the probability of currently being in this state." in sentence_ori:
            continue
        if label == '1.0':
            count_1 += 1
        if label != '-1.0' and label != '1.0':
            # split_punc(sentence)
            index_source, sentence_clean_source = find_index_source(sentence)

            if index_source != -1:
                source_word = sentence_clean_source.split(' ')[int(index_source)]
                pos_tag, lemma = get_pos_tag(sentence, source_word)
                
                
                if pos_tag == '':
                    print(sample)
                    count_no_pos += 1
                else:
                    if pos_tag not in list(pos_list.keys()):
                        continue
                    if pos_list[pos_tag] == 'verb':
                        output_dataset.append([sentence_clean_source, label_project[label], index_source, source_word, 'VERB', '', ''])
                        if label == '0.0':
                            count_0 += 1
                        elif label == '2.0':
                            count_2 += 1
                        elif label == '3.0':
                            count_3 += 1

                        else:
                            print(label)
                        head = ['sentence', 'label', 'target_position', 'target_word', 'pos_tag', 'gloss', 'eg_sent']
                        write_dataset(save_file, head, output_dataset)
                    

    
    print('0:', count_0)
    print('1:', count_1)
    print('2:', count_2)
    print('3:', count_3)
    print('count_no_pos', count_no_pos)
    print('all_sentence:', count_0+count_2+count_3)

    
# 0: 1153
# 1: 459
# 2: 405
# 3: 451
# count_no_pos 1
# all_sentence: 2009

