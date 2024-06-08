#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from readmdict import MDX, MDD  # pip install readmdict
from pyquery import PyQuery as pq    # pip install pyquery

pos_to_wordnet_pos = {
    'verb': 'v',
    'noun': 'n',
    'adjective': 'a'
}

def find_string_all_index(string, span):
    span_index_list = []
    for i in range(len(string) - len(span) + 1):
        if string[i: i + len(span)] == span:
            span_index_list.append(i)
    return span_index_list

def remove_colloinexa(string):
    start_index_list = find_string_all_index(string, '<span class="COLLOINEXA">')
    final_string = ''
    end_index_list = []
    for index in start_index_list:
        span = string[index + 25:]
        for i in range(len(span) - len('</span>') + 1):
            if span[i: i + len('</span>')] == '</span>':
                end_index_list.append(i + index + 25)
                break
    assert len(start_index_list) == len(end_index_list)
    final_string = string[:start_index_list[0]]
    for i in range(len(start_index_list)):
        final_string += string[start_index_list[i]+25:end_index_list[i]]
        if i < len(start_index_list) - 1:
            final_string += string[end_index_list[i] + len('</span>'):start_index_list[i+1]]
    final_string += string[end_index_list[-1] + len('</span>'):]
    return final_string


def longman_html_parser(word, target_pos, headwords_oxford, items_html_oxford):

    try:
        wordIndex = headwords_oxford.index(word.encode())
    except:
        print('no such word in dic', word, target_pos)
        return '', []
    word,html = items_html_oxford[wordIndex]
    word,html = word.decode(), html.decode().replace('&amp;','&')
    #print(word, html)
    doc_all = pq(html)

    # pos 列表
    # verb
    # phrasal verb
    # phrasal verb
    # phrasal verb
    # noun
    # adjective
    # adverb
    # prefix
    # verb
    # verb
    # noun
    # adjective
    # prefix

    pos_index = -1
    pos_text = ''
    for i_pos, pos in enumerate(doc_all('span[class="lm5pp_POS"]')):
        if pos.text == None:
            continue
        if pos.text.strip() == target_pos:
            pos_index = i_pos
            pos_text = pos.text
            break
    # 有些注释里面的内容不要  grammar use等等
    boxpanel = str(doc_all('span[class="BoxPanel"]'))
    if word == 'flash':
        print(word)
    if pos_index != -1:
        pos_start_index = html.index('<span class="lm5pp_POS">'+pos_text+'</span>')
        meaning_start_index = 0
        meaning_end_index = len(html)

        meanings_text = str(doc_all('span[class="DEF LDOCE_switch_lang switch_siblings"]')).replace('&amp;','&')
        meanings_indexes = find_string_all_index(meanings_text, '<span class="DEF LDOCE_switch_lang switch_siblings')
        meanings_indexes.append(len(meanings_text))
        
        meanings_indexes_start = []
        meanings_indexes_end = []
        # meaning 过滤 中文的不要
        for meaning_i, meaning_index in enumerate(meanings_indexes[:-1]):
            meaning_i_start_index = html.index(meanings_text[meaning_index: meanings_indexes[meaning_i + 1]])
            meaning_text = meanings_text[meaning_index: meanings_indexes[meaning_i + 1]]
            if '"cn_txt"' not in meaning_text:
                meanings_indexes_start.append(meaning_index)
                meanings_indexes_end.append(meanings_indexes[meaning_i+1])
        first_definition = ''
        # 全部都是中文的
        for meaning_i, meaning_index in enumerate(meanings_indexes_start):
            meaning_i_start_index = html.index(meanings_text[meaning_index: meanings_indexes_end[meaning_i]])
            if meaning_i_start_index > pos_start_index:
                meaning_start_index = meaning_i_start_index
                first_token_index = meanings_text[1:].index('>')
                first_definition_ = meanings_text[meaning_index+first_token_index+2: meanings_indexes_end[meaning_i]-7]
                start_definition_dec = find_string_all_index(first_definition_, '<')
                end_definition_dec = find_string_all_index(first_definition_, '>')
                assert len(start_definition_dec) == len(end_definition_dec)
                if len(start_definition_dec) > 0:
                    first_definition += first_definition_[:start_definition_dec[0]]
                    for dec_i in range(len(start_definition_dec)-1):
                        first_definition += first_definition_[end_definition_dec[dec_i]+1:start_definition_dec[dec_i+1]]
                    first_definition += first_definition_[end_definition_dec[-1]+1:]
                else:
                    first_definition = first_definition_
                if meaning_i < len(meanings_indexes_start) -1:
                    meaning_end_index = html.index(meanings_text[meanings_indexes_start[meaning_i+1]: meanings_indexes_end[meaning_i+1]])
                break
        
        example_sentences_text = str(doc_all('span[class="EXAMPLE"]')).replace('&amp;','&')
        example_sentences_indexes = find_string_all_index(example_sentences_text, '<span class="EXAMPLE"')
        example_sentences_indexes.append(len(example_sentences_text))
        example_sentences_list = []
        for example_i, example_index in enumerate(example_sentences_indexes[:-1]):
            example_start_index = html.index(example_sentences_text[example_index: example_sentences_indexes[example_i + 1]])
            example_end_index = example_start_index + len(example_sentences_text[example_index: example_sentences_indexes[example_i + 1]])
            if example_start_index > meaning_start_index and example_end_index < meaning_end_index:
                example_sentence_text = example_sentences_text[example_index: example_sentences_indexes[example_i + 1]].strip()[22:-7]
                if example_sentence_text in boxpanel:
                    continue
                if '</a>' in example_sentence_text:
                    index_1 = example_sentence_text.index('</a>')
                    example_sentence_text_ = example_sentence_text[index_1+4:]
                else:
                    example_sentence_text_ = example_sentence_text
                if '<span class="english LDOCE_switch_lang switch_children">' in example_sentence_text_:
                    index_1 = example_sentence_text_.index('<span class="english LDOCE_switch_lang switch_children">')
                    if '</span>' == example_sentence_text_[-7:]:
                        example_sentence_text_1 = example_sentence_text_[index_1+56:-7]
                    else:
                        print(example_sentence_text_)
                        print('something wrong')
                else:
                    example_sentence_text_1 = example_sentence_text_
                if '<span class="cn_txt">' in example_sentence_text_1:
                    index_2 = example_sentence_text_1.index('<span class="cn_txt">')
                    start_span = []
                    end_span = []
                    for index_i in range(index_2, len(example_sentence_text_1)-5):
                        if example_sentence_text_1[index_i:index_i+5] == '<span':
                            start_span.append(index_i)
                        elif example_sentence_text_1[index_i:index_i+5] == '</spa':
                            end_span.append(index_i)
                    index_3 = end_span[len(start_span)-1] + len('</span>')
                    example_sentence_text_2_1 = example_sentence_text_1[:index_2] + example_sentence_text_1[index_3:]
                else:
                    example_sentence_text_2_1 = example_sentence_text_1
                if '<span class="GLOSS">' in example_sentence_text_2_1:
                    index_2_1 = example_sentence_text_2_1.index('<span class="GLOSS">')
                    start_span = []
                    end_span = []
                    for index_i in range(index_2_1, len(example_sentence_text_2_1)-5):
                        if example_sentence_text_2_1[index_i:index_i+5] == '<span':
                            start_span.append(index_i)
                        elif example_sentence_text_2_1[index_i:index_i+5] == '</spa':
                            end_span.append(index_i)
                    index_3_1 = end_span[len(start_span)-1] + len('</span>')
                    example_sentence_text_2 = example_sentence_text_2_1[:index_2_1] + example_sentence_text_2_1[index_3_1:]
                else:
                    example_sentence_text_2 = example_sentence_text_2_1
                if '<span class="COLLOINEXA">' in example_sentence_text_2:
                    example_sentence_text_3_ = remove_colloinexa(example_sentence_text_2)
                else:
                    example_sentence_text_3_ = example_sentence_text_2
                start_dec = find_string_all_index(example_sentence_text_3_, '<')
                end_dec = find_string_all_index(example_sentence_text_3_, '>')
                assert len(start_dec) == len(end_dec)
                example_sentence_text_3 = ''
                if len(start_dec) > 0:
                    example_sentence_text_3 += example_sentence_text_3_[:start_dec[0]]
                    for dec_i in range(len(start_dec)-1):
                        example_sentence_text_3 += example_sentence_text_3_[end_dec[dec_i]+1:start_dec[dec_i+1]]
                    example_sentence_text_3 += example_sentence_text_3_[end_dec[-1]+1:]
                else:
                    example_sentence_text_3 = example_sentence_text_3_

                flag_exist_word = False
                for word_temp in example_sentence_text_3.split(' '):
                    if target_pos not in ['adverb', 'adjective']:
                        word_lemma = wnl.lemmatize(word_temp, pos_to_wordnet_pos[target_pos])  
                    else:
                        word_lemma = word_temp
                    if word in word_lemma or word in word_temp:
                        flag_exist_word = True                 
                if flag_exist_word:
                    example_sentences_list.append(example_sentence_text_3)
                if '<span' in example_sentence_text_3:
                    print(example_sentence_text_3)
            elif example_end_index > meaning_end_index:
                break

        return first_definition, example_sentences_list


    else:
        # no such target_pos
        print("no such pos", word, target_pos)
        return '', []
