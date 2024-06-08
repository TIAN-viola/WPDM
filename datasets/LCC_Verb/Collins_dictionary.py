#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from readmdict import MDX, MDD  # pip install readmdict
from pyquery import PyQuery as pq    # pip install pyquery
import html
from html.parser import HTMLParser

def find_string_all_index(string, span):
    span_index_list = []
    for i in range(len(string) - len(span) + 1):
        if string[i: i + len(span)] == span:
            span_index_list.append(i)
    return span_index_list




def remove_string(string, span_class):
    if span_class not in string:
        return string
    start_index_list = find_string_all_index(string, span_class)
    final_string = ''
    end_index_list = []
    for index in start_index_list:
        span = string[index + len(span_class):]
        for i in range(len(span) - len('</span>') + 1):
            if span[i: i + len('</span>')] == '</span>':
                end_index_list.append(i + index + len(span_class))
                break
    assert len(start_index_list) == len(end_index_list)
    
    final_string = string[:start_index_list[0]]
    for i in range(len(start_index_list)):
        if i < len(start_index_list) - 1:
            final_string += string[end_index_list[i] + len('</span>'):start_index_list[i+1]]
    final_string += string[end_index_list[-1] + len('</span>'):]
    return final_string



colloins_pos_to_pos ={
    'verb': 'verb',
    'n-uncount': 'noun',
    'n-var': 'noun',
    'n-count': 'noun',
    'n-sing':'noun',
    'adj':'adjective',
    'adv':'adverb'
}

def repair_sth_string(string):
    output_string = ''
    indexes = find_string_all_index(string, '/>')
    if len(indexes) == 0:
        return string
    elif len(indexes) == 1:
        output_string += string[:indexes[0]]
        for index_forward in range(indexes[0], -1, -1):
            if string[index_forward] == '<':
                if string[index_forward: index_forward+2] == '<s':
                    output_string += '></span>'
                elif string[index_forward: index_forward+2] == '<a':
                    output_string += '></a>'
                else:
                    print('sth new, ', string[index_forward:])
                break
        output_string += string[indexes[-1] + 2:]
        return output_string
    else:
        output_string += string[:indexes[0]]
        for i, index in enumerate(indexes[:-1]):
            for index_forward in range(index, -1, -1):
                if string[index_forward] == '<':
                    if string[index_forward: index_forward+2] == '<s':
                        output_string += '></span>'
                    elif string[index_forward: index_forward+2] == '<a':
                        output_string += '></a>'
                    else:
                        print('sth new, ', string[index_forward:])
                    break
            output_string += string[index + 2: indexes[i+1]]
        output_string += string[indexes[-1]+2:]
        return output_string
def collins_html_parser(word, target_pos, headwords_oxford, items_html_oxford):

    # 加载词典

    # first_n 前几个同义词
    try:
        wordIndex = headwords_oxford.index(word.encode())
    except:
        print('no such word in dic', word, target_pos)
        return '', []
    word,html = items_html_oxford[wordIndex]
    word,html = word.decode(), html.decode().replace('&nbsp;', '\xa0').replace('&amp;','&')
    #print(word, html)

    # 从html中提取需要的部分，到这一步需要根据自己查询的字典html格式，自行调整。
    doc_all = pq(html)
    if str(doc_all('div[class="note type-drv example"]')) != '':
        notes_1 = HTMLParser().unescape(repair_sth_string(str(doc_all('div[class="note type-drv example"]'))))
    else:
        notes_1 = ''
    if str(doc_all('div[class="note type-sense example"]')) != '':
        notes_2 = HTMLParser().unescape(repair_sth_string(str(doc_all('div[class="note type-sense example"]'))))
    else:
        notes_2 = ''

    poses_text = HTMLParser().unescape(str(doc_all('span[class="st"]')))
    poses_starts = find_string_all_index(poses_text, '<span class="st"')
    poses_starts.append(len(poses_text))
    poses = doc_all('span[class="st"]')
    poses_html_indexes = []
    pos_list = []
    for pos_i, pos in enumerate(poses):

        pos_index = html.index(poses_text[poses_starts[pos_i]:poses_starts[pos_i+1]])
        poses_html_indexes.append(pos_index)
        if 'n-' in pos.text.strip().lower():
            pos_list.append('noun')
        elif 'v-' in pos.text.strip().lower():
            pos_list.append('verb')
        elif 'adj-' in pos.text.strip().lower():
            pos_list.append('adjective')
        else:
            pos_list.append(pos.text.strip().lower())
    poses_html_indexes.append(len(html)) 
    # 寻找第一个target pos
    pos_index_start = -1
    pos_index_end = -1
    for pos_i, pos in enumerate(pos_list):
        if pos == target_pos:
            pos_index_start = poses_html_indexes[pos_i] 
            pos_index_end = poses_html_indexes[pos_i + 1] 
            break
    
    if pos_index_start == -1:
        print("no such pos", word, target_pos)
        return '', []
    else:
        ## definition 
        definitions_text_ = str(doc_all('div[class="caption hide_cn"]')) 
        definitions_text = repair_sth_string(definitions_text_)
        definitions_starts = find_string_all_index(definitions_text, '<div class="caption hide_cn')
        definitions_starts.append(len(definitions_text))
        definition_start_indexes = []
        for def_start_index_i, def_start_index in enumerate(definitions_starts[:-1]):
            if definitions_text[def_start_index:definitions_starts[def_start_index_i+1]] in notes_1:
                continue
            tag_indexes = find_string_all_index(definitions_text[def_start_index:definitions_starts[def_start_index_i+1]], '>')
            tag_index_1 = definitions_text[def_start_index:definitions_starts[def_start_index_i+1]].index('<a class="anchor" name="')
            tag_index_2 = -1
            for tag_index_i in range(tag_index_1+24, len(definitions_text)):
                if definitions_text[tag_index_i] == '"':
                    tag_index_2 = tag_index_i
                    break
            if tag_index_2 != -1:
                definition_start_indexes.append(html.index(definitions_text[def_start_index:definitions_starts[def_start_index_i+1]][:tag_index_2]))
            else:
                definition_start_indexes.append(html.index(definitions_text[def_start_index:definitions_starts[def_start_index_i+1]][:tag_indexes[1]]))
        definition_start_indexes.append(len(html))

        # 找到当前pos下的第一个definition索引
        definition_index_start = -1 # html的索引
        definition_index_end = -1
        for def_start_index_i, def_start_index in enumerate(definition_start_indexes[:-1]):

            if pos_list[def_start_index_i] == target_pos:
                definition_index_start = def_start_index
                definition_index_end = definition_start_indexes[def_start_index_i+1]
                first_definition_1_ = definitions_text[definitions_starts[def_start_index_i]
                :definitions_starts[def_start_index_i+1]]
                if '<span class="st"' not in first_definition_1_:
                    continue
                st_index = first_definition_1_.index('<span class="st"')
                first_definition_1 = first_definition_1_[st_index:]
                first_definition_2 = remove_string(first_definition_1, '<span class="num">')
                first_definition_3 = remove_string(first_definition_2, '<span class="def_cn cn_before">')
                first_definition_4 = remove_string(first_definition_3, '<span class="def_cn cn_after">')
                first_definition_ = remove_string(first_definition_4, '<span class="chinese-text">')
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
                if '   ' in first_definition:
                    first_definition_final = ' '.join(first_definition.split('   ')[1:]).strip()
                else:
                    first_definition_final = first_definition
                break
        if definition_index_start == -1:
            print("no definition in the target pos", word, target_pos)
            return '', []
        else:
            # definition text


            examples_list = []

            example_sentences = doc_all('p')
            example_sentences_text = str(doc_all('p')).replace('&nbsp;', '\xa0').replace('&amp;','&')
            example_sentences_start_indexes = find_string_all_index(example_sentences_text, '<p>')
            example_sentences_start_indexes.append(len(example_sentences_text))

            for example_i, exp_start_index in enumerate(example_sentences_start_indexes[:-1]):
                example_text_ = example_sentences_text[example_sentences_start_indexes[example_i]
                :example_sentences_start_indexes[example_i+1]]
                if example_text_ in notes_1 or example_text_ in notes_2:
                    continue
                if 'chinese-text' in example_text_:
                    continue
                if ' <p/> ' == example_text_[-6:]:
                    example_text_ = example_text_[:-6]
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
                    # 去掉 []
                    start_b = find_string_all_index(next_element_text, '[')
                    end_b = find_string_all_index(next_element_text, ']')
                    if len(start_b) == len(end_b) + 1:
                        next_element_text = next_element_text + ']'
                        end_b.append(len(next_element_text)-1)
                    elif len(start_b) == len(end_b):
                        next_element_text = next_element_text
                    else:
                        print(next_element_text)
                        print('wrong!')
                        
                    next_element_text_b = ''
                    if len(start_b) > 0:
                        next_element_text_b += next_element_text[:start_b[0]]
                        for dec_i in range(len(start_b)-1):
                            next_element_text_b += next_element_text[end_b[dec_i]+1:start_b[dec_i+1]]
                        
                        next_element_text_b += next_element_text[end_b[-1]+1:]
                    else:
                        next_element_text_b = next_element_text

                    examples_list.append(next_element_text_b.strip())
            return first_definition_final, examples_list


                
            