import os
import json
import logging
import re
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import copy
import csv
import sys
import pandas as pd
import numpy as np
import random
import pickle
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
logger = logging.getLogger(__name__)

class InputExampleUnified_Basic(object):
    """A single training/test example for simple sequence classification."""

    def __init__(
        self,
        sentence,
        label,
        core_position,
        verb_parser,
        domain_role_dict,
    ):
        self.sentence=sentence
        self.label=label
        self.core_position=core_position
        self.verb_parser=verb_parser
        self.domain_role_dict=domain_role_dict

class InputFeatures_Basic_wp(object):
    def __init__(
        self,
        input_ids_ori, 
        input_mask_ori,
        label,
        core_word_all_indexs,
        target_words_all_indexes_list,
        core_words_all_indexes_list,
        target_source_scores,
    ):
        self.input_ids_ori=input_ids_ori
        self.input_mask_ori=input_mask_ori
        self.label=label
        self.core_word_all_indexs=core_word_all_indexs
        self.target_words_all_indexes_list=target_words_all_indexes_list
        self.core_words_all_indexes_list=core_words_all_indexes_list
        self.target_source_scores=target_source_scores

class DataProcessorBasic(object):

    def __init__(self, arg):
        self.pos = 'test'


    def get_train_examples(self, data_dir, logger, k=None):
        """See base class."""
        if k is not None:
            return self._create_train_examples(
                self._read_tsv(os.path.join(data_dir, "train" + str(k) + ".csv")), logger)
        else:
            return self._create_train_examples(
                self._read_tsv(os.path.join(data_dir, "train.csv")), logger)


    def get_dev_examples(self, data_dir, logger, k=None):
        """See base class."""
        if k is not None:
            return self._create_test_examples(
                self._read_tsv(os.path.join(data_dir, "dev" + str(k) + ".csv")), logger)
        else:
            return self._create_test_examples(
                self._read_tsv(os.path.join(data_dir, "val.csv")), logger)

    def get_test_examples(self, data_dir, logger, k=None):
        """See base class."""
        if k is not None:
            return self._create_test_examples(
                self._read_tsv(os.path.join(data_dir, "test" + str(k) + ".csv")), logger)
        else:
            return self._create_test_examples(
                self._read_tsv(os.path.join(data_dir, "test.csv")), logger)

    def _read_tsv(self, input_file, quotechar='"'):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = csv.reader(f)
            next(lines)
            new_lines = []
            for i, line in enumerate(lines):
                new_lines.append(line)
            return new_lines
    def get_dev_POS(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            data = self._read_tsv(os.path.join(data_dir, "dev" + str(k) + ".csv"))
        else:
            data = self._read_tsv(os.path.join(data_dir, "val.csv"))
        POS_list = []
        for i, line in enumerate(data):

            POS_source_label = self.pos_list[line[4]]
            POS_list.append(POS_source_label)
        
        return POS_list

    def get_test_POS(self, data_dir, k=None):
        """See base class."""
        if k is not None:
            data = self._read_tsv(os.path.join(data_dir, "test" + str(k) + ".csv"))
        else:
            data = self._read_tsv(os.path.join(data_dir, "test.csv"))
        POS_list = []
        for i, line in enumerate(data):

            POS_source_label = self.pos_list[line[4]]
            POS_list.append(POS_source_label)
        
        return POS_list
    def _create_train_examples(self, data, logger):
        """Creates examples for the training and dev sets."""
        examples = []
        # head = ['sentence','label','core_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser', 'domain_role_dict']

        for i, line in enumerate(data):
            sentence = line[0]
            label = int(line[1])
            core_position = int(line[2])
            verb_parser = eval(line[13])
            domain_role_dict = eval(line[15])


            examples.append(
                InputExampleUnified_Basic(
                    sentence=sentence,
                    label=label,
                    core_position=core_position,
                    verb_parser=verb_parser,
                    domain_role_dict=domain_role_dict,
                )
                )
        logger.info(f"train samples number:{len(examples)}")       
        return examples

    def _create_test_examples(self, data, logger):
        """Creates examples for the training and dev sets."""
        examples = []
        # head = ['sentence','label','core_position','target_word','pos_tag','gloss','eg_sent','bis_def_collins','exps_collins','bis_def_longman','exps_longman','bis_def_oxford_advanced','exps_oxford_advanced', 'verb_parser', 'examples_parser', 'domain_role_dict']

        for i, line in enumerate(data):
            sentence = line[0]
            label = int(line[1])
            core_position = int(line[2])
            verb_parser = eval(line[13])
            domain_role_dict = eval(line[15])


            examples.append(
                InputExampleUnified_Basic(
                    sentence=sentence,
                    label=label,
                    core_position=core_position,
                    verb_parser=verb_parser,
                    domain_role_dict=domain_role_dict,
                )
                )
        logger.info(f"test samples number:{len(examples)}")       
        return examples



def load_train_data_basic(args, logger, processor, task_name, tokenizer, k=None):

    if task_name in ['MOH-X', 'TroFi','LCC_Verb']:
        train_examples = processor.get_train_examples(args.data_dir, logger, k)
    elif task_name in ['VUA_Verb']:
        train_examples = processor.get_train_examples(args.data_dir, logger)
    else:
        raise ("task_name is wrong")
    
    train_features = convert_examples_to_features_dict[args.model](
            train_examples, tokenizer, logger, args) 
    train_dataloader = convert_features_dict[args.model](args, train_features, 'train')

    return train_dataloader

def load_test_data_basic(args, logger, processor, task_name, tokenizer, mode, k=None):
    if task_name in ['MOH-X', 'TroFi','LCC_Verb']:
        train_examples = processor.get_test_examples(args.data_dir, logger, k)
    elif task_name in ['VUA_Verb']:
        if mode == 'dev':
            train_examples = processor.get_dev_examples(args.data_dir, logger)
        else:
            train_examples = processor.get_test_examples(args.data_dir, logger)
    else:
        raise ("task_name is wrong")
    
    train_features = convert_examples_to_features_dict[args.model](
            train_examples, tokenizer, logger, args) 
    train_dataloader = convert_features_dict[args.model](args, train_features, 'train')

    return train_dataloader
    



def convert_examples_to_features_Basic_add_wp(examples, tokenizer, logger, args):
    features = []
    max_seq_length = args.max_seq_length

    for (ex_index, example) in enumerate(examples):

        label = example.label
        words = example.sentence.lower().split(' ')
        
        core_position = example.core_position

        words_lemma = []
        for word in words:
            words_lemma.append(wnl.lemmatize(word, 'n'))
        

        final_words_lemma = words_lemma
        domain_role_dict = example.domain_role_dict

        if domain_role_dict !={}:
            target_index_roles_dict = {}



            final_core_position = core_position
            # target words index 用key_role划分
            for index_key_rol, key_role in enumerate(domain_role_dict.keys()):
                target_role_tokens = domain_role_dict[key_role]['target_tokens']
                target_index_roles = []
                
                flag = False
                for target_role_i, target_role_token_span in enumerate(target_role_tokens):
                    target_role_tokens = target_role_token_span.split('_')
                    target_role_tokens_lemma = []
                    for target_role_token in target_role_tokens:
                        target_role_tokens_lemma.append(wnl.lemmatize(target_role_token, 'n')) 
                    flag = False

                    for index_target_role in range(len(final_words_lemma) - len(target_role_tokens) + 1):
                        if final_words_lemma[index_target_role: index_target_role + len(target_role_tokens)] == target_role_tokens_lemma:
                            flag = True
                            target_index_roles.append(index_target_role)
                            break
                    if flag:
                        continue
                    for index_target_role in range(len(final_words_lemma)):
                        for index_target_role_next in range(1, len(final_words_lemma)-index_target_role+1):
                            if ''.join(final_words_lemma[index_target_role: index_target_role + index_target_role_next]) == ''.join(target_role_tokens_lemma):
                                flag = True
                                for index_target_role_i in range(index_target_role, index_target_role + index_target_role_next):
                                    target_index_roles.append(index_target_role_i)
                                break

                    if not flag: #  # ['ass'] 会被 lemma 为['as'] 所以后面再补一个if not flag
                        for index_target_role in range(len(final_words_lemma)):
                            for index_target_role_next in range(1, len(final_words_lemma)-index_target_role+1):
                                if ''.join(final_words_lemma[index_target_role: index_target_role + index_target_role_next]) == ''.join(target_role_tokens):
                                    flag = True
                                    for index_target_role_i in range(index_target_role, index_target_role + index_target_role_next):
                                        target_index_roles.append(index_target_role_i)
                                    break 

                    assert flag == True
                    if not flag:
                        print(target_role_token_span, example.sentence, words)
                    
                target_index_roles_dict[key_role] = target_index_roles
            

            
            tokens_ori = tokenizer.tokenize(tokenizer.cls_token + ' ' + ' '.join(words))
            



            # 寻找各种index
            core_word_tokens = tokenizer.tokenize(' ' + words[core_position]) 

            if final_core_position > 0:
                tokens_before_core_tokens = tokenizer.tokenize(tokenizer.cls_token + ' ' + ' '.join(words[:final_core_position]))
            else:
                tokens_before_core_tokens = tokenizer.tokenize(tokenizer.cls_token)

            core_word_all_indexs = []
            for target_i in range(len(tokens_before_core_tokens)):
                core_word_all_indexs.append(0) 
            if args.word_emb_type == 'mean':
                for target_i in range(len(core_word_tokens)):
                    core_word_all_indexs.append(1)
            elif args.word_emb_type == 'first':
                for target_i in range(len(core_word_tokens)):
                    if target_i == 0:
                        core_word_all_indexs.append(1)
                    else:
                        core_word_all_indexs.append(0)
            else:
                print('no such word_emb_type!')
                exit()
            if len(tokens_ori) - len(tokens_before_core_tokens) - len(core_word_tokens) > 0:
                for target_i in range(len(tokens_ori) - len(tokens_before_core_tokens) - len(core_word_tokens)):
                    core_word_all_indexs.append(0)
            assert len(core_word_all_indexs) == len(tokens_ori)
            core_word_all_indexs += [0] * (args.max_seq_length - len(core_word_all_indexs))

            
            target_words_all_indexes_list = []
            core_words_all_indexes_list = []
            target_source_scores = []
            
            count_exist_tokens = len(tokens_ori)
            tokens_ori_add_word_pairs_tokens = tokens_ori.copy()
            for key_role in target_index_roles_dict.keys():
                target_indexes = target_index_roles_dict[key_role]
                for target_word_index in target_indexes:
                    target_word = final_words_lemma[target_word_index]
                    target_word_tokens = tokenizer.tokenize(' ' + target_word)
                    target_word_all_indexes = []
                    core_word_all_indexes = []
                    if target_word_index > final_core_position:
                        tokens_ori_add_word_pairs_tokens.extend([tokenizer.sep_token] + core_word_tokens + target_word_tokens)
                        target_word_all_indexes.extend(count_exist_tokens * [0] + [0] + [0] * len(core_word_tokens) + [1]*len(target_word_tokens))
                        core_word_all_indexes.extend(count_exist_tokens * [0] + [0] + [1] * len(core_word_tokens) + [0]*len(target_word_tokens))
                    else:
                        tokens_ori_add_word_pairs_tokens.extend([tokenizer.sep_token] + target_word_tokens + core_word_tokens)
                        target_word_all_indexes.extend(count_exist_tokens * [0] + [0] + [1] * len(target_word_tokens) + [0]*len(core_word_tokens))
                        core_word_all_indexes.extend(count_exist_tokens * [0] + [0] + [0] * len(target_word_tokens) + [1]*len(core_word_tokens))
                    count_exist_tokens += len([0] + [0] * len(core_word_tokens) + [1]*len(target_word_tokens))

                    

                    target_word_all_indexes += [0] * (args.max_seq_length - len(target_word_all_indexes))
                    core_word_all_indexes += [0] * (args.max_seq_length - len(core_word_all_indexes))
                    target_words_all_indexes_list.append(target_word_all_indexes)
                    core_words_all_indexes_list.append(core_word_all_indexes)
                target_source_scores_ori = 1/np.array(domain_role_dict[key_role]['target_source_scores'])

                target_source_scores.extend(list(target_source_scores_ori))
            
            if len(target_words_all_indexes_list) > args.word_pair_max:
                target_words_all_indexes_list = target_words_all_indexes_list[:args.word_pair_max]
                core_words_all_indexes_list = core_words_all_indexes_list[:args.word_pair_max]
            
            padding_target_words_all_indexes_list = [[0] * args.max_seq_length] * (args.word_pair_max - len(target_words_all_indexes_list))
            target_words_all_indexes_list += padding_target_words_all_indexes_list
            core_words_all_indexes_list += padding_target_words_all_indexes_list

            if len(target_source_scores) > args.word_pair_max:
                target_source_scores = target_source_scores[:args.word_pair_max]
            
            padding_target_source_scores = [0] * (args.word_pair_max - len(target_source_scores))
            softmax_target_source_scores = list(np.exp(target_source_scores)/sum(np.exp(target_source_scores)))
            softmax_target_source_scores += padding_target_source_scores

            if len(tokens_ori_add_word_pairs_tokens) > max_seq_length:
                print("raise the length of max_length")
                logger.info(f"raise the length of max_length over{len(tokens_ori)}")
                tokens_ori_add_word_pairs_tokens = tokens_ori_add_word_pairs_tokens[: max_seq_length]


            input_ids_ori = tokenizer.convert_tokens_to_ids(tokens_ori_add_word_pairs_tokens)
            input_mask_ori = [1] * len(input_ids_ori)

            # Zero-pad up to the sequence length.
            padding_metaphor = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
                max_seq_length - len(input_ids_ori)
            )

            input_ids_ori += padding_metaphor
            input_mask_ori += [0] * len(padding_metaphor)

            assert len(input_ids_ori) == max_seq_length
            assert len(input_mask_ori) == max_seq_length

        else:
            tokens_ori = tokenizer.tokenize(tokenizer.cls_token + ' ' + ' '.join(words))
            
            if len(tokens_ori) > max_seq_length:
                print("raise the length of max_length")
                logger.info(f"raise the length of max_length over{len(tokens_ori)}")
                tokens_ori = tokens_ori[: max_seq_length]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_ids_ori = tokenizer.convert_tokens_to_ids(tokens_ori)
            input_mask_ori = [1] * len(input_ids_ori)

            # Zero-pad up to the sequence length.
            padding_metaphor = [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)] * (
                max_seq_length - len(input_ids_ori)
            )

            input_ids_ori += padding_metaphor
            input_mask_ori += [0] * len(padding_metaphor)

            assert len(input_ids_ori) == max_seq_length
            assert len(input_mask_ori) == max_seq_length
            # 寻找各种index

            core_word = words[core_position]

            # core_word index
            core_word_tokens = tokenizer.tokenize(' ' + core_word) 
            if core_position >  0:
                tokens_before_core_tokens = tokenizer.tokenize(tokenizer.cls_token + ' ' + ' '.join(words[:core_position]))
            else:
                tokens_before_core_tokens = tokenizer.tokenize(tokenizer.cls_token)
            core_word_all_indexs = []
            for target_i in range(len(tokens_before_core_tokens)):
                core_word_all_indexs.append(0) 
            if args.word_emb_type == 'mean':
                for target_i in range(len(core_word_tokens)):
                    core_word_all_indexs.append(1)
            elif args.word_emb_type == 'first':
                for target_i in range(len(core_word_tokens)):
                    if target_i == 0:
                        core_word_all_indexs.append(1)
                    else:
                        core_word_all_indexs.append(0)
            else:
                print('no such word_emb_type!')
                exit()
            if len(tokens_ori) - len(tokens_before_core_tokens) - len(core_word_tokens) > 0:
                for target_i in range(len(tokens_ori) - len(tokens_before_core_tokens) - len(core_word_tokens)):
                    core_word_all_indexs.append(0)
            assert len(core_word_all_indexs) == len(tokens_ori)
            core_word_all_indexs += [0] * len(padding_metaphor)

            target_words_all_indexes_list = []
            core_words_all_indexes_list = []
            
            padding_target_words_all_indexes_list = [[0] * args.max_seq_length] * args.word_pair_max 
            target_words_all_indexes_list += padding_target_words_all_indexes_list
            core_words_all_indexes_list = target_words_all_indexes_list.copy()

            softmax_target_source_scores = []
            padding_target_source_scores = [0] * args.word_pair_max 
            softmax_target_source_scores += padding_target_source_scores

        features.append(
            InputFeatures_Basic_wp(
                input_ids_ori=input_ids_ori, 
                input_mask_ori=input_mask_ori,
                label=label,
                core_word_all_indexs=core_word_all_indexs,
                target_words_all_indexes_list=target_words_all_indexes_list,
                core_words_all_indexes_list=core_words_all_indexes_list,
                target_source_scores=softmax_target_source_scores,

            )
        )
    return features




def convert_features_Basic_wp(args, train_features, mode='train'):   
    
    all_input_ids_ori = torch.tensor(np.array([f.input_ids_ori for f in train_features]), dtype=torch.long)
    all_input_mask_ori = torch.tensor(np.array([f.input_mask_ori for f in train_features]), dtype=torch.long)
    all_label = torch.tensor(np.array([f.label for f in train_features]), dtype=torch.long)
    all_core_word_all_indexs = torch.tensor(np.array([f.core_word_all_indexs for f in train_features]), dtype=torch.long)
    all_target_words_all_indexes_list = torch.tensor(np.array([f.target_words_all_indexes_list for f in train_features]), dtype=torch.long)
    all_core_words_all_indexes_list = torch.tensor(np.array([f.core_words_all_indexes_list for f in train_features]), dtype=torch.long)
    all_target_source_scores = torch.tensor(np.array([f.target_source_scores for f in train_features]), dtype=torch.float)

    train_data = TensorDataset(
        all_input_ids_ori,
        all_input_mask_ori,
        all_label,
        all_core_word_all_indexs,
        all_target_words_all_indexes_list,
        all_core_words_all_indexes_list,
        all_target_source_scores,
    )

    if mode == "train":
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size
        )
    else:
        train_dataloader = DataLoader(
            train_data, batch_size=args.eval_batch_size
        )
    return train_dataloader


processors ={

    "WPDM":DataProcessorBasic,

}

load_train_processors = {

    "WPDM":load_train_data_basic,

}

load_test_processors = {

    "WPDM":load_test_data_basic,

}


convert_examples_to_features_dict = {
    "WPDM":convert_examples_to_features_Basic_add_wp,
}


convert_features_dict = {

    "WPDM":convert_features_Basic_wp,

}