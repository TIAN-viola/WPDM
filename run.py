#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import os
from utils import Config, Logger, make_log_dir, compute_metrics, set_random_seed
import random
import copy
from tqdm import tqdm, trange
import numpy as np
from collections import OrderedDict
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from data_loader import processors, load_train_processors, load_test_processors
from modeling import _models
import tensorboard_logger as tb_logger
from Optimizers import optimizers



CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "training_args.bin"


def main():
    argv = sys.argv[1:]
    if len(argv) > 0:
        cmd_arg = OrderedDict()
        argvs = " ".join(argv).split(" ")
        for i in range(0, len(argvs), 2):
            arg_name, arg_value = argvs[i], argvs[i + 1]
            arg_name = arg_name.strip("-")
            cmd_arg[arg_name] = arg_value
        config = Config(conf_file=cmd_arg["conf_file"])
        config.update_params(cmd_arg)
    else:
        print("no config file!")
        exit()
    args = config
    print(args.__dict__)
    args.update_params({"model_path":os.path.join(args.model_root_path, args.task_name, args.word_type)})
    args.update_params({"data_dir":os.path.join(args.root_data_dir, args.task_name)})
    # logger
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    log_dir = make_log_dir(os.path.join(args.model_path, args.bert_model.split('/')[-1]))
    logger = Logger(log_dir)
    args.update_params({"log_dir":log_dir})

    args.update_params({"predict_dir":os.path.join(log_dir, 'predict.npz')})
    args.update_params({"tb_logger_path":os.path.join(args.log_dir, 'runs')})

    os.makedirs(args.tb_logger_path)
    config.save(log_dir)

    # set CUDA devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logger.info("device: {} n_gpu: {}".format(device, args.n_gpu))
    tb_logger.configure(args.tb_logger_path, flush_secs=5)

    # set seed
    set_random_seed(args.seed, args.n_gpu)

    processor = processors[args.model](args)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # training and eval
    if args.do_train and (args.task_name == "MOH-X" or args.task_name == "TroFi" or args.task_name == 'LCC_Verb'):
        k_result = []
        for k in tqdm(range(args.kfold), desc="K-fold"):
            logger.info(f"-------------{k}-fold------------")
            train_dataloader = load_train_processors[args.model](
                args, logger, processor, args.task_name, tokenizer, k
            )

            if args.segment != "none":
                bert = load_pretrained_model(args)
                if 'bert_layer' in args.__dict__.keys():
                    model = _models[args.segment][args.model](args=args, bert=bert, bert_layer=args.bert_layer)
                else:
                    model = _models[args.segment][args.model](args=args, bert=bert)
            else:
                bert = AutoModel.from_pretrained(args.bert_model)
                model = _models[args.segment][args.model](args=args, bert=bert)

            if args.spec_tok:
                model.encoder.resize_token_embeddings(tokenizer.vocab_size + 4)
            model.to(args.device)
            if args.n_gpu > 1 :
                model = torch.nn.DataParallel(model)
            model, best_result = run_train_dict[args.segment][args.model](
                args,
                logger,
                model,
                train_dataloader,
                processor,
                args.task_name,
                tokenizer,
                k
            )
            k_result.append(best_result)

        # Calculate average result
        avg_result = copy.deepcopy(k_result[0])
        for result in k_result[1:]:
            for k, v in result.items():
                avg_result[k] += v
        for k, v in avg_result.items():
            avg_result[k] /= len(k_result)

        logger.info(f"-----Averge Result-----")
        for key in sorted(avg_result.keys()):
            logger.info(f"  {key} = {str(avg_result[key])}")
    elif args.do_train and (args.task_name == "VUA_Verb"):

        train_dataloader = load_train_processors[args.model](
            args, logger, processor, args.task_name, tokenizer
        )
        if args.segment != "none":
            bert = load_pretrained_model(args)
            if 'bert_layer' in args.__dict__.keys():
                model = _models[args.segment][args.model](args=args, bert=bert, bert_layer=args.bert_layer)
            else:
                model = _models[args.segment][args.model](args=args, bert=bert)
        else:
            bert = AutoModel.from_pretrained(args.bert_model)
            model = _models[args.segment][args.model](args=args, bert=bert)
            
        if args.spec_tok:
            model.encoder.resize_token_embeddings(tokenizer.vocab_size + 4)
        model.to(args.device)
        if args.n_gpu > 1 :
            model = torch.nn.DataParallel(model)
        model, best_result = run_train_dict[args.segment][args.model](
            args,
            logger,
            model,
            train_dataloader,
            processor,
            args.task_name,
            tokenizer
        )
        logger.info(f"-----Best Result-----")
        for key in sorted(best_result.keys()):
            logger.info(f"  {key} = {str(best_result[key])}")

    # just eval 
    if not args.do_train:
        if args.task_name in ['MOH']:
            print('please do 10-fold cross-validation on MOH!')
        else:
            if args.segment != "none":
                bert = load_pretrained_model(args)
                if 'bert_layer' in args.__dict__.keys():
                    model = _models[args.segment][args.model](args=args, bert=bert, bert_layer=args.bert_layer)
                else:
                    model = _models[args.segment][args.model](args=args, bert=bert)
            else:
                bert = AutoModel.from_pretrained(args.bert_model)
                model = _models[args.segment][args.model](args=args, bert=bert)
                
            if args.spec_tok:
                model.encoder.resize_token_embeddings(tokenizer.vocab_size + 4)
            model.to(args.device)
            model.load_state_dict(torch.load(args.model_file))
            model.eval()
            eval_dataloader = load_test_processors[args.model](
                args, logger, processor, args.task_name, tokenizer, 'test'
            )
            logger.info(f"------ test -------")
            POS_test_list = processor.get_test_POS(args.data_dir)
            result = run_eval_dict[args.segment][args.model](args, logger, model, eval_dataloader, POS_test_list)

def load_pretrained_model(args):
    bert = AutoModel.from_pretrained(args.bert_model)
    config = bert.config
    if args.type_vocab_size != 2:
        config.type_vocab_size = args.type_vocab_size # the number of types of segment ids
        if "albert" in args.bert_model:
            bert.embeddings.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.embedding_size
            )
        else:
            bert.embeddings.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )
        bert._init_weights(bert.embeddings.token_type_embeddings)
    return bert



def run_train_wp(
    args,
    logger,
    model,
    train_dataloader,
    processor,
    task_name,
    tokenizer,
    k=None,
):

    tr_loss = 0

    num_train_optimization_steps = len(train_dataloader) * args.num_train_epoch
    if 'module_lr' not in args.__dict__.keys():
        # Prepare optimizer, scheduler
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:

        param_optimizer = list(model.named_parameters())

        weight_decay_dict = {}
        learning_rate_dict = {}
        module_name_all = args.module_name.split(", ")
        for i, module_name in enumerate(module_name_all):
            weight_decay_dict[module_name] = args.module_weight_decay[i]
            learning_rate_dict[module_name] = args.module_lr[i]
            
        lr=args.learning_rate
        if "weight_decay" not in args.__dict__.keys():
            args.weight_decay=0
        weight_decay=args.weight_decay
        params=[]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        for n, p in param_optimizer: # p=bert1_text
            ppopt={'params': p}
            if n.split('.')[0] in module_name_all: 
                ppopt["lr"] = learning_rate_dict[n.split('.')[0]]
            elif n.split('.')[1] in module_name_all:
                ppopt["lr"] = learning_rate_dict[n.split('.')[1]]
            else:
                ppopt["lr"] = lr
            if any(nd in n for nd in no_decay):
                ppopt["weight_decay"] = 0
            else:
                if n.split('.')[0] in module_name_all:
                    ppopt["weight_decay"] = weight_decay_dict[n.split('.')[0]]  
                elif n.split('.')[1] in module_name_all:
                    ppopt["weight_decay"] = weight_decay_dict[n.split('.')[1]]  
                else:
                    ppopt["weight_decay"] = weight_decay          
            params.append(ppopt)

        optimizer = optimizers[args.optimizer](params=params,lr=lr,weight_decay=weight_decay)
    
    if args.lr_schedule != False and args.lr_schedule.lower() != "none":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_epoch * len(train_dataloader)),
            num_training_steps=num_train_optimization_steps,
        )

    logger.info("***** Running training *****")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Num steps = { num_train_optimization_steps}")

    # Run training
    model.train()
    max_val_f1 = -1
    max_result = {}

    for epoch in trange(int(args.num_train_epoch), desc="Epoch"):
        tr_loss = 0

        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # move batch data to gpu
            batch = tuple(t.to(args.device) for t in batch)

            (
                input_ids_ori,
                input_mask_ori,
                label,
                core_word_all_indexs,
                target_words_all_indexes_list,
                core_words_all_indexes_list,
                target_source_scores,
            ) = batch


            # compute loss values
            logits = model( 
                input_ids_ori,
                input_mask_ori,
                core_word_all_indexs,
                target_words_all_indexes_list,
                core_words_all_indexes_list,
                target_source_scores,
            )
            # cls loss
            if 'class_weight' in list(args.__dict__.keys()):
                loss_cls_fuc = nn.CrossEntropyLoss(reduction='sum', weight=torch.Tensor([1, args.class_weight]).cuda())
            else:
                loss_cls_fuc = nn.CrossEntropyLoss(reduction='sum')
            loss_cls = loss_cls_fuc(logits, label)
            
            loss = loss_cls 

            # average loss if on multi-gpu.
            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.lr_schedule != False and args.lr_schedule.lower() != "none":
                scheduler.step()

            optimizer.zero_grad()

            tr_loss += loss.item()

        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"[epoch {epoch+1}] ,lr: {cur_lr} ,tr_loss: {tr_loss}")
        tb_logger.log_value(str(k) + '_'+'train_loss', tr_loss, step=epoch)

        # evaluate
        test_label = False
        if args.do_eval and task_name not in ['MOH-X', 'TroFi', 'LCC_Verb']:
            eval_dataloader = load_test_processors[args.model](
                args, logger, processor, task_name, tokenizer, 'dev',  k
            )
            logger.info(f"------ dev -------")
            result = run_eval_dict[args.segment][args.model](args, logger, model, eval_dataloader)
            for key in sorted(result.keys()):
                tb_logger.log_value(str(k) + '_'+'dev_' + key, result[key], step=epoch)
            # update
            if result["f1"] > max_val_f1:
                max_val_f1 = result["f1"]
                max_result = result
                test_label = True
                if args.save_best_model:
                    save_model(args, model, tokenizer)
        if args.do_test:
            eval_dataloader = load_test_processors[args.model](
                args, logger, processor, task_name, tokenizer, 'test',  k
            )
            logger.info(f"------ test -------")
            if test_label:
    
                result = run_eval_dict[args.segment][args.model](args, logger, model, eval_dataloader, test_path=args.predict_dir)
            else:
                result = run_eval_dict[args.segment][args.model](args, logger, model, eval_dataloader)
            for key in sorted(result.keys()):
                tb_logger.log_value(str(k) + '_'+'test_' + key, result[key], step=epoch)
            if args.task_name in ['MOH-X', 'TroFi', 'LCC_Verb']:
                # update
                if result["f1"] > max_val_f1:
                    max_val_f1 = result["f1"]
                    max_result = result
                    result = run_eval_dict[args.segment][args.model](args, logger, model, eval_dataloader, test_path=args.predict_dir[:-4] + str(k) +'.npz')
                    if args.save_best_model:
                        save_model(args, model, tokenizer)



    logger.info(f"-----Best Result-----")
    for key in sorted(max_result.keys()):
        logger.info(f"  {key} = {str(max_result[key])}")

    return model, max_result



def run_eval_wp(args, logger, model, eval_dataloader, POS_list=None, test_path=None):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    pred_guids = []
    out_label_ids = None
    bert = AutoModel.from_pretrained(args.bert_model)
    bert.to(args.device)

    bert.eval()
    for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_batch = tuple(t.to(args.device) for t in eval_batch)

        (
            input_ids_ori,
            input_mask_ori,
            label,
            core_word_all_indexs,
            target_words_all_indexes_list,
            core_words_all_indexes_list,
            target_source_scores,
        ) = eval_batch

        with torch.no_grad():
            # compute loss values

            logits = model( 
                input_ids_ori,
                input_mask_ori,
                core_word_all_indexs,
                target_words_all_indexes_list,
                core_words_all_indexes_list,
                target_source_scores,
            )
            # cls loss
            if 'class_weight' in list(args.__dict__.keys()):
                loss_cls_fuc = nn.CrossEntropyLoss(reduction='sum', weight=torch.Tensor([1, args.class_weight]).cuda())
            else:
                loss_cls_fuc = nn.CrossEntropyLoss(reduction='sum')
            loss_cls = loss_cls_fuc(logits, label)
            
                
            eval_loss += loss_cls.mean().item()
            nb_eval_steps += 1

            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())

                out_label_ids = label.detach().cpu().numpy()
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                out_label_ids = np.append(
                    out_label_ids, label.detach().cpu().numpy(), axis=0
                )

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    # compute metrics
    result = compute_metrics(preds, out_label_ids, args.average)
    if test_path:
        np.savez(test_path, y_pred=preds, y_true=out_label_ids)


    for key in sorted(result.keys()):
        logger.info(f"  {key} = {str(result[key])}")

    
    return result


def save_model(args, model, tokenizer):
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.log_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.log_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.encoder.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.log_dir)

    # Good practice: save your training arguments together with the trained model
    output_args_file = os.path.join(args.log_dir, ARGS_NAME)
    torch.save(args, output_args_file)

run_train_dict = {
    'none': {

        "WPDM":run_train_wp,

        
    },

}

run_eval_dict = {
    'none': {

        "WPDM":run_eval_wp,

        
    },

}

if __name__ == "__main__":
    main()