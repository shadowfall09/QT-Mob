import json
import os
import random
import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from data_mobility import *


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--base_model", type=str,
                        default="path to Llama-3.1-8B-Instruct",
                        help="basic model path")
    parser.add_argument("--quantize", type=str, default="true", help="whether to quantize the model")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", help="the dtype of the model")
    return parser

def parse_dataset_args(parser):
    parser.add_argument("--data_path", type=str, default="./data/foursquare_NYC",
                        help="data directory")
    parser.add_argument("--data_filename", type=str, default="trips.csv",help="data filename")
    parser.add_argument("--tasks", type=str, default="index",
                        help="Downstream tasks, separate by comma")
    parser.add_argument("--index_file", type=str, default="location.index.json", help="the item indices file, not path")
    # arguments related to sequential task
    parser.add_argument("--max_his_len", type=int, default=20,
                        help="the max number of location in history trajectory, -1 means no limit")
    parser.add_argument("--add_prefix",  type=str, default="false",
                        help="whether add sequential prefix in history")
    parser.add_argument("--his_sep", type=str, default=" ", help="The separator used for history")
    parser.add_argument("--sft_json_output", type=str,default="false", help="whether to output json file for sft")
    parser.add_argument("--indexing", type=str,default="true", help="whether to index the location")
    parser.add_argument("--multi_seq", type=str,default="true", help="whether to generate multiple trajectories")
    parser.add_argument("--add_profile", type=str,default="false", help="whether to add user profile")
    parser.add_argument("--multi_rec",  type=str, default="false", help="whether to use  multi mode for recovery task")
    parser.add_argument("--single_rec", type=str, default="false", help="whether to use single mode for recovery task")
    parser.add_argument("--ablation_location_prompt", type=str, default="1", help="ablation rows of location prompt")
    return parser

def parse_train_args(parser):

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--weight_decay", type=float, default=0.001)

    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.002)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj", help="separate by comma") # q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj
    parser.add_argument("--lora_modules_to_save", type=str,
                        default="embed_tokens,lm_head", help="separate by comma")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="either training checkpoint or final adapter")

    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--save_and_eval_steps", type=int, default=1000)
    parser.add_argument("--experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--path_to_sft_save_dir", type=str, help="The path to the save directory")

    return parser

def parse_test_args(parser):

    parser.add_argument("--ckpt_path", type=str,
                        default="",
                        help="The checkpoint path")
    parser.add_argument("--filter_items",  default=False,
                        help="whether filter illegal items")
    parser.add_argument("--results_file", type=str,
                        default="./results/test-ddp.json",
                        help="result output path")
    parser.add_argument("--test_batch_size", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=15)
    parser.add_argument("--test_prompt_ids", type=str, default="0",
                        help="test prompt ids, separate by comma. 'all' represents using all")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--test_task", type=str, default="recovery",
                        help="test task, one of [seq, recovery]")
    parser.add_argument("--limit_test_size",  default=False, help="whether to limit the test size to 1000")

    return parser


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    
    
def get_new_tokens(args):
    indices = load_json(os.path.join(args.data_path, args.index_file))
    new_tokens = set()
    for id in indices:
        for index in indices[id]:
            new_tokens.add(index)
    return list(sorted(new_tokens))


def load_datasets(args):
    set_seed(args.seed)
    tasks = args.tasks.split(",")

    train_datasets = []
    for task in tasks:
        if task.lower() == "seq":
            dataset = SeqDataset(args, mode="train")
        elif task.lower() == "recovery":
            dataset = RecoveryDataset(args, mode="train")
        elif task.lower() == "index":
            dataset = Index2LocationDataset(args)
        elif task.lower() == "location":
            dataset = Location2IndexDataset(args)
        elif task.lower() == "trans":
            dataset = TrajectoryTranslationDataset(args, mode="train")
        else:
            raise NotImplementedError
        train_datasets.append(dataset)

    train_data = ConcatDataset(train_datasets)
    
    valid_datasets = []
    for task in tasks:
        if task.lower() == "seq":
            dataset = SeqDataset(args, mode="valid")
        elif task.lower() == "recovery":
            dataset = RecoveryDataset(args, mode="valid")
        elif task.lower() == "trans":
            dataset = TrajectoryTranslationDataset(args, mode="valid")
        elif task.lower() == "index" or task.lower() == "location":
            continue
        else:
            raise NotImplementedError
        valid_datasets.append(dataset)
        
    if len(valid_datasets) > 0:
        valid_data = ConcatDataset(valid_datasets)
    else:
        valid_data = None

    return train_data, valid_data

def load_test_dataset(args):
    set_seed(args.seed)
    if args.test_task.lower() == "seq":
        test_data = SeqDataset(args, mode="test")
    elif args.test_task.lower() == "recovery":
        test_data = RecoveryDataset(args, mode="test")
    else:
        raise NotImplementedError

    return test_data

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file, indent=4):
    with open(file, 'w') as f:
        json.dump(data, f, indent=indent)