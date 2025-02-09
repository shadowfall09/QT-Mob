import os
CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

from utils import *
import argparse
import subprocess
from test import test
import torch


# 开关
TRAIN = True
TEST = True

PATH_TO_SFT_SAVE_DIR = "PATH_TO_SFT_SAVE_DIR"

def train_model_trl(args):
    # command = f"accelerate launch --num_processes={len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))} train.py \
    command = f"WORLD_SIZE={len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))} torchrun --nproc_per_node {len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))} --master_port 47769 train.py \
    --base_model {args.base_model} \
    --data_path {args.data_path} \
    --index_file {args.index_file} \
    --tasks {args.tasks} \
    --epochs {args.epochs} \
    --learning_rate {args.learning_rate} \
    --per_device_train_batch_size {args.per_device_train_batch_size} \
    --per_device_eval_batch_size {args.per_device_eval_batch_size} \
    --gradient_accumulation_steps {args.gradient_accumulation_steps} \
    --cutoff_len {args.cutoff_len} \
    --weight_decay {args.weight_decay} \
    --lora_r {args.lora_r} \
    --lora_alpha {args.lora_alpha} \
    --lora_dropout {args.lora_dropout} \
    --lora_target_modules {args.lora_target_modules} \
    --lora_modules_to_save {args.lora_modules_to_save} \
    --resume_from_checkpoint {args.resume_from_checkpoint} \
    --warmup_ratio {args.warmup_ratio} \
    --lr_scheduler_type {args.lr_scheduler_type} \
    --save_and_eval_steps {args.save_and_eval_steps} \
    --experiment_name {args.experiment_name} \
    --path_to_sft_save_dir {args.path_to_sft_save_dir} \
    --quantize {args.quantize} \
    --indexing {args.indexing} \
    --multi_seq {args.multi_seq} \
    --add_profile {args.add_profile} \
    --add_prefix {args.add_prefix} \
    --sft_json_output {args.sft_json_output}\
    --multi_rec {args.multi_rec}\
    --single_rec {args.single_rec}"
    print(command)
    subprocess.run(command, shell=True, check=True)
    

def test_model(args):
    args.ckpt_path = f"{args.path_to_sft_save_dir}/{args.experiment_name}"
    if args.test_task == "seq":
        args.results_file = f"{args.path_to_sft_save_dir}/{args.experiment_name}/test_results.json"
    elif args.test_task == "recovery":
        args.results_file = f"{args.path_to_sft_save_dir}/{args.experiment_name}/test_results_rec.json"
    test(args)
    
        

def choose_model(base_model):
    if base_model == "3.2":
        return "path to Llama-3.2-1B-Instruct"
    elif base_model == "3.1":
        return "path to Llama-3.1-8B-Instruct"
    elif base_model == "tiny":
        return "path to TinyLlama_v1.1"
    elif base_model == "qwen":
        return "path to Qwen2.5-1.5B-Instruct"
    elif base_model == "phi":
        return "path to phi-1_5"
    elif base_model == "olmo":
        return "path to OLMo-1B-0724-hf"
    else:
        raise NotImplementedError("Unknown base model.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QT-Mob')
    parser = parse_dataset_args(parser)
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    
    
    TEST_METRICS = "hit@1,hit@5,hit@10,ndcg@5,ndcg@10"
    args.path_to_sft_save_dir = PATH_TO_SFT_SAVE_DIR
    args.metrics = TEST_METRICS
    
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    BASE_MODEL = "3.2" # 3.2, 3.1, tiny, qwen, phi, olmo
    args.base_model = choose_model(BASE_MODEL)
    args.index_file = "index_"+args.base_model.split("/")[-1]+".json"
    DATASET_PATH = "./data"+"/fourquare_NYC"
    TRAIN_TASKS = ["seq", "recovery","index","location","trans"]
    TEST_TASK = "seq,recovery"
    CUSTOM_NAME = "NYC_latest"
    args.tasks = ",".join(TRAIN_TASKS)
    args.data_path = DATASET_PATH
    args.experiment_name = BASE_MODEL +"_"+CUSTOM_NAME
    args.indexing = "true"
    args.multi_seq = "true"
    args.add_profile = "true"
    args.multi_rec = "true"
    args.single_rec = "true"
    if TRAIN:
        print("Training model with TRL...")
        train_model_trl(args)
    if TEST:
        for task in TEST_TASK.split(","):
            args.test_task = task
            print("Testing "+task+"...")
            test_model(args)
    torch.cuda.empty_cache()
    
