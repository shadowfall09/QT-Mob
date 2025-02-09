import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import *
from collator import TestCollator
from prompt_mobility import all_prompt
from evaluate import get_topk_results, get_metrics_results
from peft import PeftConfig, PeftModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test(args):
    if isinstance(args.quantize, str):
        args.quantize = args.quantize.lower() == "true"
    if isinstance(args.indexing, str):
        args.indexing = args.indexing.lower() == "true"
    if isinstance(args.multi_seq, str):
        args.multi_seq = args.multi_seq.lower() == "true"
    if isinstance(args.add_profile, str):
        args.add_profile = args.add_profile.lower() == "true"
    if isinstance(args.add_prefix, str):
        args.add_prefix = args.add_prefix.lower() == "true"
    if isinstance(args.filter_items, str):
        args.sft_json_output = args.sft_json_output.lower() == "true"
    if isinstance(args.multi_rec, str):
        args.multi_rec = args.multi_rec.lower() == "true"
    if isinstance(args.single_rec, str):
        args.single_rec = args.single_rec.lower() == "true"

    set_seed(args.seed)
    print(vars(args))
    
    if "3.2" in args.ckpt_path:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        
    with open(os.path.join(args.ckpt_path, 'testing_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device_map = "auto"
    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    device = torch.device("cuda")
    print("Loading model from: ", args.ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = 4096
    print("Use peft model with LoRA adapter") 
    peft_config = PeftConfig.from_pretrained(args.ckpt_path)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config if args.quantize else None,
        device_map=device_map,
        # trust_remote_code=True,
    )
    
    if args.indexing:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.ckpt_path)    
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if args.test_prompt_ids == "all":
        if args.test_task == "seq":
            prompt_ids = range(len(all_prompt["seq"]))
        else:
            prompt_ids = range(len(all_prompt["rec_single"]))
    else:
        prompt_ids = [int(_) for _ in args.test_prompt_ids.split(",")]

    test_data = load_test_dataset(args)
    collator = TestCollator(args, tokenizer) # collator是一个类，用于tokenize输入
    all_items = test_data.get_all_items()
    
    if args.indexing:
        print("Using indexing")
        prefix_allowed_tokens = test_data.get_prefix_allowed_tokens_fn(tokenizer, args.test_task.lower())

    print("Using Beam Search for evaluation")
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             shuffle=True, num_workers=4, pin_memory=True)
    
    if args.limit_test_size:
        print("Limit test size to 1000")

    model.eval()

    metrics = args.metrics.split(",")
    all_prompt_results = []
    with torch.no_grad():
        for prompt_id in prompt_ids: 

            test_loader.dataset.set_prompt(prompt_id)
            metrics_results = {}
            total = 0

            for _, batch in enumerate(tqdm(test_loader)):
                # batch是一个字典，包含input_ids和labels两个key，每个key对应的value是一个list，长度为batch_size
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)
                output = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=1,
                    top_k=50,
                    top_p=0.92,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens if args.indexing else None,
                    num_beams=args.num_beams, # 使用的是beam search，并非sampling，所以会输出num_beams个结果
                    num_return_sequences=args.num_beams,
                    output_scores=True, # 返回每个token的score
                    return_dict_in_generate=True,
                    early_stopping=True
                    )
                output_ids = output["sequences"] # shape torch.Size([batch_size * num_beams, seq_len])
                scores = output["sequences_scores"] # shape torch.Size([batch_size * num_beams])
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                ) # 一个list，长度为batch_size * num_beams
                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)  
            
                batch_metrics_res = get_metrics_results(topk_res, metrics)

                for m, res in batch_metrics_res.items():
                    if m not in metrics_results:
                        metrics_results[m] = res
                    else:
                        metrics_results[m] += res

                if total % 100 == 0:
                    temp={}
                    for m in metrics_results:
                        temp[m] = metrics_results[m] / total
                    print(temp)
                
                if args.limit_test_size and total >= 1000:
                    print("Limit test size to 1000")
                    break

            for m in metrics_results:
                metrics_results[m] = metrics_results[m] / total

            all_prompt_results.append(metrics_results)
            print("======================================================")
            print("Prompt {} results: ".format(prompt_id), metrics_results)
            print("======================================================")
            print("")

    if len(all_prompt_results) == 1:
        single_result = all_prompt_results[0]
        print("======================================================")
        print("Single prompt result: ", single_result)
        print("======================================================")
    
        save_data = {}
        save_data["test_task"] = args.test_task
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["single_result"] = single_result
    else:
        mean_results = {}
        min_results = {}
        max_results = {}

        for m in metrics:
            all_res = [_[m] for _ in all_prompt_results]
            mean_results[m] = sum(all_res) / len(all_res)
            min_results[m] = min(all_res)
            max_results[m] = max(all_res)
    
        print("======================================================")
        print("Mean results: ", mean_results)
        print("Min results: ", min_results)
        print("Max results: ", max_results)
        print("======================================================")
    
        save_data = {}
        save_data["test_task"] = args.test_task
        save_data["test_prompt_ids"] = args.test_prompt_ids
        save_data["mean_results"] = mean_results
        save_data["min_results"] = min_results
        save_data["max_results"] = max_results
        save_data["all_prompt_results"] = all_prompt_results

    with open(args.results_file, "w") as f:
        json.dump(save_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QT-Mob test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_test_args(parser)

    args = parser.parse_args()
    test(args)
