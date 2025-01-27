import torch
from accelerate import Accelerator
from datasets import load_dataset
from datasets import Dataset as HF_Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from utils import *
import argparse
from liger_kernel.transformers import apply_liger_kernel_to_llama

import os
os.environ["TOKENIZERS_PARALLELISM"]="true"

gpu_name = torch.cuda.get_device_name(0)
if "4090" in gpu_name:
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

"""
Usage:

CUDA_VISIBLE_DEVICES="2,3,4,6"  torchrun --nproc_per_node=4 train.py 
"""

def main(args):
    
    set_seed(args.seed)
    model_id = args.path_to_sft_save_dir+"/"+args.experiment_name
    
    if Accelerator().is_main_process:
        ensure_dir(model_id)
        with open(os.path.join(model_id, 'training_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    if args.indexing:
        tokenizer.add_tokens(get_new_tokens(args))
        print(f"New tokens added: {len(get_new_tokens(args))}")
    
    train_data, valid_data = load_datasets(args)
    # 因为sfttrainer里有input_ids字段，会冲突，得改名
    if args.indexing:
        postfix = tokenizer.eos_token
    else:
        postfix = ". "+tokenizer.eos_token
    valid_data = [{"text": valid_data[i]["labels"]+postfix} for i in range(len(valid_data))]
    valid_data = HF_Dataset.from_list(valid_data)
    train_data = [{"text": train_data[i]["labels"]+postfix} for i in range(len(train_data))]
    train_data = HF_Dataset.from_list(train_data)
    if Accelerator().is_main_process:
        print("num of train data:", len(train_data))
        random_indices = torch.randperm(len(train_data))[:10].tolist()
        for idx in random_indices:
            print(f"Random train example {idx}:\n", train_data[idx])
    
    # response_template_with_context = "\n### Response:\n"
    response_template_with_context = " [/INST]"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    
    torch_dtype = torch.float16 if args.torch_dtype == "float16" else torch.bfloat16
    device_index = Accelerator().process_index
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        use_cache=False,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config if args.quantize else None,
        device_map={"": device_index},
        # trust_remote_code=True,
        # attn_implementation="eager"
    )
    
    apply_liger_kernel_to_llama()
    
    if args.indexing:
        model.resize_token_embeddings(len(tokenizer))
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=args.lora_modules_to_save.split(",") if args.indexing else None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model.gradient_checkpointing_enable()
    
    train_args = SFTConfig(
        gradient_checkpointing_kwargs={'use_reentrant': True},
        max_seq_length=args.cutoff_len,
        dataset_text_field="text",
        seed=args.seed,
        output_dir=model_id,
        eval_steps=args.save_and_eval_steps,
        save_steps=args.save_and_eval_steps,
        save_strategy="steps",
        eval_strategy="steps",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=torch_dtype == torch.float16,
        bf16=torch_dtype == torch.bfloat16,
        dataloader_num_workers=16,
        num_train_epochs=args.epochs,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,
        learning_rate=args.learning_rate,
        logging_steps=5,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        weight_decay=args.weight_decay,
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
        
    trainer.train()

    trainer.save_model(model_id)
    trainer.create_model_card()
    tokenizer.save_pretrained(model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TokenMob train")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    
    args = parser.parse_args()
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
    if isinstance(args.sft_json_output, str):
        args.sft_json_output = args.sft_json_output.lower() == "true"
    if isinstance(args.multi_rec, str):
        args.multi_rec = args.multi_rec.lower() == "true"
    if isinstance(args.single_rec, str):
        args.single_rec = args.single_rec.lower() == "true"
    
    
    main(args)
