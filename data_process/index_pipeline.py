import os
import subprocess
import argparse
import datetime
import multiprocessing
import time


def generate_embedding(args):
    script_path = "location_prompt_emb.py"
    command = ["python", script_path, "--root", args.data_path, "--gpu_id", args.gpu_id, "--llm_path", args.llm_path]
    subprocess.run(command, check=True)
    
    
def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")
    return cur


def generate_quantization_model(args):
    num_emb_list = [args.codebook_dim] * args.index_len
    num_emb_list = " ".join([str(_) for _ in num_emb_list])
    sk_epsilons = [0.0] * args.index_len
    sk_epsilons[-1] = 0.003
    sk_epsilons = " ".join([str(_) for _ in sk_epsilons])
    data_path = os.path.join(args.data_path, f"location.emb-{args.llm_path.split('/')[-1]}-td.npy")
    ckpt_dir = "../ckpt"
    saved_model_dir = "{}".format(get_local_time())
    ckpt_dir = os.path.join(ckpt_dir, saved_model_dir)
    command = f"""cd ../index && python -u main.py \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 1024 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --dropout_prob 0.0 \
    --bn False \
    --e_dim 32 \
    --quant_loss_weight 1.0 \
    --beta 0.25 \
    --num_emb_list {num_emb_list} \
    --sk_epsilons {sk_epsilons} \
    --layers 2048 1024 512 256 128 64 \
    --device cuda:{args.gpu_id} \
    --data_path {data_path} \
    --ckpt_dir {ckpt_dir}"""
    subprocess.run(command, shell=True, check=True)
    return ckpt_dir
        
    
def generate_index(args, ckpt_dir):
    command = f"""cd ../index && python generate_indices.py \
    --output_dir {args.data_path} \
    --gpu_id {args.gpu_id} \
    --ckpt_path {os.path.join(ckpt_dir,"best_loss_model.pth")} \
    --output_file index_{args.llm_path.split('/')[-1]}_{args.index_len}_{args.codebook_dim}.json"""
    
    subprocess.run(command, shell=True, check=True)
    

def process_llm_path(llm_path, args):
    print(f"Processing {llm_path.split('/')[-1]}")
    args.llm_path = llm_path
    generate_embedding(args)
    ckpt_dir = generate_quantization_model(args)
    generate_index(args, ckpt_dir)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QT-Mob')
    parser.add_argument('--data_path', type=str, default='../data/foursquare_NYC')
    parser.add_argument('--gpu_id', type=str, default='3', help='gpu id')
    parser.add_argument('--llm_paths', type=str,default="path to Llama-3.2-1B-Instruct")
    parser.add_argument('--index_len', type=int, default=4)
    parser.add_argument('--codebook_dim', type=int, default=64)
    args = parser.parse_args()
    
    for llm_path in args.llm_paths.split(','):
        print(f"Generating embedding for {llm_path.split('/')[-1]}")
        args.llm_path = llm_path
        generate_embedding(args)
        
    processes = []
    args.gpu_id = args.gpu_id.split(",")[0]
    for llm_path in args.llm_paths.split(','):
        p = multiprocessing.Process(target=process_llm_path, args=(llm_path, args))
        p.start()
        processes.append(p)
        time.sleep(2)  # 每个进程启动之间添加1秒延迟

    for p in processes:
        p.join()
        
    print("All done!")